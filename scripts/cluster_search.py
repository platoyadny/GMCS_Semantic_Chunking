"""
cluster_search.py — Поиск по банку функциональности с кластеризацией и GPT-отбором.

Пайплайн:
  1. knn-поиск в OpenSearch с порогом min_cosine
  2. UMAP (понижение размерности) + HDBSCAN (кластеризация)
  3. Жёсткий фильтр кластеров: кластер проходит, если хотя бы один его документ
     входит в топ-N% пула ИЛИ в абсолютный топ-K пула
  4. Мягкое ранжирование прошедших кластеров по комбинированному скору
  5. Фильтр выбросов: выброс проходит, если входит в топ-M% пула ИЛИ в топ-K пула
  6. GPT-4o получает все документы выбранных кластеров + прошедшие выбросы
     и выбирает один лучший документ
  7. Fallback (нет кластеров): GPT получает выбросы, прошедшие фильтр

Использование:
  python cluster_search.py --min_cosine 0.60

  # Полный набор параметров:
  python cluster_search.py \
    --min_cosine 0.60 \
    --top_clusters 1 \
    --umap_components 50 \
    --umap_neighbors 15 \
    --hdbscan_min_cluster_size 5 \
    --hdbscan_min_samples 3 \
    --cluster_percentile 20 \
    --outlier_percentile 10 \
    --absolute_top_k 5 \
    --density_threshold 0.70 \
    --output_dir output

Переменные окружения (.env):
  OPENAI_API_KEY — ключ для доступа к GPT-4o.
"""

import json
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime


OPENSEARCH_URL      = "https://10.40.10.111:9200"
OPENSEARCH_USER     = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_BANK          = "processscout_bank"


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------

def load_env() -> str:
    """Загружает .env, возвращает OPENAI_API_KEY."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Ошибка: переменная OPENAI_API_KEY не найдена. "
              "Добавьте её в .env файл или в переменные окружения.")
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# Модель BGE-M3
# ---------------------------------------------------------------------------

def load_model():
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"  Модель загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def encode_query(model, text: str) -> list[float]:
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.tolist()


def encode_texts(model, texts: list[str]) -> np.ndarray:
    print(f"  Перекодирование {len(texts)} текстов через BGE-M3...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# OpenSearch
# ---------------------------------------------------------------------------

def create_client():
    from opensearchpy import OpenSearch
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
    )
    info = client.info()
    print(f"  OpenSearch: {info['version']['number']}, кластер: {info['cluster_name']}")
    return client


def run_search(client, query_embedding: list[float], min_cosine: float) -> list[dict]:
    """knn-поиск, фильтрация по min_cosine, сортировка по убыванию cosine_score."""
    body = {
        "size": 10000,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": 10000,
                }
            }
        }
    }
    response = client.search(index=INDEX_BANK, body=body)
    raw_hits = response["hits"]["hits"]
    total    = response["hits"]["total"]["value"]

    if total > len(raw_hits):
        print(
            f"  Предупреждение: найдено {total} документов, "
            f"возвращено {len(raw_hits)} (лимит size=10000)."
        )

    results = []
    for h in raw_hits:
        cosine = round(h["_score"] - 1, 6)
        if cosine < min_cosine:
            continue
        source = h["_source"]
        results.append({
            "id":                 h["_id"],
            "cosine_score":       cosine,
            "chunk_text_plain":   source.get("chunk_text_plain", ""),
            "chunk_text_context": source.get("chunk_text_context", ""),
            "metadata":           source.get("metadata", {}),
            "_embedding":         source.get("embedding", None),
        })

    results.sort(key=lambda x: x["cosine_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Эмбеддинги для кластеризации
# ---------------------------------------------------------------------------

def resolve_embeddings(hits: list[dict], model) -> np.ndarray:
    """Приоритет: _source.embedding → локальное перекодирование chunk_text_plain."""
    if hits[0].get("_embedding") is not None:
        print("  Векторы найдены в _source — используем напрямую.")
        matrix = np.array([h["_embedding"] for h in hits], dtype=np.float32)
    else:
        print("  Поле 'embedding' отсутствует в _source — перекодируем локально.")
        matrix = encode_texts(model, [h["chunk_text_plain"] for h in hits])
    print(f"  Матрица эмбеддингов: {matrix.shape}")
    return matrix


# ---------------------------------------------------------------------------
# Кластеризация
# ---------------------------------------------------------------------------

def run_clustering(embeddings: np.ndarray, args) -> np.ndarray:
    """UMAP → HDBSCAN. Возвращает массив меток (-1 = outlier)."""
    import umap
    import hdbscan as hdbscan_lib

    n            = len(embeddings)
    n_components = min(args.umap_components, n - 2)
    n_neighbors  = min(args.umap_neighbors, n - 1)

    print(f"\nUMAP: {embeddings.shape[1]}d → {n_components}d  "
          f"(metric=cosine, n_neighbors={n_neighbors})...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"  Готово. Форма редуцированного пространства: {reduced.shape}")

    min_cluster_size = min(args.hdbscan_min_cluster_size, n)
    min_samples      = min(args.hdbscan_min_samples, min_cluster_size)

    print(f"\nHDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))
    print(f"  Найдено кластеров: {n_clusters}  |  выбросов: {n_outliers}")
    return labels


# ---------------------------------------------------------------------------
# Пороговые утилиты
# ---------------------------------------------------------------------------

def compute_pool_thresholds(hits: list[dict], percentile: int, absolute_top_k: int) -> tuple[float, float]:
    """
    Возвращает (percentile_threshold, absolute_threshold):
      - percentile_threshold: минимальный cosine_score для попадания в топ-{percentile}%
      - absolute_threshold:   cosine_score документа на позиции absolute_top_k
    Документ проходит фильтр, если его cosine_score >= любого из двух порогов.
    """
    scores = [h["cosine_score"] for h in hits]
    # np.percentile(scores, 100-percentile) даёт нижнюю границу верхних percentile%
    percentile_threshold = float(np.percentile(scores, 100 - percentile))
    # Абсолютный топ-K: берём скор документа на позиции K-1 (список уже отсортирован убывающе)
    k_idx                = min(absolute_top_k - 1, len(hits) - 1)
    absolute_threshold   = hits[k_idx]["cosine_score"]
    return percentile_threshold, absolute_threshold


def passes_filter(score: float, percentile_threshold: float, absolute_threshold: float) -> bool:
    """Документ проходит фильтр, если score выше хотя бы одного из порогов."""
    return score >= percentile_threshold or score >= absolute_threshold


# ---------------------------------------------------------------------------
# Статистика и ранжирование кластеров
# ---------------------------------------------------------------------------

def compute_cluster_stats(
    hits:            list[dict],
    labels:          np.ndarray,
    embeddings:      np.ndarray,
    query_embedding: list[float],
    density_threshold: float,
) -> list[dict]:
    """
    Для каждого кластера вычисляет:
      centroid_cosine  — косинус запроса с нормализованным центроидом кластера
      top3_mean        — среднее cosine_score трёх лучших документов кластера
      density          — доля документов кластера с cosine_score >= density_threshold
      max_cosine_score — максимальный cosine_score в кластере
      combined_score   — 0.4*centroid_cosine + 0.4*top3_mean + 0.2*density
    """
    query_vec   = np.array(query_embedding, dtype=np.float32)
    cluster_ids = sorted(set(labels) - {-1})

    stats = []
    for cid in cluster_ids:
        indices = [i for i, l in enumerate(labels) if l == cid]
        vecs    = embeddings[indices]
        scores  = [hits[i]["cosine_score"] for i in indices]

        # Центроид
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        centroid_cosine = round(float(np.dot(query_vec, centroid)), 6)

        # Топ-3 mean
        top3_scores = sorted(scores, reverse=True)[:3]
        top3_mean   = round(float(np.mean(top3_scores)), 6)

        # Плотность релевантности
        density = round(sum(1 for s in scores if s >= density_threshold) / len(scores), 4)

        # Комбинированный скор
        combined_score = round(0.4 * centroid_cosine + 0.4 * top3_mean + 0.2 * density, 6)

        stats.append({
            "cluster_id":       cid,
            "centroid_cosine":  centroid_cosine,
            "top3_mean":        top3_mean,
            "density":          density,
            "combined_score":   combined_score,
            "max_cosine_score": round(max(scores), 6),
            "doc_indices":      indices,
            "size":             len(indices),
        })

    return stats


def filter_and_rank_clusters(
    stats:               list[dict],
    hits:                list[dict],
    percentile_threshold: float,
    absolute_threshold:   float,
    top_n:               int,
) -> list[dict]:
    """
    Жёсткий фильтр: кластер проходит, если хотя бы один его документ
    имеет cosine_score >= percentile_threshold ИЛИ >= absolute_threshold.
    Затем — мягкое ранжирование по combined_score, возвращаем top_n.
    """
    passed = []
    for s in stats:
        cluster_scores = [hits[i]["cosine_score"] for i in s["doc_indices"]]
        if any(passes_filter(sc, percentile_threshold, absolute_threshold) for sc in cluster_scores):
            passed.append(s)

    passed.sort(key=lambda s: s["combined_score"], reverse=True)
    return passed[:top_n]


# ---------------------------------------------------------------------------
# GPT-4o
# ---------------------------------------------------------------------------

def call_gpt(query_text: str, candidates: list[dict], openai_key: str) -> dict:
    """
    Передаёт запрос и список кандидатов (документы из кластеров + выбросы) GPT-4o.
    Возвращает {"best_id": "...", "explanation": "...", "source": "cluster|outlier"}.
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    candidates_text = ""
    for i, doc in enumerate(candidates, 1):
        source_label = "выброс" if doc.get("_is_outlier") else "кластер"
        candidates_text += (
            f"\n--- Документ {i} "
            f"(ID: {doc['id']}, cosine_score: {doc['cosine_score']}, источник: {source_label}) ---\n"
            f"{doc['chunk_text_plain']}\n"
        )

    prompt = f"""Ты — эксперт по анализу бизнес-процессов. Твоя задача — найти среди предложенных документов тот, который наиболее точно и полно соответствует данному запросу.

ЗАПРОС:
{query_text}

ДОКУМЕНТЫ-КАНДИДАТЫ:
{candidates_text}

Выбери один документ, который наилучшим образом отвечает на запрос. При выборе учитывай смысловое соответствие, а не только ключевые слова. Документы помечены как «кластер» (часть тематической группы) или «выброс» (не вписался в группу, но может быть точным совпадением).

Верни ответ строго в формате JSON (без markdown, без пояснений вне JSON):
{{
  "best_id": "<ID выбранного документа>",
  "explanation": "<краткое объяснение выбора на русском языке, 2-3 предложения>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"best_id": None, "explanation": f"Ошибка парсинга ответа GPT: {raw}"}


# ---------------------------------------------------------------------------
# Очистка документа
# ---------------------------------------------------------------------------

def clean_doc(hit: dict) -> dict:
    """Убирает служебные поля (_embedding, _is_outlier) из финального вывода."""
    return {k: v for k, v in hit.items() if k not in ("_embedding", "_is_outlier")}


# ---------------------------------------------------------------------------
# Сборка выходного документа
# ---------------------------------------------------------------------------

def build_output(
    hits:            list[dict],
    labels:          np.ndarray,
    embeddings:      np.ndarray,
    query_embedding: list[float],
    args,
    openai_key:      str,
    query_text:      str,
) -> dict:

    n_total = len(hits)

    # ---- Пороги для кластерного фильтра ----
    cluster_pct_threshold, cluster_abs_threshold = compute_pool_thresholds(
        hits,
        percentile=args.cluster_percentile,   # топ-N% пула для жёсткого фильтра кластеров
        absolute_top_k=args.absolute_top_k,   # абсолютный топ-K как второй критерий фильтра
    )

    # ---- Пороги для фильтра выбросов (строже) ----
    outlier_pct_threshold, outlier_abs_threshold = compute_pool_thresholds(
        hits,
        percentile=args.outlier_percentile,   # топ-N% пула для фильтра выбросов (строже чем кластерный)
        absolute_top_k=args.absolute_top_k,   # тот же абсолютный топ-K
    )

    # ---- Карта всех кластеров (только ID) ----
    cluster_ids_formed = sorted(set(labels) - {-1})
    clusters_formed: dict[str, list[str]] = {}
    for rank, cid in enumerate(cluster_ids_formed, 1):
        clusters_formed[f"cluster_{rank}"] = [
            hits[i]["id"] for i, l in enumerate(labels) if l == cid
        ]

    outlier_ids_all = [hits[i]["id"] for i, l in enumerate(labels) if l == -1]
    n_clusters      = len(cluster_ids_formed)

    # ---- Выбросы, прошедшие фильтр ----
    qualified_outliers = []
    for i, (hit, label) in enumerate(zip(hits, labels)):
        if label == -1 and passes_filter(
            hit["cosine_score"], outlier_pct_threshold, outlier_abs_threshold
        ):
            doc = dict(hit)
            doc["_is_outlier"] = True
            qualified_outliers.append(doc)

    print(f"\n  Выбросов всего: {len(outlier_ids_all)}, "
          f"прошли фильтр: {len(qualified_outliers)}")
    print(f"  Пороги кластерного фильтра:  "
          f"перцентиль={cluster_pct_threshold:.4f}, топ-{args.absolute_top_k}={cluster_abs_threshold:.4f}")
    print(f"  Пороги фильтра выбросов:     "
          f"перцентиль={outlier_pct_threshold:.4f}, топ-{args.absolute_top_k}={outlier_abs_threshold:.4f}")

    metadata = {
        "query_text":               query_text,
        "index":                    INDEX_BANK,
        "min_cosine":               args.min_cosine,
        "total_hits":               n_total,
        "top_clusters":             args.top_clusters,
        "umap_components":          args.umap_components,
        "umap_neighbors":           args.umap_neighbors,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples":      args.hdbscan_min_samples,
        "cluster_percentile":       args.cluster_percentile,
        "outlier_percentile":       args.outlier_percentile,
        "absolute_top_k":           args.absolute_top_k,
        "density_threshold":        args.density_threshold,
        "n_clusters_formed":        n_clusters,
        "clusters_formed":          clusters_formed,
        "outlier_ids_all":          outlier_ids_all,
        "outlier_ids_qualified":    [d["id"] for d in qualified_outliers],
        "timestamp":                datetime.now().isoformat(),
    }

    # ================================================================
    # FALLBACK: нет кластеров — GPT работает только с выбросами
    # ================================================================
    if n_clusters == 0:
        metadata["fallback"]        = True
        metadata["fallback_reason"] = (
            "Кластеров не сформировалось — HDBSCAN не нашёл плотных групп "
            "при заданных параметрах."
        )

        if not qualified_outliers:
            metadata["fallback_no_outliers"] = True
            return {
                "metadata":        metadata,
                "fallback_result": None,
            }

        print(f"\nGPT-4o (fallback): анализ {len(qualified_outliers)} выбросов...")
        gpt_result = call_gpt(query_text, qualified_outliers, openai_key)
        print(f"  Выбранный документ: {gpt_result.get('best_id')}")

        best_id  = gpt_result.get("best_id")
        best_doc = next(
            (clean_doc(d) for d in qualified_outliers if d["id"] == best_id),
            clean_doc(qualified_outliers[0]),
        )

        return {
            "metadata": metadata,
            "fallback_result": {
                "candidates": [clean_doc(d) for d in qualified_outliers],
                "best_document": {
                    **best_doc,
                    "gpt_explanation": gpt_result.get("explanation", ""),
                },
            },
        }

    # ================================================================
    # СТАНДАРТНЫЙ ПУТЬ: кластеры есть
    # ================================================================

    # Статистика и ранжирование кластеров
    stats  = compute_cluster_stats(
        hits, labels, embeddings, query_embedding,
        density_threshold=args.density_threshold,   # порог cosine_score для подсчёта density
    )
    ranked = filter_and_rank_clusters(
        stats, hits,
        percentile_threshold=cluster_pct_threshold,
        absolute_threshold=cluster_abs_threshold,
        top_n=args.top_clusters,                    # сколько лучших кластеров передавать GPT
    )

    if not ranked:
        metadata["no_clusters_passed_filter"] = True
        metadata["no_clusters_passed_filter_reason"] = (
            "Ни один кластер не прошёл жёсткий фильтр. "
            "Попробуйте снизить --cluster_percentile или --absolute_top_k."
        )

    output: dict = {"metadata": metadata}

    # ---- GPT-обработка каждого выбранного кластера ----
    for result_rank, cluster_stat in enumerate(ranked, 1):
        indices = cluster_stat["doc_indices"]

        # Все документы кластера + прошедшие выбросы
        cluster_docs = sorted(
            [hits[i] for i in indices],
            key=lambda x: x["cosine_score"],
            reverse=True,
        )
        candidates = [dict(d) for d in cluster_docs] + list(qualified_outliers)

        print(f"\nGPT-4o: result_cluster_{result_rank} "
              f"({len(cluster_docs)} из кластера + {len(qualified_outliers)} выбросов "
              f"= {len(candidates)} кандидатов)...")
        gpt_result = call_gpt(query_text, candidates, openai_key)
        print(f"  Выбранный документ: {gpt_result.get('best_id')}")

        best_id  = gpt_result.get("best_id")
        best_doc = next(
            (clean_doc(d) for d in candidates if d["id"] == best_id),
            clean_doc(candidates[0]),
        )
        is_outlier_win = best_id in [d["id"] for d in qualified_outliers]

        output[f"result_cluster_{result_rank}"] = {
            "cluster_rank":          result_rank,
            "centroid_cosine":       cluster_stat["centroid_cosine"],
            "top3_mean":             cluster_stat["top3_mean"],
            "density":               cluster_stat["density"],
            "combined_score":        cluster_stat["combined_score"],
            "max_cosine_in_cluster": cluster_stat["max_cosine_score"],
            "cluster_size":          cluster_stat["size"],
            "n_qualified_outliers":  len(qualified_outliers),
            "best_document_source":  "outlier" if is_outlier_win else "cluster",
            "cluster_docs":          [clean_doc(d) for d in cluster_docs],
            "qualified_outliers":    [clean_doc(d) for d in qualified_outliers],
            "best_document": {
                **best_doc,
                "gpt_explanation": gpt_result.get("explanation", ""),
            },
        }

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Поиск по банку функциональности с кластеризацией и GPT-отбором"
    )

    # ----- Поиск -----
    parser.add_argument(
        "--min_cosine", type=float, required=True,
        help="Минимальный порог cosine_score для запроса к OpenSearch. "
             "Определяет радиус поиска: чем ниже, тем больше документов вернётся."
    )

    # ----- Кластеризация -----
    parser.add_argument(
        "--umap_components", type=int, default=50,
        help="Целевая размерность после UMAP. Меньше = быстрее, но теряется информация. "
             "Рекомендуемый диапазон: 20–100. (по умолч.: 50)"
    )
    parser.add_argument(
        "--umap_neighbors", type=int, default=15,
        help="n_neighbors для UMAP. Меньше = акцент на локальную структуру, "
             "больше = глобальная. Рекомендуемый диапазон: 5–50. (по умолч.: 15)"
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size", type=int, default=5,
        help="Минимальный размер кластера в HDBSCAN. Главный параметр кластеризации: "
             "меньше = больше мелких кластеров, больше = крупные плотные кластеры. (по умолч.: 5)"
    )
    parser.add_argument(
        "--hdbscan_min_samples", type=int, default=3,
        help="min_samples для HDBSCAN. Влияет на устойчивость к шуму: "
             "больше = консервативнее, меньше выбросов. (по умолч.: 3)"
    )

    # ----- Отбор кластеров -----
    parser.add_argument(
        "--top_clusters", type=int, default=1,
        help="Сколько лучших кластеров передавать GPT. "
             "При значении 2 GPT делает 2 независимых вызова. (по умолч.: 1)"
    )
    parser.add_argument(
        "--cluster_percentile", type=int, default=20,
        help="Жёсткий фильтр кластеров: кластер проходит, если хотя бы один его документ "
             "входит в топ-N%% всего пула результатов. "
             "Служит для отсева слабо релевантных кластеров. (по умолч.: 20)"
    )
    parser.add_argument(
        "--absolute_top_k", type=int, default=5,
        help="Второй критерий жёсткого фильтра (для кластеров и выбросов): "
             "документ проходит, если входит в абсолютный топ-K пула. "
             "Страхует при малом числе результатов. (по умолч.: 5)"
    )
    parser.add_argument(
        "--density_threshold", type=float, default=0.70,
        help="Порог cosine_score для подсчёта density кластера. "
             "Density = доля документов кластера с cosine_score >= этого порога. "
             "Входит в комбинированный скор с весом 0.2. (по умолч.: 0.70)"
    )

    # ----- Фильтр выбросов -----
    parser.add_argument(
        "--outlier_percentile", type=int, default=10,
        help="Фильтр выбросов: выброс передаётся GPT, если входит в топ-N%% пула. "
             "Строже чем cluster_percentile — отсекает всё кроме очень сильных совпадений. (по умолч.: 10)"
    )

    # ----- Вывод -----
    parser.add_argument(
        "--output_dir", default="output",
        help="Директория для выходных файлов. (по умолч.: output)"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not (-1.0 <= args.min_cosine <= 1.0):
        print("Ошибка: --min_cosine должен быть в диапазоне [-1.0, 1.0]")
        sys.exit(1)
    if args.top_clusters < 1:
        print("Ошибка: --top_clusters должен быть >= 1")
        sys.exit(1)
    if not (0 < args.cluster_percentile <= 100):
        print("Ошибка: --cluster_percentile должен быть в диапазоне (0, 100]")
        sys.exit(1)
    if not (0 < args.outlier_percentile <= 100):
        print("Ошибка: --outlier_percentile должен быть в диапазоне (0, 100]")
        sys.exit(1)

    openai_key = load_env()

    print("\nВведите текст для поиска:")
    query_text = input("> ").strip()
    if not query_text:
        print("Ошибка: текст запроса не может быть пустым.")
        sys.exit(1)

    print(f"\nЗапрос:                    «{query_text}»")
    print(f"min_cosine:                {args.min_cosine}")
    print(f"top_clusters:              {args.top_clusters}")
    print(f"UMAP:                      {args.umap_components}d, n_neighbors={args.umap_neighbors}")
    print(f"HDBSCAN:                   min_cluster_size={args.hdbscan_min_cluster_size}, "
          f"min_samples={args.hdbscan_min_samples}")
    print(f"Фильтр кластеров:          топ-{args.cluster_percentile}% ИЛИ топ-{args.absolute_top_k}")
    print(f"Фильтр выбросов:           топ-{args.outlier_percentile}% ИЛИ топ-{args.absolute_top_k}")
    print(f"density_threshold:         {args.density_threshold}")

    model = load_model()

    print("\nПодключение к OpenSearch...")
    client = create_client()

    print("\nКодирование запроса...")
    query_embedding = encode_query(model, query_text)

    print(f"\nПоиск в {INDEX_BANK} (min_cosine={args.min_cosine})...")
    hits = run_search(client, query_embedding, args.min_cosine)
    print(f"  Документов после фильтрации: {len(hits)}")

    if not hits:
        print("\nНет результатов. Попробуйте снизить --min_cosine.")
        sys.exit(0)

    print("\nПодготовка эмбеддингов...")
    embeddings = resolve_embeddings(hits, model)

    if len(hits) >= 2:
        labels = run_clustering(embeddings, args)
    else:
        labels = np.array([-1] * len(hits))

    print("\nФормирование выходного документа...")
    output = build_output(
        hits, labels, embeddings, query_embedding,
        args, openai_key, query_text,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "results_clustered.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nСохранено: {filepath}")
    meta = output["metadata"]
    if meta.get("fallback"):
        print(f"  Режим fallback: {meta['fallback_reason']}")
        if output.get("fallback_result") and output["fallback_result"]:
            print(f"  Лучший документ: {output['fallback_result']['best_document']['id']}")
        else:
            print("  Выбросов, прошедших фильтр, не найдено.")
    else:
        print(f"  Кластеров сформировано: {meta['n_clusters_formed']}")
        print(f"  Выбросов прошло фильтр: {len(meta['outlier_ids_qualified'])}")
        for i in range(1, args.top_clusters + 1):
            key = f"result_cluster_{i}"
            if key in output:
                r = output[key]
                print(f"  {key}: combined_score={r['combined_score']}, "
                      f"источник лучшего: {r['best_document_source']}, "
                      f"документ: {r['best_document']['id']}")
    print("\nГотово.")


if __name__ == "__main__":
    main()