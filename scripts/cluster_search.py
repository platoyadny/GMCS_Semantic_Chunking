"""
cluster_search.py — Кластерный поиск по банку функциональности для всех ASIS-чанков.

Для каждого чанка из переданного JSON-файла выполняет полный пайплайн:
  1. Кодирование чанка через BGE-M3 (как в hybrid_search: поле search_text или chunk_text_plain)
  2. knn-поиск в OpenSearch INDEX_BANK по полученному вектору
  3. UMAP + HDBSCAN — кластеризация результатов поиска
  4. Жёсткий фильтр кластеров (топ-N% ИЛИ абсолютный топ-K)
  5. Мягкое ранжирование по комбинированному скору (centroid + top3_mean + density)
  6. Фильтр выбросов (строже кластерного)
  7. GPT-4o выбирает один лучший документ из объединённого пула (кластер + выбросы)
  8. Все результаты записываются в один общий выходной JSON

Использование:
  python cluster_search.py --asis path/to/asis_chunks.json

  # Полный набор параметров:
  python cluster_search.py \
    --asis path/to/asis_chunks.json \
    --min_cosine 0.30 \
    --top_clusters 1 \
    --umap_components 50 \
    --umap_neighbors 15 \
    --hdbscan_min_cluster_size 3 \
    --hdbscan_min_samples 3 \
    --cluster_percentile 20 \
    --outlier_percentile 10 \
    --absolute_top_k 5 \
    --density_threshold 35 \
    --output_dir output

Переменные окружения (.env):
  OPENAI_API_KEY — ключ для доступа к GPT-4o.
"""

import json
import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime


OPENSEARCH_URL      = "https://10.40.10.111:9200"
OPENSEARCH_USER     = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_BANK          = "processscout_bank"

TIE_THRESHOLD = 0.025

# Retry-параметры для запросов к OpenSearch
RETRY_MAX_ATTEMPTS = 5       # максимум попыток на один запрос
RETRY_BACKOFF_BASE = 2.0     # базовая задержка в секундах (экспоненциальный backoff)
RETRY_BACKOFF_MAX  = 30.0    # максимальная задержка между попытками (сек)


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------

def load_env() -> str:
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
# BGE-M3
# ---------------------------------------------------------------------------

def load_model():
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"  Модель загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def encode_chunk(model, chunk: dict) -> list[float]:
    """
    Кодирует чанк в вектор.
    Приоритет полей (как в hybrid_search pass_asis_to_bank):
      search_text → chunk_text_plain
    """
    text = chunk.get("search_text") or chunk.get("chunk_text_plain", "")
    if not text:
        return None
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.tolist()


# ---------------------------------------------------------------------------
# OpenSearch — одно соединение + retry
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
        max_retries=0,        # retry реализован вручную ниже
        retry_on_timeout=False,
    )
    info = client.info()
    print(f"  OpenSearch: {info['version']['number']}, кластер: {info['cluster_name']}")
    return client


def search_with_retry(client, index: str, body: dict) -> dict:
    """
    Выполняет запрос к OpenSearch с экспоненциальным backoff.
    При исчерпании попыток бросает исключение.
    """
    last_exc = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return client.search(index=index, body=body)
        except Exception as exc:
            last_exc = exc
            if attempt == RETRY_MAX_ATTEMPTS:
                break
            delay = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
            print(f"    [retry {attempt}/{RETRY_MAX_ATTEMPTS}] Ошибка запроса: {exc}. "
                  f"Повтор через {delay:.1f}с...")
            time.sleep(delay)
    raise RuntimeError(
        f"OpenSearch не ответил после {RETRY_MAX_ATTEMPTS} попыток. "
        f"Последняя ошибка: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Поиск в INDEX_BANK
# ---------------------------------------------------------------------------

def run_search(client, query_embedding: list[float], min_cosine: float) -> list[dict]:
    """
    knn-поиск по INDEX_BANK. Возвращает документы с cosine_score >= min_cosine,
    отсортированные по убыванию.
    """
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
    response = search_with_retry(client, INDEX_BANK, body)
    raw_hits = response["hits"]["hits"]
    total    = response["hits"]["total"]["value"]

    if total > len(raw_hits):
        print(f"    Предупреждение: найдено {total}, возвращено {len(raw_hits)} (лимит size=10000).")

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

def resolve_embeddings(hits: list[dict], query_embedding: list[float]) -> np.ndarray:
    """
    Приоритет: _source.embedding → используем как есть.
    Если отсутствует — используем query_embedding как fallback для одиночных документов,
    но в норме INDEX_BANK всегда хранит векторы в _source.
    """
    if hits and hits[0].get("_embedding") is not None:
        matrix = np.array([h["_embedding"] for h in hits], dtype=np.float32)
    else:
        # Fallback: представляем каждый документ query_embedding (крайний случай)
        print("    Предупреждение: embedding отсутствует в _source. "
              "Кластеризация будет некорректной.")
        matrix = np.tile(
            np.array(query_embedding, dtype=np.float32),
            (len(hits), 1)
        )
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

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)

    min_cluster_size = min(args.hdbscan_min_cluster_size, n)
    min_samples      = min(args.hdbscan_min_samples, min_cluster_size)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return clusterer.fit_predict(reduced)


# ---------------------------------------------------------------------------
# Пороговые утилиты
# ---------------------------------------------------------------------------

def compute_pool_thresholds(hits: list[dict], percentile: int, absolute_top_k: int):
    scores               = [h["cosine_score"] for h in hits]
    pct_threshold        = float(np.percentile(scores, 100 - percentile))
    k_idx                = min(absolute_top_k - 1, len(hits) - 1)
    abs_threshold        = hits[k_idx]["cosine_score"]
    return pct_threshold, abs_threshold


def passes_filter(score: float, pct_threshold: float, abs_threshold: float) -> bool:
    return score >= pct_threshold or score >= abs_threshold


# ---------------------------------------------------------------------------
# Статистика и ранжирование кластеров
# ---------------------------------------------------------------------------

def compute_cluster_stats(
    hits: list[dict], labels: np.ndarray, embeddings: np.ndarray,
    query_embedding: list[float], density_threshold: float,
) -> list[dict]:
    query_vec   = np.array(query_embedding, dtype=np.float32)
    cluster_ids = sorted(set(labels) - {-1})
    stats = []

    for cid in cluster_ids:
        indices = [i for i, l in enumerate(labels) if l == cid]
        vecs    = embeddings[indices]
        scores  = [hits[i]["cosine_score"] for i in indices]

        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        centroid_cosine = round(float(np.dot(query_vec, centroid)), 6)

        top3_mean = round(float(np.mean(sorted(scores, reverse=True)[:3])), 6)
        density   = round(sum(1 for s in scores if s >= density_threshold) / len(scores), 4)

        # Комбинированный скор: взвешенная сумма трёх сигналов
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
    stats: list[dict], hits: list[dict],
    pct_threshold: float, abs_threshold: float, top_n: int,
) -> list[dict]:
    """
    Жёсткий фильтр + мягкое ранжирование по combined_score.
    """
    passed = [
        s for s in stats
        if any(
            passes_filter(hits[i]["cosine_score"], pct_threshold, abs_threshold)
            for i in s["doc_indices"]
        )
    ]
    passed.sort(key=lambda s: s["combined_score"], reverse=True)
    return passed[:top_n]


# ---------------------------------------------------------------------------
# GPT-4o
# ---------------------------------------------------------------------------

def call_gpt(query_text: str, candidates: list[dict], openai_key: str) -> dict:
    from openai import OpenAI
    oa_client = OpenAI(api_key=openai_key)

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

    response = oa_client.chat.completions.create(
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
    return {k: v for k, v in hit.items() if k not in ("_embedding", "_is_outlier")}


# ---------------------------------------------------------------------------
# Полный пайплайн для одного ASIS-чанка
# ---------------------------------------------------------------------------

def process_chunk(chunk: dict, query_embedding: list[float], client, openai_key: str, args) -> dict:
    """
    Выполняет полный пайплайн для одного ASIS-чанка.
    Возвращает словарь результата, готовый к записи в выходной JSON.
    """
    chunk_id   = chunk.get("chunk_id", chunk.get("id", "unknown"))
    query_text = chunk.get("search_text") or chunk.get("chunk_text_plain", "")

    # ---- Поиск ----
    hits_raw = run_search(client, query_embedding, args.min_cosine)

    if not hits_raw:
        return {
            "chunk_id":   chunk_id,
            "chunk_meta": chunk.get("metadata", {}),
            "status":     "no_results",
            "reason":     f"Нет результатов с cosine >= {args.min_cosine}",
            "result":     None,
        }

    # ---- Динамический пре-фильтр: топ-pre_filter_percentile% по cosine_score ----
    # Заменяет статичный min_cosine как рабочий фильтр.
    # Из всех документов, вернувшихся из OpenSearch, оставляем только лучшие N%
    # — именно этот пул идёт на кластеризацию и все последующие шаги.
    raw_scores        = [h["cosine_score"] for h in hits_raw]
    pre_filter_thr    = float(np.percentile(raw_scores, 100 - args.pre_filter_percentile))
    hits              = [h for h in hits_raw if h["cosine_score"] >= pre_filter_thr]

    if not hits:
        return {
            "chunk_id":   chunk_id,
            "chunk_meta": chunk.get("metadata", {}),
            "status":     "no_results_after_prefilter",
            "reason":     (f"После динамического пре-фильтра (топ-{args.pre_filter_percentile}%, "
                           f"порог={pre_filter_thr:.4f}) не осталось документов."),
            "result":     None,
        }

    # ---- Эмбеддинги ----
    embeddings = resolve_embeddings(hits, query_embedding)

    # ---- Кластеризация ----
    if len(hits) >= 2:
        labels = run_clustering(embeddings, args)
    else:
        labels = np.array([-1] * len(hits))

    cluster_ids_formed = sorted(set(labels) - {-1})
    n_clusters         = len(cluster_ids_formed)

    # ---- Карта кластеров для metadata ----
    clusters_formed: dict[str, list[str]] = {}
    for rank, cid in enumerate(cluster_ids_formed, 1):
        clusters_formed[f"cluster_{rank}"] = [
            hits[i]["id"] for i, l in enumerate(labels) if l == cid
        ]
    outlier_ids_all = [hits[i]["id"] for i, l in enumerate(labels) if l == -1]

    # ---- Пороги ----
    cluster_pct_thr, cluster_abs_thr = compute_pool_thresholds(
        hits,
        percentile=args.cluster_percentile,    # топ-N% пула для жёсткого фильтра кластеров
        absolute_top_k=args.absolute_top_k,    # абсолютный топ-K как второй критерий
    )
    outlier_pct_thr, outlier_abs_thr = compute_pool_thresholds(
        hits,
        percentile=args.outlier_percentile,    # топ-N% пула для фильтра выбросов (строже)
        absolute_top_k=args.absolute_top_k,    # тот же абсолютный топ-K
    )

    # density_threshold: нижняя граница верхних density_threshold% пула.
    # Вычисляется динамически — не зависит от абсолютного уровня косинусов запроса.
    pool_scores       = [h["cosine_score"] for h in hits]
    density_threshold = float(np.percentile(pool_scores, 100 - args.density_threshold))

    # ---- Фильтр выбросов ----
    qualified_outliers = []
    for i, (hit, label) in enumerate(zip(hits, labels)):
        if label == -1 and passes_filter(hit["cosine_score"], outlier_pct_thr, outlier_abs_thr):
            doc = dict(hit)
            doc["_is_outlier"] = True
            qualified_outliers.append(doc)

    # ---- Общий metadata результата ----
    result_meta = {
        "query_text":                   query_text,
        "min_cosine":                   args.min_cosine,
        "total_hits_raw":               len(hits_raw),
        "pre_filter_percentile":        args.pre_filter_percentile,
        "pre_filter_threshold":         round(pre_filter_thr, 6),
        "total_hits_after_prefilter":   len(hits),
        "n_clusters_formed":            n_clusters,
        "clusters_formed":              clusters_formed,
        "outlier_ids_all":              outlier_ids_all,
        "outlier_ids_qualified":        [d["id"] for d in qualified_outliers],
        "cluster_percentile":           args.cluster_percentile,
        "outlier_percentile":           args.outlier_percentile,
        "absolute_top_k":               args.absolute_top_k,
        "density_threshold_percentile": args.density_threshold,
        "density_threshold_value":      round(density_threshold, 6),
    }

    # ================================================================
    # FALLBACK: нет кластеров
    # ================================================================
    if n_clusters == 0:
        result_meta["fallback"]        = True
        result_meta["fallback_reason"] = (
            "Кластеров не сформировалось — HDBSCAN не нашёл плотных групп."
        )

        if not qualified_outliers:
            return {
                "chunk_id":   chunk_id,
                "chunk_meta": chunk.get("metadata", {}),
                "status":     "fallback_no_outliers",
                "metadata":   result_meta,
                "result":     None,
            }

        gpt_result = call_gpt(query_text, qualified_outliers, openai_key)
        best_id    = gpt_result.get("best_id")
        best_doc   = next(
            (clean_doc(d) for d in qualified_outliers if d["id"] == best_id),
            clean_doc(qualified_outliers[0]),
        )
        return {
            "chunk_id":   chunk_id,
            "chunk_meta": chunk.get("metadata", {}),
            "status":     "fallback_outliers_only",
            "metadata":   result_meta,
            "result": {
                "best_document_source": "outlier",
                "best_document": {
                    **best_doc,
                    "gpt_explanation": gpt_result.get("explanation", ""),
                },
            },
        }

    # ================================================================
    # СТАНДАРТНЫЙ ПУТЬ: кластеры есть
    # ================================================================
    stats  = compute_cluster_stats(
        hits, labels, embeddings, query_embedding,
        density_threshold=density_threshold,  # вычисленный из перцентиля пула
    )
    ranked = filter_and_rank_clusters(
        stats, hits,
        pct_threshold=cluster_pct_thr,
        abs_threshold=cluster_abs_thr,
        top_n=args.top_clusters,                   # сколько лучших кластеров обрабатывать
    )



    if not ranked:
        result_meta["no_clusters_passed_filter"] = True
        return {
            "chunk_id":   chunk_id,
            "chunk_meta": chunk.get("metadata", {}),
            "status":     "no_clusters_passed_filter",
            "metadata":   result_meta,
            "result":     None,
        }

    # Берём первый (лучший) кластер
    best_cluster = ranked[0]
    indices      = best_cluster["doc_indices"]
    cluster_docs = sorted(
        [hits[i] for i in indices],
        key=lambda x: x["cosine_score"],
        reverse=True,
    )

    # Объединённый пул для GPT: все документы кластера + прошедшие выбросы
    candidates = [dict(d) for d in cluster_docs] + list(qualified_outliers)

    gpt_result = call_gpt(query_text, candidates, openai_key)
    best_id    = gpt_result.get("best_id")
    best_doc   = next(
        (clean_doc(d) for d in candidates if d["id"] == best_id),
        clean_doc(candidates[0]),
    )
    is_outlier_win = best_id in {d["id"] for d in qualified_outliers}

    # Обновляем metadata статистикой выбранного кластера
    result_meta.update({
        "selected_cluster_rank":          1,
        "selected_cluster_centroid_cosine": best_cluster["centroid_cosine"],
        "selected_cluster_top3_mean":     best_cluster["top3_mean"],
        "selected_cluster_density":       best_cluster["density"],
        "selected_cluster_combined_score": best_cluster["combined_score"],
        "selected_cluster_size":          best_cluster["size"],
        "n_qualified_outliers":           len(qualified_outliers),
        "all_clusters_scores": [
            {
                "cluster_id": s["cluster_id"],
                "combined_score": s["combined_score"],
                "centroid_cosine": s["centroid_cosine"],
                "top3_mean": s["top3_mean"],
                "density": s["density"],
                "size": s["size"],
            }
            for s in stats
        ],

    })

    return {
        "chunk_id":   chunk_id,
        "chunk_meta": chunk.get("metadata", {}),
        "status":     "ok",
        "metadata":   result_meta,
        "result": {
            "best_document_source": "outlier" if is_outlier_win else "cluster",
            "best_document": {
                **best_doc,
                "gpt_explanation": gpt_result.get("explanation", ""),
            },
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Кластерный поиск по банку функциональности для всех ASIS-чанков"
    )

    parser.add_argument(
        "--asis", required=True,
        help="Путь к JSON-файлу с ASIS-чанками (список объектов)"
    )

    # ----- Поиск -----
    parser.add_argument(
        "--min_cosine", type=float, default=0.30,
        help="Минимальный cosine_score для knn-запроса к OpenSearch. "
             "Определяет радиус поиска: ниже = больше результатов. (по умолч.: 0.30)"
    )

    parser.add_argument(
        "--pre_filter_percentile", type=int, default=35,
        help="Динамический пре-фильтр: из всех документов, вернувшихся из OpenSearch, "
             "оставляем только топ-N%% по cosine_score. Именно этот пул идёт на кластеризацию. "
             "Заменяет min_cosine как рабочий фильтр — min_cosine остаётся лишь грубым внешним радиусом. "
             "(по умолч.: 35)"
    )

    # ----- Кластеризация -----
    parser.add_argument(
        "--umap_components", type=int, default=50,
        help="Целевая размерность после UMAP. Меньше = быстрее, но теряется информация. "
             "Диапазон: 20–100. (по умолч.: 50)"
    )
    parser.add_argument(
        "--umap_neighbors", type=int, default=15,
        help="n_neighbors для UMAP. Меньше = локальная структура, больше = глобальная. "
             "Диапазон: 5–50. (по умолч.: 15)"
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size", type=int, default=3,
        help="Минимальный размер кластера HDBSCAN. Меньше = больше узких кластеров. "
             "(по умолч.: 3)"
    )
    parser.add_argument(
        "--hdbscan_min_samples", type=int, default=3,
        help="min_samples для HDBSCAN. Больше = консервативнее, меньше выбросов. "
             "(по умолч.: 3)"
    )

    # ----- Отбор кластеров -----
    parser.add_argument(
        "--top_clusters", type=int, default=1,
        help="Сколько лучших кластеров обрабатывать. При значении >1 GPT вызывается "
             "для каждого, но в результат записывается только первый (лучший). (по умолч.: 1)"
    )
    parser.add_argument(
        "--cluster_percentile", type=int, default=20,
        help="Жёсткий фильтр кластеров: кластер проходит, если хотя бы один его документ "
             "входит в топ-N%% пула. Служит для отсева нерелевантных кластеров. (по умолч.: 20)"
    )
    parser.add_argument(
        "--absolute_top_k", type=int, default=5,
        help="Второй критерий жёсткого фильтра (кластеры и выбросы): документ проходит, "
             "если входит в абсолютный топ-K пула. Страхует при малом числе результатов. "
             "(по умолч.: 5)"
    )
    parser.add_argument(
        "--density_threshold", type=int, default=35,
        help="Порог для подсчёта density кластера, задаётся как перцентиль пула результатов. "
             "Density = доля документов кластера с cosine >= cosine топ-N%% пула. "
             "Например, 35 означает: порог = нижняя граница верхних 35%% пула. "
             "Вес в combined_score: 0.2. (по умолч.: 35)"
    )

    # ----- Фильтр выбросов -----
    parser.add_argument(
        "--outlier_percentile", type=int, default=10,
        help="Фильтр выбросов: выброс передаётся GPT, если входит в топ-N%% пула. "
             "Строже чем cluster_percentile. (по умолч.: 10)"
    )

    # ----- Вывод -----
    parser.add_argument(
        "--output_dir", default="output",
        help="Директория для выходного файла. (по умолч.: output)"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Валидация
    if not (-1.0 <= args.min_cosine <= 1.0):
        print("Ошибка: --min_cosine должен быть в диапазоне [-1.0, 1.0]")
        sys.exit(1)

    openai_key = load_env()

    # Загрузка ASIS-чанков
    asis_path = Path(args.asis)
    if not asis_path.exists():
        print(f"Ошибка: файл не найден: {asis_path}")
        sys.exit(1)
    with open(asis_path, encoding="utf-8") as f:
        asis_chunks = json.load(f)
    if not isinstance(asis_chunks, list):
        print("Ошибка: JSON-файл должен содержать список (array) чанков.")
        sys.exit(1)
    print(f"\nЗагружено ASIS-чанков: {len(asis_chunks)}")

    # Загрузка BGE-M3
    model = load_model()

    # Единое подключение к OpenSearch
    print("\nПодключение к OpenSearch...")
    client = create_client()

    print(f"\nПараметры:")
    print(f"  min_cosine:                {args.min_cosine}")
    print(f"  pre_filter_percentile:     топ-{args.pre_filter_percentile}% пула перед кластеризацией")
    print(f"  top_clusters:              {args.top_clusters}")
    print(f"  UMAP:                      {args.umap_components}d, n_neighbors={args.umap_neighbors}")
    print(f"  HDBSCAN:                   min_cluster_size={args.hdbscan_min_cluster_size}, "
          f"min_samples={args.hdbscan_min_samples}")
    print(f"  Фильтр кластеров:          топ-{args.cluster_percentile}% ИЛИ топ-{args.absolute_top_k}")
    print(f"  Фильтр выбросов:           топ-{args.outlier_percentile}% ИЛИ топ-{args.absolute_top_k}")
    print(f"  density_threshold:         топ-{args.density_threshold}% пула (динамический)")

    # Итерация по чанкам
    results       = []
    total         = len(asis_chunks)
    ok_count      = 0
    skip_count    = 0
    error_count   = 0

    print(f"\nОбработка {total} чанков...\n")

    for i, chunk in enumerate(asis_chunks, 1):
        chunk_id = chunk.get("chunk_id", chunk.get("id", f"chunk_{i}"))
        print(f"[{i}/{total}] chunk_id={chunk_id}")

        # Кодирование запроса
        query_embedding = encode_chunk(model, chunk)
        if query_embedding is None:
            print(f"  Пропуск: нет текста для кодирования (search_text и chunk_text_plain пусты).")
            results.append({
                "chunk_id":   chunk_id,
                "chunk_meta": chunk.get("metadata", {}),
                "status":     "skipped_no_text",
                "result":     None,
            })
            skip_count += 1
            continue

        # Полный пайплайн
        try:
            entry = process_chunk(chunk, query_embedding, client, openai_key, args)
            results.append(entry)
            status = entry.get("status", "?")
            if status == "ok":
                best = entry["result"]["best_document"]
                print(f"  ✓ {status} | источник: {entry['result']['best_document_source']} "
                      f"| лучший: {best['id']} | cosine: {best['cosine_score']}")
                ok_count += 1
            else:
                print(f"  — {status}")
                skip_count += 1
        except Exception as exc:
            print(f"  ✗ Ошибка: {exc}")
            results.append({
                "chunk_id":   chunk_id,
                "chunk_meta": chunk.get("metadata", {}),
                "status":     "error",
                "error":      str(exc),
                "result":     None,
            })
            error_count += 1

    # Сохранение
    output = {
        "metadata": {
            "asis_file":                args.asis,
            "index_bank":               INDEX_BANK,
            "total_chunks":             total,
            "ok_count":                 ok_count,
            "skip_count":               skip_count,
            "error_count":              error_count,
            "min_cosine":               args.min_cosine,
            "pre_filter_percentile":    args.pre_filter_percentile,
            "top_clusters":             args.top_clusters,
            "umap_components":          args.umap_components,
            "umap_neighbors":           args.umap_neighbors,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples":      args.hdbscan_min_samples,
            "cluster_percentile":       args.cluster_percentile,
            "outlier_percentile":       args.outlier_percentile,
            "absolute_top_k":           args.absolute_top_k,
            "density_threshold_percentile": args.density_threshold,
            "timestamp":                datetime.now().isoformat(),
        },
        "results": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "results_clustered.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Готово. Сохранено: {filepath}")
    print(f"  Успешно:  {ok_count}")
    print(f"  Пропущено / без результата: {skip_count}")
    print(f"  Ошибки:   {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()