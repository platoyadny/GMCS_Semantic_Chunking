"""
cluster_search.py — Поиск по банку функциональности с кластеризацией и GPT-отбором.

Пайплайн:
  1. knn-поиск в OpenSearch с порогом min_cosine
  2. UMAP (понижение размерности) + HDBSCAN (кластеризация)
  3. Выбор лучшего кластера по центроиду (тай-брейкер по max cosine_score при разнице < 0.025)
  4. Топ-N документов из каждого выбранного кластера → GPT-4o выбирает лучший
  5. Сохранение результатов в results_clustered.json

Использование:
  python cluster_search.py --min_cosine 0.60

  # Полный набор параметров:
  python cluster_search.py \
    --min_cosine 0.60 \
    --top_clusters 2 \
    --top_docs 10 \
    --umap_components 50 \
    --umap_neighbors 15 \
    --hdbscan_min_cluster_size 5 \
    --hdbscan_min_samples 3 \
    --output_dir output

Аргументы:
  --min_cosine                Минимальный порог косинусного сходства (обязателен).
  --top_clusters              Сколько лучших кластеров обрабатывать через GPT (по умолч.: 1).
  --top_docs                  Сколько топ-документов из кластера передавать GPT (по умолч.: 10).
  --umap_components           Целевая размерность UMAP (по умолч.: 50).
  --umap_neighbors            n_neighbors для UMAP (по умолч.: 15).
  --hdbscan_min_cluster_size  Минимальный размер кластера (по умолч.: 5).
  --hdbscan_min_samples       min_samples для HDBSCAN (по умолч.: 3).
  --output_dir                Директория для выходных файлов (по умолч.: output).

Fallback (нет кластеров):
  Если HDBSCAN не сформировал ни одного кластера, возвращает топ-3 документа
  по cosine_score без вызова GPT. В metadata указывается причина.

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

TIE_THRESHOLD = 0.025   # разница центроидов, при которой включается тай-брейкер


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------

def load_env() -> str:
    """Загружает .env, возвращает OPENAI_API_KEY."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv не установлен — читаем из os.environ напрямую

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
    """knn-поиск, фильтрация по min_cosine, сортировка по убыванию score."""
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
    response  = client.search(index=INDEX_BANK, body=body)
    raw_hits  = response["hits"]["hits"]
    total     = response["hits"]["total"]["value"]

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
            "_embedding":         source.get("embedding", None),  # служебное, в JSON не пишем
        })

    results.sort(key=lambda x: x["cosine_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Эмбеддинги для кластеризации
# ---------------------------------------------------------------------------

def resolve_embeddings(hits: list[dict], model) -> np.ndarray:
    """
    Приоритет: _source.embedding → локальное перекодирование chunk_text_plain.
    """
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

    n = len(embeddings)
    n_components = min(args.umap_components, n - 2)
    n_neighbors  = min(args.umap_neighbors,  n - 1)

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
# Выбор лучших кластеров
# ---------------------------------------------------------------------------

def compute_cluster_stats(
    hits: list[dict],
    labels: np.ndarray,
    embeddings: np.ndarray,
    query_embedding: list[float],
) -> list[dict]:
    """
    Для каждого кластера вычисляет centroid_cosine и max_cosine_score.
    """
    query_vec   = np.array(query_embedding, dtype=np.float32)
    cluster_ids = sorted(set(labels) - {-1})

    stats = []
    for cid in cluster_ids:
        indices = [i for i, l in enumerate(labels) if l == cid]
        vecs    = embeddings[indices]

        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        stats.append({
            "cluster_id":       cid,
            "centroid_cosine":  round(float(np.dot(query_vec, centroid)), 6),
            "max_cosine_score": round(max(hits[i]["cosine_score"] for i in indices), 6),
            "doc_indices":      indices,
            "size":             len(indices),
        })

    return stats


def rank_clusters(stats: list[dict], top_n: int) -> list[dict]:
    """
    Ранжирует кластеры с тай-брейкером:
      - Основной критерий: centroid_cosine (убывание).
      - Если разница между лидером и любым другим кластером < TIE_THRESHOLD,
        из группы «в ничью» побеждает тот, у кого выше max_cosine_score.

    Выбор применяется жадно: определяем победителя среди оставшихся,
    удаляем его из пула, повторяем до top_n.
    """
    if not stats:
        return []

    remaining = sorted(stats, key=lambda s: s["centroid_cosine"], reverse=True)
    result    = []

    while remaining and len(result) < top_n:
        top_score  = remaining[0]["centroid_cosine"]
        tie_group  = [s for s in remaining if top_score - s["centroid_cosine"] < TIE_THRESHOLD]
        winner     = max(tie_group, key=lambda s: s["max_cosine_score"])
        result.append(winner)
        remaining.remove(winner)

    return result


# ---------------------------------------------------------------------------
# GPT-4o: выбор лучшего документа
# ---------------------------------------------------------------------------

def call_gpt(query_text: str, candidates: list[dict], openai_key: str) -> dict:
    """
    Передаёт запрос и топ-N кандидатов GPT-4o.
    Возвращает: {"best_id": "...", "explanation": "..."}.
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    candidates_text = ""
    for i, doc in enumerate(candidates, 1):
        candidates_text += (
            f"\n--- Документ {i} (ID: {doc['id']}, cosine_score: {doc['cosine_score']}) ---\n"
            f"{doc['chunk_text_plain']}\n"
        )

    prompt = f"""Ты — эксперт по анализу бизнес-процессов. Твоя задача — найти среди предложенных документов тот, который наиболее точно и полно соответствует данному запросу.

ЗАПРОС:
{query_text}

ДОКУМЕНТЫ-КАНДИДАТЫ:
{candidates_text}

Выбери один документ, который наилучшим образом отвечает на запрос. При выборе учитывай смысловое соответствие, а не только ключевые слова.

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
# Очистка документа (убираем служебное поле _embedding)
# ---------------------------------------------------------------------------

def clean_doc(hit: dict) -> dict:
    return {k: v for k, v in hit.items() if k != "_embedding"}


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

    # ---- Карта всех кластеров (только ID документов) ----
    cluster_ids_formed = sorted(set(labels) - {-1})
    clusters_formed: dict[str, list[str]] = {}
    for rank, cid in enumerate(cluster_ids_formed, 1):
        clusters_formed[f"cluster_{rank}"] = [
            hits[i]["id"] for i, l in enumerate(labels) if l == cid
        ]

    outlier_ids = [hits[i]["id"] for i, l in enumerate(labels) if l == -1]
    n_clusters  = len(cluster_ids_formed)

    metadata = {
        "query_text":               query_text,
        "index":                    INDEX_BANK,
        "min_cosine":               args.min_cosine,
        "total_hits":               len(hits),
        "top_clusters":             args.top_clusters,
        "top_docs":                 args.top_docs,
        "umap_components":          args.umap_components,
        "umap_neighbors":           args.umap_neighbors,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples":      args.hdbscan_min_samples,
        "n_clusters_formed":        n_clusters,
        "clusters_formed":          clusters_formed,
        "outlier_ids":              outlier_ids,
        "timestamp":                datetime.now().isoformat(),
    }

    # ---- Fallback: нет кластеров ----
    if n_clusters == 0:
        metadata["fallback"]        = True
        metadata["fallback_reason"] = (
            "Кластеров не сформировалось — HDBSCAN не нашёл плотных групп "
            "при заданных параметрах."
        )
        return {
            "metadata":        metadata,
            "fallback_results": [clean_doc(h) for h in hits[:3]],
        }

    # ---- Ранжирование кластеров ----
    stats  = compute_cluster_stats(hits, labels, embeddings, query_embedding)
    ranked = rank_clusters(stats, top_n=args.top_clusters)

    output: dict = {"metadata": metadata}

    # ---- GPT-обработка каждого выбранного кластера ----
    for result_rank, cluster_stat in enumerate(ranked, 1):
        indices = cluster_stat["doc_indices"]

        # Документы кластера, отсортированные по cosine_score убывающе
        cluster_docs_sorted = sorted(
            [hits[i] for i in indices],
            key=lambda x: x["cosine_score"],
            reverse=True,
        )
        top_candidates = [clean_doc(d) for d in cluster_docs_sorted[:args.top_docs]]

        print(f"\nGPT-4o: обработка result_cluster_{result_rank} "
              f"({len(top_candidates)} кандидатов)...")
        gpt_result = call_gpt(query_text, top_candidates, openai_key)
        print(f"  Выбранный документ: {gpt_result.get('best_id')}")

        # Полная запись выбранного документа
        best_id  = gpt_result.get("best_id")
        best_doc = next(
            (d for d in top_candidates if d["id"] == best_id),
            top_candidates[0],  # fallback: первый по cosine, если GPT вернул неизвестный id
        )

        output[f"result_cluster_{result_rank}"] = {
            "cluster_rank":          result_rank,
            "centroid_cosine":       cluster_stat["centroid_cosine"],
            "max_cosine_in_cluster": cluster_stat["max_cosine_score"],
            "cluster_size":          cluster_stat["size"],
            "top_candidates":        top_candidates,
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
    parser.add_argument(
        "--min_cosine", type=float, required=True,
        help="Минимальный порог косинусного сходства"
    )
    parser.add_argument(
        "--top_clusters", type=int, default=1,
        help="Сколько лучших кластеров обрабатывать через GPT (по умолч.: 1)"
    )
    parser.add_argument(
        "--top_docs", type=int, default=10,
        help="Сколько топ-документов из кластера передавать GPT (по умолч.: 10)"
    )
    parser.add_argument(
        "--umap_components", type=int, default=50,
        help="Целевая размерность UMAP (по умолч.: 50)"
    )
    parser.add_argument(
        "--umap_neighbors", type=int, default=15,
        help="n_neighbors для UMAP (по умолч.: 15)"
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size", type=int, default=5,
        help="Минимальный размер кластера HDBSCAN (по умолч.: 5)"
    )
    parser.add_argument(
        "--hdbscan_min_samples", type=int, default=3,
        help="min_samples для HDBSCAN (по умолч.: 3)"
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Директория для выходных файлов (по умолч.: output)"
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

    openai_key = load_env()

    print("\nВведите текст для поиска:")
    query_text = input("> ").strip()
    if not query_text:
        print("Ошибка: текст запроса не может быть пустым.")
        sys.exit(1)

    print(f"\nЗапрос:              «{query_text}»")
    print(f"Минимальный cosine:  {args.min_cosine}")
    print(f"Топ кластеров:       {args.top_clusters}")
    print(f"Топ документов/GPT:  {args.top_docs}")
    print(f"UMAP:                {args.umap_components}d, n_neighbors={args.umap_neighbors}")
    print(f"HDBSCAN:             min_cluster_size={args.hdbscan_min_cluster_size}, "
          f"min_samples={args.hdbscan_min_samples}")

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
        print(f"  Возвращено топ-3 документа без GPT.")
    else:
        print(f"  Кластеров сформировано: {meta['n_clusters_formed']}")
        for i in range(1, args.top_clusters + 1):
            key = f"result_cluster_{i}"
            if key in output:
                r = output[key]
                print(f"  {key}: centroid_cosine={r['centroid_cosine']}, "
                      f"лучший документ: {r['best_document']['id']}")
    print("\nГотово.")


if __name__ == "__main__":
    main()