"""
cluster_search.py — Поиск по банку функциональности с кластеризацией результатов.

Принимает текстовый запрос из терминала, кодирует его через BGE-M3,
выполняет knn-запрос к OpenSearch, затем кластеризует полученные векторы
через UMAP (понижение размерности) + HDBSCAN (кластеризация).

Использование:
  python cluster_search.py --min_cosine 0.60

  # С явными параметрами UMAP и HDBSCAN:
  python cluster_search.py \
    --min_cosine 0.60 \
    --umap_components 50 \
    --umap_neighbors 15 \
    --hdbscan_min_cluster_size 5 \
    --hdbscan_min_samples 3 \
    --output_dir output

Аргументы:
  --min_cosine                Минимальный порог косинусного сходства.
                              Определяет радиус поиска в OpenSearch. Обязателен.

  --umap_components           Целевая размерность после UMAP (по умолчанию: 50).
                              Если документов мало, автоматически снижается.

  --umap_neighbors            n_neighbors для UMAP (по умолчанию: 15).
                              Меньше значение — локальная структура, больше — глобальная.

  --hdbscan_min_cluster_size  Минимальный размер кластера (по умолчанию: 5).
                              Главный гиперпараметр: меньше значение — больше мелких кластеров.

  --hdbscan_min_samples       min_samples для HDBSCAN (по умолчанию: 3).
                              Влияет на устойчивость к шуму: больше значение — консервативнее.

  --output_dir                Директория для выходных файлов (по умолчанию: output).

Выходной файл:
  results_clustered.json — JSON с разделением по кластерам:
    {
      "metadata": { ... параметры запроса и статистика ... },
      "cluster_1": [ { id, cosine_score, chunk_text_plain, chunk_text_context, metadata }, ... ],
      "cluster_2": [ ... ],
      ...
      "outliers":  [ ... ]   <- документы, не вошедшие ни в один кластер
    }

Примечание по векторам:
  Скрипт сначала проверяет, хранятся ли векторы в _source документа OpenSearch
  (поле "embedding"). Если нет — перекодирует тексты через BGE-M3 локально.
  Перекодирование медленнее, но не требует изменений в индексе.
"""

import json
import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime


OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_BANK = "processscout_bank"


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

def load_model():
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"  Модель загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def encode_query(model, text: str) -> list[float]:
    """Кодирует текст запроса. normalize_embeddings=True — косинус через inner product."""
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.tolist()


def encode_texts(model, texts: list[str]) -> np.ndarray:
    """Перекодирует список текстов (fallback, если векторов нет в _source)."""
    print(f"  Кодирование {len(texts)} текстов батчами...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
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
        timeout=60
    )
    info = client.info()
    print(f"  OpenSearch: {info['version']['number']}, кластер: {info['cluster_name']}")
    return client


def run_search(client, query_embedding: list[float], min_cosine: float, index_name: str = INDEX_BANK) -> list[dict]:
    """
    Выполняет knn-поиск и возвращает документы с cosine_score >= min_cosine.
    Пытается получить поле "embedding" из _source — оно нужно для кластеризации.
    Если его там нет, вернёт None в _embedding и скрипт перекодирует тексты.
    """
    body = {
        "size": 10000,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": 10000
                }
            }
        }
    }

    response = client.search(index=index_name, body=body)
    raw_hits = response["hits"]["hits"]
    total_available = response["hits"]["total"]["value"]

    if total_available > len(raw_hits):
        print(
            f"  Предупреждение: найдено {total_available} документов, "
            f"но возвращено только {len(raw_hits)} (лимит size=10000).\n"
            f"  Для полного результата увеличьте index.max_result_window в OpenSearch."
        )

    results = []
    for h in raw_hits:
        # inner product на нормализованных векторах = cosine similarity, сдвинутый в [0,2]
        cosine = round(h["_score"] - 1, 6)
        if cosine < min_cosine:
            continue
        source = h["_source"]
        results.append({
            "id":               h["_id"],
            "cosine_score":     cosine,
            "chunk_text_plain": source.get("chunk_text_plain", ""),
            "chunk_text_context": source.get("chunk_text_context", ""),
            "metadata":         source.get("metadata", {}),
            "_embedding":       source.get("embedding", None),  # может отсутствовать
        })

    results.sort(key=lambda x: x["cosine_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Получение матрицы эмбеддингов
# ---------------------------------------------------------------------------

def resolve_embeddings(hits: list[dict], model) -> np.ndarray:
    """
    Возвращает матрицу (n_docs, dim) float32.
    Приоритет: _source.embedding → перекодирование chunk_text_plain.
    """
    has_stored = hits[0].get("_embedding") is not None

    if has_stored:
        print("  Векторы найдены в _source, используем их напрямую.")
        embeddings = np.array([h["_embedding"] for h in hits], dtype=np.float32)
    else:
        print("  Поле 'embedding' отсутствует в _source — перекодируем тексты локально.")
        texts = [h["chunk_text_plain"] for h in hits]
        embeddings = encode_texts(model, texts)

    print(f"  Матрица эмбеддингов: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Кластеризация
# ---------------------------------------------------------------------------

def run_clustering(embeddings: np.ndarray, args) -> np.ndarray:
    """
    1. UMAP — понижение размерности (косинусная метрика).
    2. HDBSCAN — кластеризация (евклидова метрика на сниженном пространстве).

    Возвращает массив меток: -1 = outlier, 0..K-1 = кластеры.
    """
    import umap
    import hdbscan as hdbscan_lib

    n_samples = len(embeddings)

    # --- UMAP ---
    n_components = min(args.umap_components, n_samples - 2)
    n_neighbors  = min(args.umap_neighbors, n_samples - 1)

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

    # --- HDBSCAN ---
    min_cluster_size = min(args.hdbscan_min_cluster_size, n_samples)
    min_samples      = min(args.hdbscan_min_samples, min_cluster_size)

    print(f"\nHDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",  # excess of mass — стандартный метод
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))
    print(f"  Найдено кластеров: {n_clusters}  |  выбросов: {n_outliers}")

    return labels


# ---------------------------------------------------------------------------
# Сборка и сохранение результата
# ---------------------------------------------------------------------------

def build_output(hits: list[dict], labels: np.ndarray, shared_meta: dict) -> dict:
    """
    Собирает итоговый словарь:
      metadata, cluster_1, cluster_2, ..., outliers

    Служебное поле _embedding убирается из каждого документа.
    Кластеры нумеруются с 1, сортируются по убыванию размера.
    """
    buckets: dict[int, list] = {}
    outlier_list = []

    for hit, label in zip(hits, labels):
        doc = {k: v for k, v in hit.items() if k != "_embedding"}
        if label == -1:
            outlier_list.append(doc)
        else:
            buckets.setdefault(int(label), []).append(doc)

    # Нумеруем кластеры по убыванию размера (самый большой = cluster_1)
    sorted_buckets = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    named_clusters = {
        f"cluster_{i + 1}": docs
        for i, (_, docs) in enumerate(sorted_buckets)
    }

    return {
        "metadata": {
            **shared_meta,
            "n_clusters": len(named_clusters),
            "n_outliers": len(outlier_list),
            "cluster_sizes": {k: len(v) for k, v in named_clusters.items()},
            "timestamp": datetime.now().isoformat(),
        },
        **named_clusters,
        "outliers": outlier_list,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Поиск по банку функциональности с кластеризацией результатов"
    )

    parser.add_argument(
        "--min_cosine", type=float, required=True,
        help="Минимальный порог косинусного сходства — радиус поиска в OpenSearch"
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Директория для выходных файлов (по умолчанию: output)"
    )

    # UMAP
    parser.add_argument(
        "--umap_components", type=int, default=50,
        help="Целевая размерность UMAP (по умолчанию: 50)"
    )
    parser.add_argument(
        "--umap_neighbors", type=int, default=15,
        help="n_neighbors для UMAP (по умолчанию: 15)"
    )

    # HDBSCAN
    parser.add_argument(
        "--hdbscan_min_cluster_size", type=int, default=5,
        help="Минимальный размер кластера в HDBSCAN (по умолчанию: 5)"
    )
    parser.add_argument(
        "--hdbscan_min_samples", type=int, default=3,
        help="min_samples для HDBSCAN (по умолчанию: 3)"
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

    print("\nВведите текст для поиска:")
    query_text = input("> ").strip()
    if not query_text:
        print("Ошибка: текст запроса не может быть пустым.")
        sys.exit(1)

    print(f"\nЗапрос:              «{query_text}»")
    print(f"Минимальный cosine:  {args.min_cosine}")
    print(f"UMAP:                {args.umap_components}d, n_neighbors={args.umap_neighbors}")
    print(f"HDBSCAN:             min_cluster_size={args.hdbscan_min_cluster_size}, "
          f"min_samples={args.hdbscan_min_samples}")

    # Загрузка модели и подключение
    model = load_model()
    print("\nПодключение к OpenSearch...")
    client = create_client()

    # Поиск
    print("\nКодирование запроса...")
    query_embedding = encode_query(model, query_text)

    print(f"\nПоиск в {INDEX_BANK} (min_cosine={args.min_cosine})...")
    hits = run_search(client, query_embedding, args.min_cosine)
    print(f"  Документов после фильтрации по порогу: {len(hits)}")

    if len(hits) < 2:
        print("\nНедостаточно результатов для кластеризации (нужно минимум 2). "
              "Попробуйте снизить --min_cosine.")
        sys.exit(0)

    # Получение матрицы векторов
    print("\nПодготовка эмбеддингов...")
    embeddings = resolve_embeddings(hits, model)

    # Кластеризация
    labels = run_clustering(embeddings, args)

    # Сборка и сохранение
    shared_meta = {
        "query_text":               query_text,
        "index":                    INDEX_BANK,
        "min_cosine":               args.min_cosine,
        "total_hits":               len(hits),
        "umap_components":          args.umap_components,
        "umap_neighbors":           args.umap_neighbors,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples":      args.hdbscan_min_samples,
    }

    output = build_output(hits, labels, shared_meta)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "results_clustered.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nСохранено: {filepath}")
    print(f"  Кластеров: {output['metadata']['n_clusters']}")
    for cluster_name, size in output['metadata']['cluster_sizes'].items():
        print(f"    {cluster_name}: {size} документов")
    print(f"  Выбросов:  {output['metadata']['n_outliers']}")
    print("\nГотово.")


if __name__ == "__main__":
    main()