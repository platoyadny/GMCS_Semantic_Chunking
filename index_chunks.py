"""
index_chunks.py — Индексация чанков в OpenSearch с эмбеддингами BGE-M3.

Загружает JSON-файлы чанков (AS-IS и банк), генерирует эмбеддинги через
sentence-transformers (BGE-M3, 1024 измерений), и отправляет в OpenSearch
через bulk API.

Два индекса:
  - processscout_asis  — чанки из документов обследования AS-IS
  - processscout_bank  — чанки из банка функциональности

Концепция bulk-индексации из search.py (AI Companion FileProcessing).
Подключение к OpenSearch на vm-osrag01 с vm-israg02.

Использование:
  # Индексация обоих источников
  python scripts/index_chunks.py \
    --asis output/chunks_bp-06.json \
    --bank output/chunks_bank.json

  # Только банк
  python scripts/index_chunks.py --bank output/chunks_bank.json

  # Только AS-IS
  python scripts/index_chunks.py --asis output/chunks_bp-06.json
"""

import json
import argparse
import sys
from pathlib import Path

# OpenSearch подключение (vm-osrag01)
OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"

# Индексы
INDEX_ASIS = "processscout_asis"
INDEX_BANK = "processscout_bank"

# Батчинг: сколько чанков обрабатывать за раз
# Эмбеддинги генерируются батчами (быстрее чем по одному),
# bulk-запрос в OpenSearch тоже идёт батчами
EMBED_BATCH_SIZE = 32
INDEX_BATCH_SIZE = 50


def load_embedding_model():
    """
    Загружает BGE-M3 через sentence-transformers.
    Модель уже скачана на диск (~2.27 GB), повторная загрузка из кэша.
    Возвращает объект модели для вызова model.encode().
    """
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"  Модель загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(model, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """
    Генерирует эмбеддинги для списка текстов батчами.
    BGE-M3 на CPU — ~10-20 текстов/сек, батчинг ускоряет за счёт параллелизма.
    Возвращает список векторов (каждый — list[float] длиной 1024).
    """
    import numpy as np

    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        # normalize_embeddings=True — для innerproduct = cosine similarity
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())

        done = min(i + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"  Эмбеддинги: {done}/{total}")

    return all_embeddings


def create_opensearch_client():
    """
    Создаёт клиент OpenSearch для подключения к vm-osrag01.
    SSL без проверки сертификата (self-signed в dev-режиме).
    Паттерн подключения из search.py (AI Companion).
    """
    from opensearchpy import OpenSearch

    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,       # self-signed сертификат в dev
        ssl_show_warn=False,
        timeout=60
    )

    # Проверяем подключение
    info = client.info()
    print(f"  OpenSearch: {info['version']['number']}, кластер: {info['cluster_name']}")
    return client


def bulk_index(client, index_name: str, documents: list[dict], batch_size: int = INDEX_BATCH_SIZE):
    """
    Bulk-индексация документов в OpenSearch.
    Паттерн из search.py: формируем bulk body (action + document),
    отправляем батчами.

    Каждый document должен содержать:
      - _id: уникальный идентификатор документа
      - остальные поля: тело документа для индексации
    """
    total = len(documents)
    indexed = 0
    errors = 0

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        bulk_body = []

        for doc in batch:
            doc_id = doc.pop("_id")
            bulk_body.append({"index": {"_index": index_name, "_id": doc_id}})
            bulk_body.append(doc)

        response = client.bulk(body=bulk_body, refresh=False)

        if response.get("errors"):
            for item in response["items"]:
                if "error" in item.get("index", {}):
                    errors += 1
                    err = item["index"]["error"]
                    print(f"    ОШИБКА: {err.get('reason', err)}")

        indexed += len(batch)
        if indexed % 100 == 0 or indexed == total:
            print(f"  Индексация {index_name}: {indexed}/{total}")

    # Принудительный refresh после всех батчей — данные становятся доступны для поиска
    client.indices.refresh(index=index_name)
    print(f"  Готово: {indexed - errors} успешно, {errors} ошибок")
    return indexed - errors


def prepare_asis_documents(chunks: list[dict], embeddings: list[list[float]]) -> list[dict]:
    """
    Подготавливает AS-IS чанки для индексации.
    Формат JSON из chunk_asis.py → формат документа OpenSearch.

    Поле для эмбеддинга: search_text (оптимизирован для поиска).
    full_context сохраняется как есть — для LLM на этапе маппинга.
    """
    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        meta = chunk.get("metadata", {})
        doc = {
            "_id": chunk["chunk_id"],
            "search_text": chunk.get("search_text", ""),
            "full_context": chunk.get("full_context", ""),
            "embedding": embedding,
            "metadata": meta
        }
        documents.append(doc)
    return documents


def prepare_bank_documents(chunks: list[dict], embeddings: list[list[float]]) -> list[dict]:
    """
    Подготавливает чанки банка для индексации.
    Формат JSON из chunk_function_bank.py → формат документа OpenSearch.

    Поле для эмбеддинга: chunk_text_context (LLM-переформулировка,
    оптимизирована для семантического поиска).
    chunk_text_plain сохраняется для BM25 (с иерархией и аббревиатурами).
    """
    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        meta = chunk.get("metadata", {})
        doc = {
            "_id": meta.get("chunk_id", chunk.get("chunk_id", "")),
            "chunk_text_plain": chunk.get("chunk_text_plain", ""),
            "chunk_text_context": chunk.get("chunk_text_context", ""),
            "embedding": embedding,
            "metadata": meta
        }
        documents.append(doc)
    return documents


def main():
    parser = argparse.ArgumentParser(description="Индексация чанков в OpenSearch с BGE-M3 эмбеддингами")
    parser.add_argument("--asis", default=None, help="JSON с AS-IS чанками")
    parser.add_argument("--bank", default=None, help="JSON с чанками банка")
    args = parser.parse_args()

    if not args.asis and not args.bank:
        print("Укажите хотя бы один файл: --asis и/или --bank")
        sys.exit(1)

    # Загружаем модель эмбеддингов (один раз для обоих индексов)
    model = load_embedding_model()

    # Подключаемся к OpenSearch
    print("\nПодключение к OpenSearch...")
    client = create_opensearch_client()

    # --- Индексация AS-IS ---
    if args.asis:
        print(f"\n{'='*60}")
        print(f"Индексация AS-IS: {args.asis}")
        print(f"{'='*60}")

        with open(args.asis, "r", encoding="utf-8") as f:
            asis_chunks = json.load(f)
        print(f"  Загружено чанков: {len(asis_chunks)}")

        # Эмбеддинги генерируются из search_text (не full_context) —
        # search_text оптимизирован для поиска (чистый, без метаданных)
        asis_texts = [c.get("search_text", "") for c in asis_chunks]
        print(f"\n  Генерация эмбеддингов ({len(asis_texts)} текстов)...")
        asis_embeddings = generate_embeddings(model, asis_texts)

        asis_docs = prepare_asis_documents(asis_chunks, asis_embeddings)
        print(f"\n  Отправка в {INDEX_ASIS}...")
        bulk_index(client, INDEX_ASIS, asis_docs)

    # --- Индексация банка ---
    if args.bank:
        print(f"\n{'='*60}")
        print(f"Индексация банка: {args.bank}")
        print(f"{'='*60}")

        with open(args.bank, "r", encoding="utf-8") as f:
            bank_chunks = json.load(f)
        print(f"  Загружено чанков: {len(bank_chunks)}")

        # Эмбеддинги из chunk_text_context (LLM-переформулировка) —
        # он богаче лексически и лучше для семантического поиска
        bank_texts = [c.get("chunk_text_context", "") for c in bank_chunks]
        print(f"\n  Генерация эмбеддингов ({len(bank_texts)} текстов)...")
        bank_embeddings = generate_embeddings(model, bank_texts)

        bank_docs = prepare_bank_documents(bank_chunks, bank_embeddings)
        print(f"\n  Отправка в {INDEX_BANK}...")
        bulk_index(client, INDEX_BANK, bank_docs)

    # --- Итоги ---
    print(f"\n{'='*60}")
    print("Проверка индексов:")
    for idx in [INDEX_ASIS, INDEX_BANK]:
        try:
            count = client.count(index=idx)["count"]
            print(f"  {idx}: {count} документов")
        except Exception:
            pass
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
