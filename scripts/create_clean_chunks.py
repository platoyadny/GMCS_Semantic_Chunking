"""
create_clean_chunks.py — Создаёт чистые чанки (step_clean) из metadata.operation.

1. Берёт operation из каждого step-чанка (чистая формулировка из таблицы БП)
2. Составные (2+ предложения) → разбивает на подчанки
3. Генерирует эмбеддинги BGE-M3
4. Индексирует в OpenSearch (добавляет к существующим, не удаляет)
5. Сохраняет JSON с новыми чанками

  python scripts/create_clean_chunks.py
  python scripts/create_clean_chunks.py --dry-run  # только показать, не индексировать
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_ASIS = "processscout_asis"

ASIS_FILE = "output/chunks_bp-06.json"
OUTPUT_FILE = "output/chunks_clean.json"


def split_operation(operation):
    """
    Разбивает составную операцию на подоперации.
    Разделители: '. ' (точка+пробел) в середине текста.
    Убирает финальную точку с запятой.
    """
    # Убираем финальный ; или .
    text = operation.strip().rstrip(";").rstrip(".").strip()

    # Разбиваем по '. ' — точка + пробел (не просто точка, чтобы не ломать сокращения)
    parts = re.split(r'\.\s+', text)

    # Убираем пустые
    parts = [p.strip() for p in parts if p.strip()]

    return parts


def create_clean_chunks(asis_chunks):
    """
    Создаёт step_clean чанки из metadata.operation.
    Для составных операций — несколько подчанков.
    """
    # Собираем уникальные pair_id с их operation
    seen_pairs = {}
    for c in asis_chunks:
        pid = c["metadata"].get("pair_id", "")
        ct = c["metadata"].get("chunk_type", "")
        op = c["metadata"].get("operation", "")

        # Берём operation из step_table (есть у всех шагов)
        if ct == "step_table" and pid and op and pid not in seen_pairs:
            seen_pairs[pid] = c

    clean_chunks = []
    stats = {"simple": 0, "compound": 0, "sub_chunks": 0}

    for pid, source_chunk in sorted(seen_pairs.items()):
        meta = source_chunk["metadata"]
        operation = meta.get("operation", "")

        parts = split_operation(operation)

        if len(parts) == 1:
            # Простая операция — один чистый чанк
            stats["simple"] += 1
            clean_chunks.append({
                "chunk_id": f"{pid}-clean",
                "search_text": parts[0],
                "full_context": parts[0],
                "metadata": {
                    "bp_id": meta.get("bp_id", ""),
                    "bp_name": meta.get("bp_name", ""),
                    "step_id": meta.get("step_id", ""),
                    "pair_id": pid,
                    "chunk_type": "step_clean",
                    "mapping_priority": "high",
                    "operation": parts[0],
                    "executor": meta.get("executor", ""),
                    "system": meta.get("system", ""),
                    "source_doc": meta.get("source_doc", ""),
                    "is_compound": False,
                    "sub_index": 0,
                    "total_subs": 1,
                }
            })
        else:
            # Составная операция — несколько подчанков
            stats["compound"] += 1
            stats["sub_chunks"] += len(parts)

            for i, part in enumerate(parts):
                clean_chunks.append({
                    "chunk_id": f"{pid}-clean-{i+1}",
                    "search_text": part,
                    "full_context": part,
                    "metadata": {
                        "bp_id": meta.get("bp_id", ""),
                        "bp_name": meta.get("bp_name", ""),
                        "step_id": meta.get("step_id", ""),
                        "pair_id": pid,
                        "chunk_type": "step_clean",
                        "mapping_priority": "high",
                        "operation": part,
                        "executor": meta.get("executor", ""),
                        "system": meta.get("system", ""),
                        "source_doc": meta.get("source_doc", ""),
                        "is_compound": True,
                        "sub_index": i,
                        "total_subs": len(parts),
                        "original_operation": operation,
                    }
                })

    return clean_chunks, stats


def index_chunks(clean_chunks, dry_run=False):
    """Генерирует эмбеддинги и индексирует в OpenSearch."""
    from sentence_transformers import SentenceTransformer
    from opensearchpy import OpenSearch

    print("Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")

    # Генерация эмбеддингов
    texts = [c["search_text"] for c in clean_chunks]
    print(f"Генерация эмбеддингов ({len(texts)} текстов)...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    if dry_run:
        print("\n--- DRY RUN: не индексируем ---")
        return

    # Индексация
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60
    )
    print(f"OpenSearch: {client.info()['version']['number']}")

    success = 0
    errors = 0

    for chunk, emb in zip(clean_chunks, embeddings):
        doc = {
            "search_text": chunk["search_text"],
            "full_context": chunk["full_context"],
            "embedding": emb.tolist(),
            "metadata": chunk["metadata"],
        }
        try:
            client.index(index=INDEX_ASIS, id=chunk["chunk_id"], body=doc)
            success += 1
        except Exception as e:
            print(f"  Ошибка {chunk['chunk_id']}: {e}")
            errors += 1

    print(f"\nИндексация: {success} успешно, {errors} ошибок")

    # Проверка
    count = client.count(index=INDEX_ASIS)["count"]
    print(f"Всего документов в {INDEX_ASIS}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать чанки, не индексировать")
    parser.add_argument("--asis", default=ASIS_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    # Загрузка
    with open(args.asis) as f:
        asis_chunks = json.load(f)
    print(f"Загружено AS-IS чанков: {len(asis_chunks)}")

    # Создание чистых чанков
    clean_chunks, stats = create_clean_chunks(asis_chunks)

    print(f"\nСоздано step_clean чанков: {len(clean_chunks)}")
    print(f"  Простых операций: {stats['simple']}")
    print(f"  Составных операций: {stats['compound']} → {stats['sub_chunks']} подчанков")

    # Показать все чистые чанки
    print(f"\n{'='*70}")
    for c in clean_chunks:
        compound = " [ПОДЧАНК {}/{}]".format(
            c["metadata"]["sub_index"] + 1, c["metadata"]["total_subs"]
        ) if c["metadata"]["is_compound"] else ""
        print(f"{c['chunk_id']}{compound}: {c['search_text']}")
    print(f"{'='*70}")

    # Сохраняем JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clean_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nСохранено: {args.output}")

    # Индексация
    index_chunks(clean_chunks, dry_run=args.dry_run)


if __name__ == "__main__":
    main()