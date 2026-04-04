"""
assemble_chunks.py — Фаза 0: сборка chunks_3types.json из docx → чанки → clean → merge.

Полный пайплайн чанкинга для одного документа AS-IS:
  1. chunk_asis.py  → narr + enriched + table + rec (все типы)
  2. create_clean_chunks.py → step_clean (из metadata.operation, разбивка составных)
  3. Merge: убрать step_table, оставить step_narrative + step_enriched + step_clean + rec_*
  4. Сохранить chunks_3types.json

ИСТОРИЯ:
  Изначально было 3 типа: step_table, step_narrative, step_enriched.
  step_table показал 14% exact match (мусор: исполнители, системы, входы/выходы).
  Создали step_clean — чистая формулировка операции, 43% exact match.
  Заменили step_table на step_clean → итоговый набор: 3 типа + рекомендации.

ИСПОЛЬЗОВАНИЕ:
  # Один документ:
  python scripts/assemble_chunks.py --input bp-06.docx --output_dir output/bp_06_01/chunks

  # Все документы в папке:
  python scripts/assemble_chunks.py --input_dir docs/ --output_dir output

  # Без индексации в OpenSearch (только JSON):
  python scripts/assemble_chunks.py --input bp-06.docx --output_dir output/bp_06_01/chunks --no-index

  # С LLM-переформулировкой step_enriched:
  python scripts/assemble_chunks.py --input bp-06.docx --output_dir output/bp_06_01/chunks --use-llm
"""

import json
import argparse
import sys
from pathlib import Path


# Типы чанков, которые попадают в итоговый файл
# step_table ИСКЛЮЧЁН — заменён на step_clean
KEEP_TYPES = {
    "step_narrative",
    "step_enriched",
    "step_clean",
    "rec_automation",
    "rec_process",
    "rec_consulting",
}

# Типы, которые исключаем явно (для отчётности)
DROP_TYPES = {
    "step_table",
    "card",
    "asis_notes",
    "intro",
    "narrative",
}


def run_chunk_asis(docx_path: str, output_json: str, use_llm: bool = False) -> list[dict]:
    """
    Шаг 1: запуск chunk_asis.py → все типы чанков.
    Импортируем chunk_document напрямую, без subprocess.
    """
    # Добавляем asis_chunker в path
    asis_dir = str(Path(__file__).resolve().parent.parent / "asis_chunker")
    if asis_dir not in sys.path:
        sys.path.insert(0, asis_dir)

    from chunk_asis import chunk_document, enhance_enriched_with_llm

    print(f"\n{'='*60}")
    print(f"ШАГ 1: chunk_asis.py → {docx_path}")
    print(f"{'='*60}")

    chunks = chunk_document(docx_path)
    print(f"  Создано чанков: {len(chunks)}")

    # LLM-переформулировка step_enriched (опционально)
    if use_llm:
        print("  LLM-переформулировка step_enriched...")
        try:
            from openai import OpenAI
            client = OpenAI()
            chunks = enhance_enriched_with_llm(chunks, client)
        except Exception as e:
            print(f"  Ошибка LLM: {e}, используем программные формулировки")

    # Сохраняем промежуточный результат
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Сохранено: {output_json}")

    return chunks


def run_create_clean(asis_chunks: list[dict]) -> list[dict]:
    """
    Шаг 2: создание step_clean из metadata.operation.
    Импортируем create_clean_chunks напрямую.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from create_clean_chunks import create_clean_chunks

    print(f"\n{'='*60}")
    print(f"ШАГ 2: create_clean_chunks → step_clean")
    print(f"{'='*60}")

    clean_chunks, stats = create_clean_chunks(asis_chunks)

    print(f"  Создано step_clean: {len(clean_chunks)}")
    print(f"    Простых: {stats['simple']}")
    print(f"    Составных: {stats['compound']} → {stats['sub_chunks']} подчанков")

    return clean_chunks


def merge_chunks(asis_chunks: list[dict], clean_chunks: list[dict]) -> list[dict]:
    """
    Шаг 3: merge — убрать step_table, оставить 3 типа + рекомендации.
    """
    print(f"\n{'='*60}")
    print(f"ШАГ 3: merge → chunks_3types")
    print(f"{'='*60}")

    # Считаем что имеем
    type_counts_before = {}
    for c in asis_chunks:
        ct = c.get("metadata", {}).get("chunk_type", "unknown")
        type_counts_before[ct] = type_counts_before.get(ct, 0) + 1

    print(f"  До merge (chunk_asis): {len(asis_chunks)} чанков")
    for ct, count in sorted(type_counts_before.items()):
        action = "✓ KEEP" if ct in KEEP_TYPES else "✗ DROP"
        print(f"    {action} {ct}: {count}")

    # Фильтруем: оставляем только KEEP_TYPES из asis
    filtered = [
        c for c in asis_chunks
        if c.get("metadata", {}).get("chunk_type", "") in KEEP_TYPES
    ]

    # Добавляем step_clean
    merged = filtered + clean_chunks

    # Итоговая статистика
    type_counts_after = {}
    for c in merged:
        ct = c.get("metadata", {}).get("chunk_type", "unknown")
        type_counts_after[ct] = type_counts_after.get(ct, 0) + 1

    print(f"\n  После merge: {len(merged)} чанков")
    for ct, count in sorted(type_counts_after.items()):
        print(f"    {ct}: {count}")

    # Проверка pair_id
    pair_ids = set()
    for c in merged:
        pid = c.get("metadata", {}).get("pair_id", "")
        if pid:
            pair_ids.add(pid)
    step_pairs = {p for p in pair_ids if not "-rec-" in p}
    rec_pairs = {p for p in pair_ids if "-rec-" in p}
    print(f"\n  Уникальных pair_id шагов: {len(step_pairs)}")
    print(f"  Уникальных pair_id рекомендаций: {len(rec_pairs)}")

    return merged


def index_to_opensearch(chunks: list[dict]):
    """Шаг 4 (опционально): индексация в OpenSearch."""
    from sentence_transformers import SentenceTransformer
    from opensearchpy import OpenSearch

    OPENSEARCH_URL = "https://10.40.10.111:9200"
    OPENSEARCH_USER = "admin"
    OPENSEARCH_PASSWORD = "ProcessScout_2026!"
    INDEX_ASIS = "processscout_asis"

    print(f"\n{'='*60}")
    print(f"ШАГ 4: индексация в OpenSearch ({INDEX_ASIS})")
    print(f"{'='*60}")

    print("  Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")

    texts = [c.get("search_text", "") for c in chunks]
    print(f"  Генерация эмбеддингов ({len(texts)} текстов)...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60
    )
    print(f"  OpenSearch: {client.info()['version']['number']}")

    success = 0
    errors = 0
    for chunk, emb in zip(chunks, embeddings):
        doc = {
            "search_text": chunk.get("search_text", ""),
            "full_context": chunk.get("full_context", ""),
            "embedding": emb.tolist(),
            "metadata": chunk.get("metadata", {}),
        }
        try:
            client.index(index=INDEX_ASIS, id=chunk["chunk_id"], body=doc)
            success += 1
        except Exception as e:
            print(f"    Ошибка {chunk['chunk_id']}: {e}")
            errors += 1

    print(f"  Индексация: {success} успешно, {errors} ошибок")
    count = client.count(index=INDEX_ASIS)["count"]
    print(f"  Всего в {INDEX_ASIS}: {count}")


def assemble(docx_path: str, output_dir: str, use_llm: bool = False,
             no_index: bool = False) -> str:
    """
    Полный пайплайн: docx → chunk_asis → clean → merge → chunks_3types.json.
    Возвращает путь к итоговому файлу.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Промежуточные файлы
    raw_json = str(output_dir / "chunks_raw.json")
    final_json = str(output_dir / "chunks_3types.json")

    # Шаг 1: chunk_asis
    asis_chunks = run_chunk_asis(docx_path, raw_json, use_llm=use_llm)

    # Шаг 2: create clean
    clean_chunks = run_create_clean(asis_chunks)

    # Шаг 3: merge
    merged = merge_chunks(asis_chunks, clean_chunks)

    # Сохраняем итог
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"ИТОГО: {final_json}")
    print(f"  {len(merged)} чанков готовы для маппинга")
    print(f"{'='*60}")

    # Шаг 4: индексация (опционально)
    if not no_index:
        index_to_opensearch(merged)

    return final_json


def main():
    parser = argparse.ArgumentParser(
        description="Фаза 0: docx → chunk_asis → clean → merge → chunks_3types.json"
    )
    parser.add_argument("--input", default=None,
                        help="Путь к .docx файлу (один документ)")
    parser.add_argument("--input_dir", default=None,
                        help="Путь к папке с .docx файлами (все документы)")
    parser.add_argument("--output_dir", default="output",
                        help="Базовая директория для результатов")
    parser.add_argument("--use-llm", action="store_true",
                        help="LLM-переформулировка step_enriched")
    parser.add_argument("--no-index", action="store_true",
                        help="Не индексировать в OpenSearch (только JSON)")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Укажите --input (один файл) или --input_dir (папка)")

    if args.input:
        # Один документ
        docx_path = Path(args.input)
        if not docx_path.exists():
            print(f"Ошибка: файл не найден: {docx_path}")
            sys.exit(1)

        # Определяем output_dir: output/bp_06_01/chunks/
        # Из имени файла bp-06.docx → bp_06
        stem = docx_path.stem.replace("-", "_")
        out_dir = Path(args.output_dir) / stem / "chunks"

        assemble(str(docx_path), str(out_dir),
                 use_llm=args.use_llm, no_index=args.no_index)

    elif args.input_dir:
        # Все .docx в папке
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Ошибка: папка не найдена: {input_dir}")
            sys.exit(1)

        docx_files = sorted(input_dir.glob("*.docx"))
        if not docx_files:
            print(f"Ошибка: нет .docx файлов в {input_dir}")
            sys.exit(1)

        print(f"Найдено документов: {len(docx_files)}")
        for docx_path in docx_files:
            stem = docx_path.stem.replace("-", "_")
            out_dir = Path(args.output_dir) / stem / "chunks"

            try:
                assemble(str(docx_path), str(out_dir),
                         use_llm=args.use_llm, no_index=args.no_index)
            except Exception as e:
                print(f"\n✗ Ошибка обработки {docx_path}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Все документы обработаны")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
