"""
search_v5.py — Поиск v5: multi-type union + кластерный cross-encoder.

Архитектура:
  1. Для каждого AS-IS шага берём все типы чанков (table, enriched, narrative)
  2. Каждый тип → KNN top-50 (без BM25)
  3. Union результатов → определяем top-3 секции L2 по частоте попадания
  4. Берём ВСЕ чанки из этих секций (из банка, не только найденные KNN)
  5. Cross-encoder ОТДЕЛЬНО по каждой секции (10-50 чанков, не 100+)
  6. Лучший кандидат из каждой секции → итоговый список

Использование:
  python scripts/search_v4.py --mode etalon
  python scripts/search_v4.py --mode full --output output/search_v4_results.json
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter


# === КОНФИГУРАЦИЯ ===

OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_BANK = "processscout_bank"

KNN_TOP_K = 50          # KNN top-K для каждого типа чанка
TOP_SECTIONS = 3        # Сколько секций L2 брать
RERANK_TOP_N = 3        # Лучших кандидатов из каждой секции

SEARCH_CHUNK_TYPES = ["step_table", "step_enriched", "step_narrative"]
REC_CHUNK_TYPES = ["rec_automation", "rec_process", "rec_consulting", "rec_general"]

ETALON = [
    ("BP-06.01.01", ["FB-4.7.1", "FB-4.16.1"],
     "Формирование годового номенклатурного плана"),
    ("BP-06.01.02", ["FB-4.7.3"],
     "Разработка сетевого графика выпуска"),
    ("BP-06.01.03", ["FB-4.3", "FB-4.7.2"],
     "Прогнозный план + MRP"),
    ("BP-06.01.04", ["FB-4.6.2"],
     "Ограничения мощности"),
    ("BP-06.01.05", ["FB-5.13.1"],
     "% выполнения плана"),
    ("BP-06.01.06", ["FB-4.10.1.2", "FB-4.6.8.4"],
     "Расчет загрузки РЦ"),
    ("BP-06.01.07", ["FB-4.10.1.1"],
     "Потребность в персонале"),
]


def get_section_l2(bank_id):
    """FB-4.7.1 → '4.7', FB-5.13.6.1 → '5.13', FB-4.6.8.4 → '4.6'"""
    clean = bank_id.replace("FB-", "")
    clean = re.sub(r'[а-яё]\)$', '', clean)
    parts = clean.split(".")
    return ".".join(parts[:2])


def create_opensearch_client():
    from opensearchpy import OpenSearch
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60
    )
    print(f"OpenSearch: {client.info()['version']['number']}")
    return client


def load_biencoder():
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3 (bi-encoder)...")
    return SentenceTransformer("BAAI/bge-m3")


def load_crossencoder():
    from sentence_transformers import CrossEncoder
    print("Загрузка bge-reranker-v2-m3 (cross-encoder)...")
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


def load_bank_by_section(bank_chunks):
    """
    Группирует все чанки банка по секции L2.
    Возвращает: {'4.7': [chunk1, chunk2, ...], '5.13': [...], ...}
    """
    by_section = defaultdict(list)
    for c in bank_chunks:
        cid = c.get("metadata", {}).get("chunk_id", c.get("chunk_id", ""))
        section = get_section_l2(cid)
        by_section[section].append({
            "chunk_id": cid,
            "chunk_text_plain": c.get("chunk_text_plain", ""),
            "chunk_text_context": c.get("chunk_text_context", ""),
        })
    return dict(by_section)


def knn_search(client, query_embedding, top_k=50):
    result = client.search(
        index=INDEX_BANK,
        body={
            "size": top_k,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}}
        }
    )
    return result["hits"]["hits"]


def union_and_find_sections(all_hits, top_n_sections=3):
    """
    1. Union всех KNN результатов
    2. Считаем частоту секций L2
    3. Возвращаем top-N секций и union-кандидатов
    """
    # Union с дедупликацией
    seen = {}
    for source_type, hits in all_hits:
        for hit in hits:
            bank_id = hit["_id"]
            score = hit["_score"]
            if bank_id not in seen or score > seen[bank_id]["best_score"]:
                seen[bank_id] = {
                    "bank_id": bank_id,
                    "best_score": score,
                    "best_source": source_type,
                    "found_by": set()
                }
            seen[bank_id]["found_by"].add(source_type)

    # Считаем секции L2 по частоте (сколько кандидатов из каждой секции)
    section_counter = Counter()
    # Взвешенный подсчёт: кандидат найденный несколькими типами весит больше
    for bank_id, info in seen.items():
        section = get_section_l2(bank_id)
        weight = len(info["found_by"])  # 1-3 в зависимости от числа типов
        section_counter[section] += weight

    # Top-N секций
    top_sections = [s for s, _ in section_counter.most_common(top_n_sections)]

    # Преобразуем found_by в list
    candidates = []
    for info in seen.values():
        info["found_by"] = sorted(info["found_by"])
        info["section"] = get_section_l2(info["bank_id"])
        candidates.append(info)

    return candidates, top_sections, dict(section_counter)


def rerank_within_section(crossencoder, query_text, section_chunks, top_n=3):
    """
    Cross-encoder reranking внутри одной секции.
    Принимает ВСЕ чанки секции (не только найденные KNN).
    """
    if not section_chunks:
        return []

    pairs = []
    for chunk in section_chunks:
        text = chunk.get("chunk_text_context", "") or chunk.get("chunk_text_plain", "")
        pairs.append((query_text, text))

    scores = crossencoder.predict(pairs)

    results = []
    for chunk, score in zip(section_chunks, scores):
        results.append({
            "bank_id": chunk["chunk_id"],
            "rerank_score": float(score),
            "chunk_text_plain": chunk.get("chunk_text_plain", "")[:150],
        })

    results.sort(key=lambda x: -x["rerank_score"])
    return results[:top_n]


def get_best_query_text(chunks_by_type):
    """Лучший текст для cross-encoder: enriched > narrative > table."""
    for ct in ["step_enriched", "step_narrative", "step_table"]:
        if ct in chunks_by_type:
            return chunks_by_type[ct].get("search_text", "")
    for ct in REC_CHUNK_TYPES:
        if ct in chunks_by_type:
            return chunks_by_type[ct].get("search_text", "")
    return ""


def search_one_pair(client, biencoder, crossencoder, chunks_by_type,
                    bank_by_section, top_k=50, top_sections=3, rerank_top_n=3):
    """
    Полный поиск для одного pair_id:
    1. KNN по каждому типу чанка
    2. Union → определяем top секции
    3. Берём ВСЕ чанки из этих секций
    4. Cross-encoder отдельно по каждой секции
    5. Лучшие из каждой секции → итог
    """
    # Шаг 1: KNN по каждому типу
    all_hits = []
    for chunk_type, chunk in chunks_by_type.items():
        query_text = chunk.get("search_text", "")
        if not query_text:
            continue
        embedding = biencoder.encode(query_text, normalize_embeddings=True).tolist()
        hits = knn_search(client, embedding, top_k)
        all_hits.append((chunk_type, hits))

    if not all_hits:
        return [], [], {}

    # Шаг 2: Union и определение секций
    candidates, top_secs, section_weights = union_and_find_sections(all_hits, top_sections)

    # Шаг 3-4: Cross-encoder по каждой секции
    query_for_rerank = get_best_query_text(chunks_by_type)
    reranked_by_section = {}

    for section in top_secs:
        section_chunks = bank_by_section.get(section, [])
        if not section_chunks:
            continue
        top_in_section = rerank_within_section(
            crossencoder, query_for_rerank, section_chunks, rerank_top_n
        )
        reranked_by_section[section] = top_in_section

    # Шаг 5: Собираем итоговый список (лучшие из каждой секции)
    final = []
    for section in top_secs:
        for item in reranked_by_section.get(section, []):
            item["section"] = section
            final.append(item)

    # Сортируем по rerank_score
    final.sort(key=lambda x: -x["rerank_score"])

    return candidates, final, {
        "top_sections": top_secs,
        "section_weights": section_weights,
        "union_size": len(candidates)
    }


def run_etalon_test(client, biencoder, crossencoder, asis_chunks, bank_by_section):
    """Тест на эталонных парах."""
    print("\n" + "=" * 80)
    print("ЭТАЛОННЫЙ ТЕСТ v4.1: union + кластерный cross-encoder")
    print("=" * 80)

    by_pair = defaultdict(dict)
    for c in asis_chunks:
        pid = c["metadata"].get("pair_id", "")
        ct = c["metadata"].get("chunk_type", "")
        by_pair[pid][ct] = c

    stats = {"top1": 0, "top3": 0, "top5": 0, "total": 0, "miss": 0}

    for pair_id, expected_banks, desc in ETALON:
        chunks_by_type = by_pair.get(pair_id, {})
        if not chunks_by_type:
            print(f"\n{pair_id}: НЕТ ЧАНКОВ")
            continue

        candidates, final, meta = search_one_pair(
            client, biencoder, crossencoder, chunks_by_type,
            bank_by_section, KNN_TOP_K, TOP_SECTIONS, RERANK_TOP_N
        )

        final_ids = [c["bank_id"] for c in final]

        print(f"\n{'=' * 80}")
        print(f"{pair_id}: {desc}")
        print(f"Эталон: {expected_banks}")
        print(f"Union: {meta['union_size']} кандидатов")
        print(f"Top секции: {meta['top_sections']} (веса: {meta['section_weights']})")

        # Проверяем: в каких секциях лежат эталоны?
        for exp in expected_banks:
            exp_section = get_section_l2(exp)
            in_top_sections = exp_section in meta['top_sections']
            print(f"  {exp} (секция {exp_section}, "
                  f"{'В top секциях' if in_top_sections else '!!! НЕ в top секциях'}):")

            stats["total"] += 1

            if exp in final_ids:
                pos = final_ids.index(exp) + 1
                score = final[pos - 1]["rerank_score"]
                if pos <= 1: stats["top1"] += 1
                if pos <= 3: stats["top3"] += 1
                if pos <= 5: stats["top5"] += 1
                print(f"    → rerank@{pos} (score={score:.3f})")
            else:
                stats["miss"] += 1
                # Проверяем: секция была в top, но пункт не лучший?
                if in_top_sections:
                    print(f"    → НЕ в top-{RERANK_TOP_N} своей секции")
                else:
                    print(f"    → Секция не попала в top-{TOP_SECTIONS}")

        # Top результаты
        print(f"  --- Итоговый top-{min(9, len(final))} ---")
        for i, c in enumerate(final[:9]):
            marker = " <<<" if c["bank_id"] in expected_banks else ""
            print(f"    {i+1}. [{c['section']}] {c['bank_id']} "
                  f"(rerank={c['rerank_score']:.3f}){marker}")

    # Итог
    t = stats['total']
    print(f"\n{'=' * 80}")
    print(f"ИТОГО (из {t} эталонных пар):")
    print(f"  top-1:  {stats['top1']}/{t} ({stats['top1']/t*100:.0f}%)")
    print(f"  top-3:  {stats['top3']}/{t} ({stats['top3']/t*100:.0f}%)")
    print(f"  top-5:  {stats['top5']}/{t} ({stats['top5']/t*100:.0f}%)")
    print(f"  MISS:   {stats['miss']}/{t}")
    print(f"{'=' * 80}")


def run_full_search(client, biencoder, crossencoder, asis_chunks, bank_by_section, output_path):
    """Полный прогон."""
    print("\nПолный поиск v4.1...")

    by_pair = defaultdict(dict)
    for c in asis_chunks:
        pid = c["metadata"].get("pair_id", "")
        ct = c["metadata"].get("chunk_type", "")
        priority = c["metadata"].get("mapping_priority", "low")
        if priority in ("high", "medium"):
            by_pair[pid][ct] = c

    search_pairs = {}
    for pid, chunks in by_pair.items():
        step_types = {k: v for k, v in chunks.items() if k in SEARCH_CHUNK_TYPES}
        rec_types = {k: v for k, v in chunks.items() if k in REC_CHUNK_TYPES}
        if step_types:
            search_pairs[pid] = step_types
        elif rec_types:
            search_pairs[pid] = rec_types

    print(f"  Pair_ids: {len(search_pairs)}")

    results = []
    total = len(search_pairs)

    for i, (pid, chunks_by_type) in enumerate(sorted(search_pairs.items())):
        candidates, final, meta = search_one_pair(
            client, biencoder, crossencoder, chunks_by_type,
            bank_by_section, KNN_TOP_K, TOP_SECTIONS, RERANK_TOP_N
        )

        bp_id = ""
        for c in chunks_by_type.values():
            bp_id = c["metadata"].get("bp_id", "")
            if bp_id:
                break

        for rank, cand in enumerate(final):
            results.append({
                "pair_id": pid,
                "bank_id": cand["bank_id"],
                "bp_id": bp_id,
                "section": cand["section"],
                "rerank_score": round(cand["rerank_score"], 4),
                "rerank_position": rank + 1,
                "top_sections": meta.get("top_sections", []),
            })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] обработано")

    output = {
        "metadata": {
            "version": "v4.1",
            "method": "multi-type union + cluster cross-encoder",
            "knn_top_k": KNN_TOP_K,
            "top_sections": TOP_SECTIONS,
            "rerank_top_n_per_section": RERANK_TOP_N,
            "total_pairs": len(search_pairs),
            "total_results": len(results),
        },
        "results": results
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nСохранено: {output_path}")
    print(f"  Пар: {len(search_pairs)}, результатов: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Поиск v4.1: union + кластерный cross-encoder")
    parser.add_argument("--mode", choices=["etalon", "full"], default="etalon")
    parser.add_argument("--asis", default="output/chunks_bp-06.json")
    parser.add_argument("--bank", default="output/chunks_bank.json")
    parser.add_argument("--output", default="output/search_v4_results.json")
    args = parser.parse_args()

    biencoder = load_biencoder()
    crossencoder = load_crossencoder()
    client = create_opensearch_client()

    with open(args.asis) as f:
        asis_chunks = json.load(f)
    with open(args.bank) as f:
        bank_chunks = json.load(f)

    print(f"AS-IS: {len(asis_chunks)}, банк: {len(bank_chunks)}")

    # Группируем банк по секциям L2
    bank_by_section = load_bank_by_section(bank_chunks)
    print(f"Секций банка: {len(bank_by_section)}")
    for s in sorted(bank_by_section.keys()):
        print(f"  {s}: {len(bank_by_section[s])} чанков")

    if args.mode == "etalon":
        run_etalon_test(client, biencoder, crossencoder, asis_chunks, bank_by_section)
    else:
        run_full_search(client, biencoder, crossencoder, asis_chunks, bank_by_section, args.output)


if __name__ == "__main__":
    main()