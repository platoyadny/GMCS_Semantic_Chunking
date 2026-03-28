"""
search_v4.py — Поиск v4: multi-type union + cross-encoder reranking.

Архитектура:
  1. Для каждого AS-IS шага (pair_id) берём все типы чанков (table, enriched, narrative)
  2. Каждый тип → KNN top-50 (без BM25)
  3. Union результатов, дедупликация по bank_id
  4. Cross-encoder (bge-reranker-v2-m3) переранжирует все ~60 кандидатов
  5. Top-N → готово к LLM-judge

Использование:
  # Эталонный тест (7 пар из БП 06.01)
  python scripts/search_v4.py --mode etalon

  # Полный прогон
  python scripts/search_v4.py --mode full \
    --asis output/chunks_bp-06.json \
    --bank output/chunks_bank.json \
    --output output/search_v4_results.json
"""

import json
import re
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict


# === КОНФИГУРАЦИЯ ===

OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_BANK = "processscout_bank"

# KNN top-K для каждого типа чанка
KNN_TOP_K = 50

# Сколько лучших кандидатов оставить после cross-encoder
RERANK_TOP_N = 10

# Типы чанков для поиска (в порядке приоритета)
SEARCH_CHUNK_TYPES = ["step_table", "step_enriched", "step_narrative"]

# Типы чанков рекомендаций (один чанк = один поиск)
REC_CHUNK_TYPES = ["rec_automation", "rec_process", "rec_consulting", "rec_general"]

# Эталонные пары из маппинга БП 06.01
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


def create_opensearch_client():
    """Подключение к OpenSearch."""
    from opensearchpy import OpenSearch
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60
    )
    version = client.info()['version']['number']
    print(f"OpenSearch: {version}")
    return client


def load_biencoder():
    """Загрузка BGE-M3 bi-encoder для генерации эмбеддингов запросов."""
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3 (bi-encoder)...")
    return SentenceTransformer("BAAI/bge-m3")


def load_crossencoder():
    """Загрузка cross-encoder для reranking."""
    from sentence_transformers import CrossEncoder
    print("Загрузка bge-reranker-v2-m3 (cross-encoder)...")
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


def knn_search(client, query_embedding, top_k=50):
    """Чистый KNN поиск без BM25."""
    result = client.search(
        index=INDEX_BANK,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            }
        }
    )
    return result["hits"]["hits"]


def union_candidates(all_hits):
    """
    Объединяет результаты нескольких KNN-поисков.
    Дедупликация по bank_id, сохраняет лучший скор и источник.
    """
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
                    "source": hit["_source"],
                    "found_by": set()
                }
            seen[bank_id]["found_by"].add(source_type)

    # Конвертируем set в list для JSON-сериализации
    candidates = list(seen.values())
    for c in candidates:
        c["found_by"] = sorted(c["found_by"])
        c["num_sources"] = len(c["found_by"])

    # Сортируем по количеству источников (desc), потом по лучшему скору (desc)
    candidates.sort(key=lambda x: (-x["num_sources"], -x["best_score"]))
    return candidates


def rerank_candidates(crossencoder, query_text, candidates, top_n=10):
    """
    Cross-encoder reranking.
    Для каждого кандидата формирует пару (запрос, текст банка) и получает скор.
    """
    if not candidates:
        return []

    # Формируем пары для cross-encoder
    pairs = []
    for cand in candidates:
        bank_text = cand["source"].get("chunk_text_context", "")
        if not bank_text:
            bank_text = cand["source"].get("chunk_text_plain", "")
        pairs.append((query_text, bank_text))

    # Получаем скоры
    scores = crossencoder.predict(pairs)

    # Добавляем скоры к кандидатам
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    # Сортируем по cross-encoder скору
    candidates.sort(key=lambda x: -x["rerank_score"])

    return candidates[:top_n]


def get_best_query_text(chunks_by_type):
    """
    Выбирает лучший текст запроса для cross-encoder reranking.
    Приоритет: enriched (самый полный контекст).
    """
    for ct in ["step_enriched", "step_narrative", "step_table"]:
        if ct in chunks_by_type:
            return chunks_by_type[ct].get("search_text", "")
    # Для рекомендаций — берём что есть
    for ct in REC_CHUNK_TYPES:
        if ct in chunks_by_type:
            return chunks_by_type[ct].get("search_text", "")
    return ""


def search_one_pair(client, biencoder, crossencoder, chunks_by_type, top_k=50, rerank_top_n=10):
    """
    Полный поиск для одного pair_id:
    1. KNN top-K для каждого типа чанка
    2. Union + дедупликация
    3. Cross-encoder reranking
    """
    all_hits = []

    for chunk_type, chunk in chunks_by_type.items():
        query_text = chunk.get("search_text", "")
        if not query_text:
            continue

        # Генерируем эмбеддинг и ищем
        embedding = biencoder.encode(query_text, normalize_embeddings=True).tolist()
        hits = knn_search(client, embedding, top_k)
        all_hits.append((chunk_type, hits))

    # Union всех результатов
    candidates = union_candidates(all_hits)

    if not candidates:
        return [], []

    # Cross-encoder reranking: используем лучший текст запроса
    query_for_rerank = get_best_query_text(chunks_by_type)
    reranked = rerank_candidates(crossencoder, query_for_rerank, candidates, rerank_top_n)

    return candidates, reranked


def run_etalon_test(client, biencoder, crossencoder, asis_chunks):
    """Тест на эталонных парах из БП 06.01."""
    print("\n" + "=" * 80)
    print("ЭТАЛОННЫЙ ТЕСТ: union 3 типов + cross-encoder reranking")
    print("=" * 80)

    # Группируем AS-IS чанки по pair_id и chunk_type
    by_pair = defaultdict(dict)
    for c in asis_chunks:
        pid = c["metadata"].get("pair_id", "")
        ct = c["metadata"].get("chunk_type", "")
        by_pair[pid][ct] = c

    stats_union = {"top5": 0, "top10": 0, "top30": 0, "total": 0, "miss": 0}
    stats_rerank = {"top1": 0, "top3": 0, "top5": 0, "total": 0, "miss": 0}

    for pair_id, expected_banks, desc in ETALON:
        chunks_by_type = by_pair.get(pair_id, {})
        if not chunks_by_type:
            print(f"\n{pair_id}: НЕТ ЧАНКОВ")
            continue

        # Поиск
        candidates, reranked = search_one_pair(
            client, biencoder, crossencoder, chunks_by_type,
            top_k=KNN_TOP_K, rerank_top_n=RERANK_TOP_N
        )

        # Все bank_id из union
        union_ids = [c["bank_id"] for c in candidates]
        # Все bank_id после reranking
        rerank_ids = [c["bank_id"] for c in reranked]

        print(f"\n{'=' * 80}")
        print(f"{pair_id}: {desc}")
        print(f"Эталон: {expected_banks}")
        print(f"Union: {len(candidates)} кандидатов")

        for exp in expected_banks:
            stats_union["total"] += 1
            stats_rerank["total"] += 1

            # Позиция в union
            if exp in union_ids:
                upos = union_ids.index(exp) + 1
                if upos <= 5: stats_union["top5"] += 1
                if upos <= 10: stats_union["top10"] += 1
                if upos <= 30: stats_union["top30"] += 1

                # Найден каким типом
                cand = candidates[upos - 1]
                found_by = cand.get("found_by", [])
            else:
                upos = None
                stats_union["miss"] += 1
                found_by = []

            # Позиция после reranking
            if exp in rerank_ids:
                rpos = rerank_ids.index(exp) + 1
                rscore = reranked[rpos - 1]["rerank_score"]
                if rpos <= 1: stats_rerank["top1"] += 1
                if rpos <= 3: stats_rerank["top3"] += 1
                if rpos <= 5: stats_rerank["top5"] += 1
            else:
                rpos = None
                rscore = None
                if upos is not None:
                    # Был в union, но не попал в top-N rerank
                    pass
                else:
                    stats_rerank["miss"] += 1

            ustr = f"union@{upos} ({', '.join(found_by)})" if upos else "union@MISS"
            rstr = f"rerank@{rpos} (score={rscore:.3f})" if rpos else "rerank@MISS"
            print(f"  {exp}: {ustr} → {rstr}")

        # Top-5 после reranking
        print(f"  --- Top-5 после cross-encoder ---")
        for i, c in enumerate(reranked[:5]):
            marker = " <<<" if c["bank_id"] in expected_banks else ""
            print(f"    {i+1}. {c['bank_id']} (rerank={c['rerank_score']:.3f}, "
                  f"knn={c['best_score']:.4f}, by={c['found_by']}){marker}")

    # Итог
    print(f"\n{'=' * 80}")
    print(f"ИТОГО (из {stats_union['total']} эталонных пар):")
    print(f"")
    print(f"  UNION (до reranking):")
    t = stats_union['total']
    print(f"    top-5:  {stats_union['top5']}/{t}")
    print(f"    top-10: {stats_union['top10']}/{t}")
    print(f"    top-30: {stats_union['top30']}/{t}")
    print(f"    MISS:   {stats_union['miss']}/{t}")
    print(f"")
    print(f"  CROSS-ENCODER (после reranking, top-{RERANK_TOP_N}):")
    t = stats_rerank['total']
    print(f"    top-1:  {stats_rerank['top1']}/{t}")
    print(f"    top-3:  {stats_rerank['top3']}/{t}")
    print(f"    top-5:  {stats_rerank['top5']}/{t}")
    print(f"    MISS:   {stats_rerank['miss']}/{t}")
    print(f"{'=' * 80}")


def run_full_search(client, biencoder, crossencoder, asis_chunks, bank_chunks, output_path):
    """Полный прогон по всем AS-IS чанкам."""
    print("\nПолный поиск v4...")

    # Группируем AS-IS по pair_id
    by_pair = defaultdict(dict)
    for c in asis_chunks:
        pid = c["metadata"].get("pair_id", "")
        ct = c["metadata"].get("chunk_type", "")
        priority = c["metadata"].get("mapping_priority", "low")
        if priority in ("high", "medium"):
            by_pair[pid][ct] = c

    # Дедупликация: для каждого pair_id оставляем step-типы и rec-типы
    search_pairs = {}
    for pid, chunks in by_pair.items():
        step_types = {k: v for k, v in chunks.items() if k in SEARCH_CHUNK_TYPES}
        rec_types = {k: v for k, v in chunks.items() if k in REC_CHUNK_TYPES}
        if step_types:
            search_pairs[pid] = step_types
        elif rec_types:
            search_pairs[pid] = rec_types

    print(f"  Pair_ids для поиска: {len(search_pairs)}")

    results = []
    total = len(search_pairs)

    for i, (pid, chunks_by_type) in enumerate(sorted(search_pairs.items())):
        candidates, reranked = search_one_pair(
            client, biencoder, crossencoder, chunks_by_type,
            top_k=KNN_TOP_K, rerank_top_n=RERANK_TOP_N
        )

        bp_id = ""
        for c in chunks_by_type.values():
            bp_id = c["metadata"].get("bp_id", "")
            if bp_id:
                break

        for rank, cand in enumerate(reranked):
            results.append({
                "pair_id": pid,
                "bank_id": cand["bank_id"],
                "bp_id": bp_id,
                "rerank_score": round(cand["rerank_score"], 4),
                "knn_score": round(cand["best_score"], 4),
                "found_by": cand["found_by"],
                "num_sources": cand["num_sources"],
                "rerank_position": rank + 1,
            })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] обработано")

    # Сохраняем
    output = {
        "metadata": {
            "version": "v4",
            "method": "multi-type union + cross-encoder reranking",
            "knn_top_k": KNN_TOP_K,
            "rerank_top_n": RERANK_TOP_N,
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
    parser = argparse.ArgumentParser(description="Поиск v4: multi-type union + cross-encoder")
    parser.add_argument("--mode", choices=["etalon", "full"], default="etalon",
                        help="etalon = тест на 7 парах, full = полный прогон")
    parser.add_argument("--asis", default="output/chunks_bp-06.json")
    parser.add_argument("--bank", default="output/chunks_bank.json")
    parser.add_argument("--output", default="output/search_v4_results.json")
    args = parser.parse_args()

    # Загрузка моделей
    biencoder = load_biencoder()
    crossencoder = load_crossencoder()
    client = create_opensearch_client()

    # Загрузка данных
    with open(args.asis) as f:
        asis_chunks = json.load(f)
    print(f"AS-IS чанков: {len(asis_chunks)}")

    if args.mode == "etalon":
        run_etalon_test(client, biencoder, crossencoder, asis_chunks)
    else:
        with open(args.bank) as f:
            bank_chunks = json.load(f)
        print(f"Банк чанков: {len(bank_chunks)}")
        run_full_search(client, biencoder, crossencoder, asis_chunks, bank_chunks, args.output)


if __name__ == "__main__":
    main()