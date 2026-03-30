"""
hybrid_search.py v3 — Двухпроходный гибридный поиск.

v3 изменения:
  - Фильтр глубины банка ≤4 сегмента в merge. Подбуквы (а,б,в,г) отсекаются.
  - Default top-k=15 (больше кандидатов для LLM-judge).
  - Chunk_type-aware merge (из v2).
"""

import json
import re
import argparse
import sys
from pathlib import Path
from collections import defaultdict

OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_ASIS = "processscout_asis"
INDEX_BANK = "processscout_bank"
SEARCH_PIPELINE = "hybrid_rrf_pipeline"

MAX_BANK_DEPTH = 4

CHUNK_TYPE_WEIGHTS = {
    "step_table": 0.5,
    "step_narrative": 0.4,
    "step_enriched": 0.1,
    "rec_automation": 1.0,
    "rec_process": 1.0,
    "rec_consulting": 1.0,
    "rec_general": 1.0,
    "narrative": 0.5,
    "card": 0.3,
    "asis_notes": 0.3,
    "intro": 0.2,
}

MIN_INTERNAL_SCORE = 0.3


def get_bank_depth(bank_id: str) -> int:
    """
    FB-4.13.2 → 3, FB-5.15.5.1 → 4, FB-4.16.3г) → 4 (подбуква = +1).
    """
    clean = bank_id.replace("FB-", "")
    has_subletter = bool(re.search(r'[а-яё]\)$', clean))
    segments = clean.split(".")
    depth = len(segments)
    if has_subletter:
        depth += 1
    return depth


def load_model():
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3...")
    return SentenceTransformer("BAAI/bge-m3")


def create_client():
    from opensearchpy import OpenSearch
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60
    )
    print(f"OpenSearch: {client.info()['version']['number']}")
    return client


def hybrid_search(client, index, query_text, query_embedding, text_fields, top_k=15):
    body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries": [
                    {"multi_match": {"query": query_text, "fields": text_fields, "type": "best_fields"}},
                    {"knn": {"embedding": {"vector": query_embedding, "k": top_k * 2}}}
                ]
            }
        }
    }
    response = client.search(index=index, body=body, params={"search_pipeline": SEARCH_PIPELINE})
    return response["hits"]["hits"]


def pass_asis_to_bank(client, model, asis_chunks, top_k):
    bank_fields = ["chunk_text_plain^3", "chunk_text_context^2"]
    search_chunks = [c for c in asis_chunks if c.get("metadata", {}).get("mapping_priority") in ("high", "medium")]

    print(f"\nПроход 1: AS-IS → банк ({len(search_chunks)} чанков)")
    results = {}
    total = len(search_chunks)

    for i, chunk in enumerate(search_chunks):
        query_text = chunk.get("search_text", "")
        if not query_text:
            continue
        embedding = model.encode(query_text, normalize_embeddings=True).tolist()
        hits = hybrid_search(client, INDEX_BANK, query_text, embedding, bank_fields, top_k)

        chunk_id = chunk.get("chunk_id", "")
        candidates = [{"id": h["_id"], "raw_score": round(h["_score"], 6), "source": h["_source"]} for h in hits]
        if candidates:
            results[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_type": chunk["metadata"].get("chunk_type", ""),
                "bp_id": chunk["metadata"].get("bp_id", ""),
                "pair_id": chunk["metadata"].get("pair_id", chunk_id),
                "query_text": query_text,
                "candidates": candidates
            }
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] обработано, найдено маппингов: {len(results)}")
    return results


def pass_bank_to_asis(client, model, bank_chunks, top_k):
    asis_fields = ["search_text^3", "metadata.operation^2", "metadata.bp_name^1"]
    print(f"\nПроход 2: банк → AS-IS ({len(bank_chunks)} чанков)")
    results = {}
    total = len(bank_chunks)

    for i, chunk in enumerate(bank_chunks):
        query_text = chunk.get("chunk_text_context", "") or chunk.get("chunk_text_plain", "")
        if not query_text:
            continue
        embedding = model.encode(query_text, normalize_embeddings=True).tolist()
        hits = hybrid_search(client, INDEX_ASIS, query_text, embedding, asis_fields, top_k)

        chunk_id = chunk.get("metadata", {}).get("chunk_id", chunk.get("chunk_id", ""))
        candidates = [{"id": h["_id"], "raw_score": round(h["_score"], 6), "source": h["_source"]} for h in hits]
        if candidates:
            results[chunk_id] = {"chunk_id": chunk_id, "query_text": query_text, "candidates": candidates}
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] обработано, найдено маппингов: {len(results)}")
    return results


def merge_results(pass1, pass2):
    pair_bank_types = defaultdict(lambda: {"chunk_types": set(), "bp_id": "", "pass2_found": False})

    for chunk_id, detail in pass1.items():
        pair_id = detail.get("pair_id", chunk_id)
        chunk_type = detail.get("chunk_type", "")
        bp_id = detail.get("bp_id", "")
        for candidate in detail["candidates"]:
            bank_id = candidate["id"]
            # Фильтр глубины: отсекаем подпункты глубже MAX_BANK_DEPTH
            if get_bank_depth(bank_id) > MAX_BANK_DEPTH:
                continue
            key = (pair_id, bank_id)
            pair_bank_types[key]["chunk_types"].add(chunk_type)
            pair_bank_types[key]["bp_id"] = bp_id

    for bank_id, detail in pass2.items():
        for candidate in detail["candidates"]:
            asis_id = candidate["id"]
            asis_meta = candidate.get("source", {}).get("metadata", {})
            pair_id = asis_meta.get("pair_id", asis_id)
            # Фильтр глубины для банковского ID
            if get_bank_depth(bank_id) > MAX_BANK_DEPTH:
                continue
            key = (pair_id, bank_id)
            pair_bank_types[key]["pass2_found"] = True
            if not pair_bank_types[key]["bp_id"]:
                pair_bank_types[key]["bp_id"] = asis_meta.get("bp_id", "")

    mappings = []
    for (pair_id, bank_id), info in pair_bank_types.items():
        chunk_types = info["chunk_types"]
        mutual = info["pass2_found"]
        internal_score = min(sum(CHUNK_TYPE_WEIGHTS.get(ct, 0.3) for ct in chunk_types), 1.0)
        final_score = internal_score * (1.0 if mutual else 0.5)
        if final_score < MIN_INTERNAL_SCORE:
            continue

        if final_score >= 0.7 and mutual:
            quality = "strong"
        elif final_score >= 0.4:
            quality = "medium"
        else:
            quality = "weak"

        mappings.append({
            "pair_id": pair_id, "bank_id": bank_id, "bp_id": info["bp_id"],
            "chunk_types_found": sorted(chunk_types), "num_types": len(chunk_types),
            "internal_score": round(final_score, 3), "mutual_confirmation": mutual, "quality": quality
        })

    quality_order = {"strong": 0, "medium": 1, "weak": 2}
    mappings.sort(key=lambda x: (quality_order[x["quality"]], -x["internal_score"]))
    return mappings


def print_stats(mappings):
    total = len(mappings)
    if not total:
        print("Нет результатов"); return
    mutual = sum(1 for m in mappings if m["mutual_confirmation"])
    strong = sum(1 for m in mappings if m["quality"] == "strong")
    medium = sum(1 for m in mappings if m["quality"] == "medium")
    weak = sum(1 for m in mappings if m["quality"] == "weak")
    unique_asis = len(set(m["pair_id"] for m in mappings))
    unique_bank = len(set(m["bank_id"] for m in mappings))
    by_types = defaultdict(int)
    for m in mappings:
        by_types[m["num_types"]] += 1
    enriched_only = sum(1 for m in mappings if m["chunk_types_found"] == ["step_enriched"])
    depth_dist = defaultdict(int)
    for m in mappings:
        depth_dist[get_bank_depth(m["bank_id"])] += 1

    print(f"\n{'='*60}")
    print(f"Результаты маппинга (v3: depth filter + chunk_type)")
    print(f"{'='*60}")
    print(f"  Всего пар:               {total}")
    print(f"  Mutual confirmation:     {mutual} ({mutual/total*100:.0f}%)")
    print(f"  Strong:                  {strong}")
    print(f"  Medium:                  {medium}")
    print(f"  Weak:                    {weak}")
    print(f"  Уникальных AS-IS:        {unique_asis}")
    print(f"  Уникальных банк:         {unique_bank}")
    print(f"\n  По chunk_type подтверждениям:")
    for n in sorted(by_types):
        print(f"    {n} тип(а/ов): {by_types[n]}")
    print(f"  Enriched-only: {enriched_only}")
    print(f"\n  По глубине банковского пункта:")
    for d in sorted(depth_dist):
        print(f"    {d} сегмента: {depth_dist[d]}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Двухпроходный гибридный поиск v3")
    parser.add_argument("--asis", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", default="output/mapping_results.json")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    model = load_model()
    client = create_client()
    with open(args.asis) as f:
        asis_chunks = json.load(f)
    with open(args.bank) as f:
        bank_chunks = json.load(f)

    print(f"AS-IS: {len(asis_chunks)}, банк: {len(bank_chunks)}")
    pass1 = pass_asis_to_bank(client, model, asis_chunks, args.top_k)
    pass2 = pass_bank_to_asis(client, model, bank_chunks, args.top_k)
    print(f"\nПроход 1: {len(pass1)} AS-IS с кандидатами")
    print(f"Проход 2: {len(pass2)} банк с кандидатами")

    print("\nОбъединение (v3: depth filter + chunk_type)...")
    mappings = merge_results(pass1, pass2)
    print_stats(mappings)

    output = {
        "metadata": {"version": "v3", "asis_file": args.asis, "bank_file": args.bank,
                      "top_k": args.top_k, "max_bank_depth": MAX_BANK_DEPTH,
                      "asis_total": len(asis_chunks), "bank_total": len(bank_chunks)},
        "pass1_details": pass1, "pass2_details": pass2, "mappings": mappings
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
