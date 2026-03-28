"""
experiments.py — Два эксперимента для диагностики поиска.

Эксперимент 1: Union трёх типов чанков + анализ секций
Эксперимент 3: Exact cosine search (brute-force) вместо HNSW KNN

  python scripts/experiments.py --exp 1
  python scripts/experiments.py --exp 3
  python scripts/experiments.py --exp all
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch


OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_BANK = "processscout_bank"

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
    clean = bank_id.replace("FB-", "")
    clean = re.sub(r'[а-яё]\)$', '', clean)
    parts = clean.split(".")
    return ".".join(parts[:2])


def create_client():
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60
    )
    print(f"OpenSearch: {client.info()['version']['number']}")
    return client


def load_asis():
    with open("output/chunks_bp-06.json") as f:
        chunks = json.load(f)
    by_pair = defaultdict(dict)
    for c in chunks:
        pid = c["metadata"].get("pair_id", "")
        ct = c["metadata"].get("chunk_type", "")
        by_pair[pid][ct] = c
    return by_pair


def knn_search(client, embedding, top_k=50):
    result = client.search(index=INDEX_BANK, body={
        "size": top_k,
        "query": {"knn": {"embedding": {"vector": embedding, "k": top_k}}}
    })
    return result["hits"]["hits"]


def exact_cosine_search(client, embedding, top_k=50):
    result = client.search(index=INDEX_BANK, body={
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "embedding",
                        "query_value": embedding,
                        "space_type": "innerproduct"
                    }
                }
            }
        }
    })
    return result["hits"]["hits"]


def check_positions(hits, expected_banks):
    hit_ids = [h["_id"] for h in hits]
    results = {}
    for exp in expected_banks:
        if exp in hit_ids:
            results[exp] = hit_ids.index(exp) + 1
        else:
            results[exp] = None
    return results


def print_section_analysis(all_hits_by_type, expected_banks):
    union = {}
    for ctype, hits in all_hits_by_type.items():
        for h in hits:
            bid = h["_id"]
            if bid not in union:
                union[bid] = {"score": h["_score"], "types": set()}
            union[bid]["types"].add(ctype)

    section_counter = Counter()
    for bid, info in union.items():
        section = get_section_l2(bid)
        section_counter[section] += len(info["types"])

    print(f"  Union: {len(union)} уникальных кандидатов")
    print(f"  Top-5 секций:")
    for sec, weight in section_counter.most_common(5):
        etalon_in = [e for e in expected_banks if get_section_l2(e) == sec]
        marker = f" ← ЭТАЛОН: {etalon_in}" if etalon_in else ""
        print(f"    {sec}: вес {weight}{marker}")

    for exp in expected_banks:
        exp_sec = get_section_l2(exp)
        rank = None
        for i, (sec, _) in enumerate(section_counter.most_common()):
            if sec == exp_sec:
                rank = i + 1
                break
        in_union = exp in union
        found_by = sorted(union[exp]["types"]) if in_union else []
        print(f"  {exp} (секция {exp_sec}): "
              f"секция на месте {rank}/{len(section_counter)}, "
              f"{'в union (' + ','.join(found_by) + ')' if in_union else 'НЕ В UNION'}")


def experiment_1(client, model, by_pair):
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ 1: Union трёх типов + анализ секций")
    print("=" * 80)

    search_types = ["step_table", "step_enriched", "step_narrative"]
    total = 0
    found_union = 0

    for pair_id, expected, desc in ETALON:
        chunks = by_pair.get(pair_id, {})
        print(f"\n{'=' * 70}")
        print(f"{pair_id}: {desc}")
        print(f"Эталон: {expected}")

        all_hits_by_type = {}
        for ct in search_types:
            chunk = chunks.get(ct)
            if not chunk:
                continue
            query = chunk["search_text"]
            emb = model.encode(query, normalize_embeddings=True).tolist()
            hits = knn_search(client, emb, 50)
            all_hits_by_type[ct] = hits

            positions = check_positions(hits, expected)
            pos_str = ", ".join(f"{e}@{p if p else 'MISS'}" for e, p in positions.items())
            print(f"  [{ct:15}]: {pos_str}")

        print_section_analysis(all_hits_by_type, expected)

        union_ids = set()
        for hits in all_hits_by_type.values():
            union_ids.update(h["_id"] for h in hits)

        for exp in expected:
            total += 1
            if exp in union_ids:
                found_union += 1

    print(f"\n{'=' * 80}")
    print(f"ИТОГО: Union recall = {found_union}/{total} ({found_union/total*100:.0f}%)")
    print(f"{'=' * 80}")


def experiment_3(client, model, by_pair):
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ 3: Exact cosine (brute-force) vs HNSW KNN")
    print("=" * 80)

    search_types = ["step_table", "step_enriched", "step_narrative"]
    total = 0
    knn_found = 0
    exact_found = 0

    for pair_id, expected, desc in ETALON:
        chunks = by_pair.get(pair_id, {})
        print(f"\n{pair_id}: {desc}")

        for ct in search_types:
            chunk = chunks.get(ct)
            if not chunk:
                continue

            query = chunk["search_text"]
            emb = model.encode(query, normalize_embeddings=True).tolist()

            knn_hits = knn_search(client, emb, 50)
            knn_positions = check_positions(knn_hits, expected)

            exact_hits = exact_cosine_search(client, emb, 50)
            exact_positions = check_positions(exact_hits, expected)

            diffs = []
            for exp in expected:
                kp = knn_positions.get(exp)
                ep = exact_positions.get(exp)

                kstr = f"@{kp}" if kp else "@MISS"
                estr = f"@{ep}" if ep else "@MISS"
                diff = ""
                if not kp and ep:
                    diff = " !!! HNSW ПОТЕРЯЛ !!!"
                elif kp and ep and kp != ep:
                    diff = f" (сдвиг {kp-ep:+d})"
                diffs.append(f"{exp}: HNSW{kstr} Exact{estr}{diff}")

            print(f"  [{ct:15}]: {'; '.join(diffs)}")

        # Считаем union по обоим методам (narrative как основной)
        for ct in ["step_narrative", "step_enriched", "step_table"]:
            chunk = chunks.get(ct)
            if not chunk:
                continue
            emb = model.encode(chunk["search_text"], normalize_embeddings=True).tolist()
            knn_h = knn_search(client, emb, 50)
            exact_h = exact_cosine_search(client, emb, 50)
            for exp in expected:
                total += 1
                if exp in [h["_id"] for h in knn_h]:
                    knn_found += 1
                if exp in [h["_id"] for h in exact_h]:
                    exact_found += 1
            break  # только один тип для итоговой статистики

    print(f"\n{'=' * 80}")
    print(f"ИТОГО (narrative/enriched, {total} эталонных):")
    print(f"  HNSW KNN:     {knn_found}/{total} ({knn_found/total*100:.0f}%)")
    print(f"  Exact cosine: {exact_found}/{total} ({exact_found/total*100:.0f}%)")
    if exact_found > knn_found:
        print(f"  !!! Exact нашёл на {exact_found - knn_found} больше !!!")
    elif exact_found == knn_found:
        print(f"  Одинаково — HNSW не теряет")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", help="1, 3, or all")
    args = parser.parse_args()

    print("Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")
    client = create_client()
    by_pair = load_asis()

    exps = [args.exp] if args.exp != "all" else ["1", "3"]

    if "1" in exps:
        experiment_1(client, model, by_pair)
    if "3" in exps:
        experiment_3(client, model, by_pair)


if __name__ == "__main__":
    main()