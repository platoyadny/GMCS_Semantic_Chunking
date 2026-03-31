"""
consolidate_recs.py — Консолидация результатов маппинга для РЕКОМЕНДАЦИЙ.

В отличие от consolidate_results.py (для шагов), рекомендации:
  - Не имеют трёх типов чанков (narrative/enriched/clean)
  - Каждая рекомендация = один чанк = один результат
  - Нет голосования/мержа — результат берётся напрямую
  - pair_id = '?' (привязка через bp_id и chunk_id)

ЗАПУСК:
  python scripts/consolidate_recs.py \
    --input output/bp_06_04/mapping/results_clustered.json \
    --output output/bp_06_04/mapping/consolidated_recs.json
"""

import json
import re
import argparse
from pathlib import Path


def extract_l3(bank_id: str) -> str:
    clean = bank_id.replace("FB-", "")
    clean = re.sub(r'[а-ед]\)$', '', clean)
    parts = clean.split(".")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    elif len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def consolidate_recs(input_path: str) -> list[dict]:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    recs = []
    for r in data["results"]:
        ct = r.get("chunk_meta", {}).get("chunk_type", "")
        if not ct.startswith("rec") or r.get("status") != "ok":
            continue

        res = r["result"]
        best = res["best_document"]
        secs = res.get("secondary_documents", [])
        sections = []

        bid = best["id"]
        sections.append({
            "section_id": extract_l3(bid),
            "category": "основной",
            "representative_id": bid,
            "representative_text": best.get("chunk_text_plain", ""),
            "total_votes": 1,
            "primary_votes": 1,
            "secondary_votes": 0,
            "chunk_types": [ct],
            "all_candidates_in_section": [bid],
            "explanations": [best.get("step2_explanation", "")],
        })

        for s in secs:
            sid = s["id"]
            sections.append({
                "section_id": extract_l3(sid),
                "category": "дополнительный",
                "representative_id": sid,
                "representative_text": s.get("chunk_text_plain", ""),
                "total_votes": 1,
                "primary_votes": 0,
                "secondary_votes": 1,
                "chunk_types": [ct],
                "all_candidates_in_section": [sid],
                "explanations": [s.get("step2_explanation", "")],
            })

        recs.append({
            "pair_id": r["chunk_id"],
            "operation": r.get("chunk_meta", {}).get("operation",
                         r.get("chunk_meta", {}).get("text", "")),
            "bp_name": r.get("chunk_meta", {}).get("bp_name", ""),
            "chunk_type": ct,
            "total_unique_candidates": len(sections),
            "total_sections": len(sections),
            "summary": {
                "основной": [s["representative_id"] for s in sections if s["category"] == "основной"],
                "дополнительный": [s["representative_id"] for s in sections if s["category"] == "дополнительный"],
                "неочевидный": [],
            },
            "sections": sections,
        })

    return recs


def main():
    parser = argparse.ArgumentParser(description="Консолидация рекомендаций")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    recs = consolidate_recs(args.input)
    if not recs:
        print(f"Рекомендаций не найдено в {args.input}")
        return

    rec_proc = sum(1 for r in recs if r.get("chunk_type") == "rec_process")
    rec_auto = sum(1 for r in recs if r.get("chunk_type") == "rec_automation")

    output = {
        "metadata": {
            "source": "recommendations",
            "input_file": args.input,
            "total_recs": len(recs),
            "rec_process": rec_proc,
            "rec_automation": rec_auto,
        },
        "consolidated": recs,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Сохранено: {out_path}")
    print(f"  rec_process: {rec_proc}, rec_automation: {rec_auto}, всего: {len(recs)}")
    for r in recs[:3]:
        expl = r['sections'][0]['explanations'][0][:80] if r['sections'][0]['explanations'][0] else '(пусто)'
        print(f"  {r['pair_id']}: {r['summary']['основной'][0]} | LLM: {expl}")
    if len(recs) > 3:
        print(f"  ... и ещё {len(recs)-3}")


if __name__ == "__main__":
    main()
