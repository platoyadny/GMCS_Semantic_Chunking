"""
consolidate_recs.py — Консолидация результатов маппинга для РЕКОМЕНДАЦИЙ.

В отличие от consolidate_results.py (для шагов), рекомендации:
  - Не имеют трёх типов чанков (narrative/enriched/clean)
  - Каждая рекомендация = один чанк = один результат
  - Нет голосования/мержа — результат берётся напрямую
  - pair_id = '?' (привязка через bp_id и chunk_id)

Скрипт читает results_clustered.json, фильтрует рекомендации (chunk_type
начинается с 'rec'), и формирует consolidated_recs.json в формате,
совместимом с consolidated_l3.json (для expand_direction1/2 и UI).

ЗАПУСК:
  python scripts/consolidate_recs.py \
    --input output/bp_06_04/mapping/results_clustered.json \
    --output output/bp_06_04/mapping/consolidated_recs.json

  # Батч для нескольких подпроцессов:
  for BP in 01 02 03 04; do
    python scripts/consolidate_recs.py \
      --input output/bp_06_${BP}/mapping/results_clustered.json \
      --output output/bp_06_${BP}/mapping/consolidated_recs.json
  done
"""

import json
import re
import argparse
from pathlib import Path


def extract_l3(bank_id: str) -> str:
    """
    Извлекает L3-секцию из ID банка.
    FB-4.7.3 → 4.7.3
    FB-4.7.1б) → 4.7.1
    FB-4.7 → 4.7
    """
    # Убираем префикс FB-
    clean = bank_id.replace("FB-", "")
    # Убираем буквенные суффиксы: а), б), в) и т.д.
    clean = re.sub(r'[а-ед]\)$', '', clean)
    # Берём до 3 уровней
    parts = clean.split(".")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    elif len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def consolidate_recs(input_path: str) -> list[dict]:
    """
    Читает results_clustered.json, извлекает рекомендации,
    формирует список в формате consolidated_l3.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    recs = []

    for r in data["results"]:
        # Фильтруем: только рекомендации с успешным статусом
        ct = r.get("chunk_meta", {}).get("chunk_type", "")
        if not ct.startswith("rec") or r.get("status") != "ok":
            continue

        res = r["result"]
        best = res["best_document"]
        secs = res.get("secondary_documents", [])

        # Формируем sections (как в consolidated_l3)
        sections = []

        # PRIMARY — основной кандидат
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
            "explanations": [best.get("gpt_explanation", "")],
        })

        # SECONDARY — дополнительные кандидаты
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
                "explanations": [s.get("gpt_explanation", "")],
            })

        # Итоговая запись для одной рекомендации
        recs.append({
            "pair_id": r["chunk_id"],  # например BP-06.04-rec-proc-01
            "operation": r.get("chunk_meta", {}).get("operation",
                         r.get("chunk_meta", {}).get("text", "")),
            "bp_name": r.get("chunk_meta", {}).get("bp_name", ""),
            "chunk_type": ct,  # rec_process или rec_automation
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
    parser = argparse.ArgumentParser(
        description="Консолидация результатов маппинга для рекомендаций"
    )
    parser.add_argument("--input", required=True,
                        help="Путь к results_clustered.json")
    parser.add_argument("--output", required=True,
                        help="Путь для consolidated_recs.json")
    args = parser.parse_args()

    # Консолидация
    recs = consolidate_recs(args.input)

    if not recs:
        print(f"Рекомендаций не найдено в {args.input}")
        return

    # Подсчёт по типам
    rec_proc = sum(1 for r in recs if r.get("chunk_type") == "rec_process")
    rec_auto = sum(1 for r in recs if r.get("chunk_type") == "rec_automation")

    # Сохранение
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

    # Краткая сводка
    for r in recs[:5]:
        print(f"  {r['pair_id']}: {r['summary']['основной'][0]} ← {r['operation'][:60]}")
    if len(recs) > 5:
        print(f"  ... и ещё {len(recs)-5}")


if __name__ == "__main__":
    main()