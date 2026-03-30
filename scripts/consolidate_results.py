"""
consolidate_results.py — Консолидация результатов двухступенчатого LLM
по 3 типам чанков (narrative, enriched, clean) в единый маппинг.

Алгоритм:
  1. Группирует результаты по pair_id (шаг БП)
  2. Собирает все primary + secondary из всех типов чанков
  3. Дедуплицирует по ID банковского пункта
  4. Группирует по секции L3 (первые 3 сегмента ID: 4.7.3, 5.13.1 и т.д.)
  5. Ранжирует секции по количеству «голосов» (сколько типов чанков нашли)
  6. Классифицирует: основной / дополнительный / неочевидный
  7. Выбирает представителя из каждой секции (предпочитает родителя)

Использование:
  python consolidate_results.py --input output/test_2step/results_clustered.json
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


def extract_l3_section(bank_id: str) -> str:
    """
    Извлекает секцию L3 из ID банковского пункта.
    FB-4.7.3 → 4.7.3
    FB-4.6.2.6 → 4.6.2
    FB-5.13.1 → 5.13.1
    FB-5.5.1.1 → 5.5.1
    FB-4.10.1.2 → 4.10.1
    FB-4.3 → 4.3
    FB-4.16 → 4.16
    """
    clean = bank_id.replace("FB-", "")
    clean = re.sub(r'[а-ед]\)$', '', clean)
    parts = clean.split(".")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    elif len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def is_parent_of(parent_id: str, child_id: str) -> bool:
    """Проверяет, является ли parent_id родителем child_id."""
    child_clean = re.sub(r'[а-яА-Я]\)$', '', child_id)
    return child_clean.startswith(parent_id + ".") or child_clean.startswith(parent_id + "а")


def choose_representative(candidates: list[dict]) -> dict:
    """
    Выбирает представителя из списка кандидатов одной секции.
    Правило: если есть родитель и ребёнок — предпочитаем родителя.
    При равенстве — тот, у кого больше голосов.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Сортируем: короче ID = выше в иерархии, при равенстве — больше голосов
    sorted_cands = sorted(candidates, key=lambda c: (len(c["id"]), -c["total_votes"]))
    top = sorted_cands[0]

    # Если самый короткий — родитель всех остальных, берём его
    all_children = all(is_parent_of(top["id"], other["id"]) for other in sorted_cands[1:])
    if all_children:
        return top

    # Иначе — тот, у кого больше голосов (primary весомее)
    return max(candidates, key=lambda c: (c["total_votes"], c["primary_votes"], -len(c["id"])))


def consolidate(results: list[dict]) -> list[dict]:
    """Основная функция: группировка → дедупликация → ранжирование."""

    # Группировка по pair_id
    by_pair = defaultdict(list)
    for r in results:
        pid = r["chunk_meta"].get("pair_id", "")
        if pid:
            by_pair[pid].append(r)

    consolidated = []

    for pair_id in sorted(by_pair.keys()):
        entries = by_pair[pair_id]

        # Описание шага из первого непустого чанка
        operation = ""
        bp_name = ""
        for e in entries:
            op = e["chunk_meta"].get("operation", "")
            if op:
                operation = op
                bp_name = e["chunk_meta"].get("bp_name", "")
                break

        # Агрегация кандидатов: bank_id → сводка
        cand_map = defaultdict(lambda: {
            "id": "",
            "chunk_text_plain": "",
            "relevance_votes": [],
            "chunk_types_found": set(),
            "primary_votes": 0,
            "secondary_votes": 0,
            "total_votes": 0,
            "explanations": [],
            "cluster_ranks": [],
        })

        for entry in entries:
            if entry.get("status") != "ok":
                continue

            ct = entry["chunk_meta"].get("chunk_type", "")
            chunk_id = entry.get("chunk_id", "")

            # Label типа чанка
            if "clean-1" in chunk_id:
                ct_label = "clean-1"
            elif "clean-2" in chunk_id:
                ct_label = "clean-2"
            elif ct == "step_narrative":
                ct_label = "narrative"
            elif ct == "step_enriched":
                ct_label = "enriched"
            elif ct == "step_clean":
                ct_label = "clean"
            else:
                ct_label = ct

            result = entry.get("result", {})
            if not result:
                continue

            # Primary
            best = result.get("best_document", {})
            if best and best.get("id"):
                bid = best["id"]
                c = cand_map[bid]
                c["id"] = bid
                if not c["chunk_text_plain"]:
                    c["chunk_text_plain"] = best.get("chunk_text_plain", "")
                c["relevance_votes"].append((ct_label, best.get("relevance", "прямое")))
                c["chunk_types_found"].add(ct_label)
                c["primary_votes"] += 1
                c["total_votes"] += 1
                c["cluster_ranks"].append(best.get("cluster_rank", 0))
                expl = best.get("step2_explanation", "")
                if expl:
                    c["explanations"].append(f"[{ct_label}] {expl}")

            # Secondary
            for sec in result.get("secondary_documents", []):
                if sec and sec.get("id"):
                    sid = sec["id"]
                    c = cand_map[sid]
                    c["id"] = sid
                    if not c["chunk_text_plain"]:
                        c["chunk_text_plain"] = sec.get("chunk_text_plain", "")
                    c["relevance_votes"].append((ct_label, sec.get("relevance", "дополнительный")))
                    c["chunk_types_found"].add(ct_label)
                    c["secondary_votes"] += 1
                    c["total_votes"] += 1
                    c["cluster_ranks"].append(sec.get("cluster_rank", 0))
                    expl = sec.get("step2_explanation", "")
                    if expl:
                        c["explanations"].append(f"[{ct_label}] {expl}")

        # Преобразуем в список, добавляем L3 секцию
        all_cands = list(cand_map.values())
        for c in all_cands:
            c["chunk_types_found"] = sorted(c["chunk_types_found"])
            c["l3_section"] = extract_l3_section(c["id"])

        # Группировка по секции L3
        by_section = defaultdict(list)
        for c in all_cands:
            by_section[c["l3_section"]].append(c)

        # Для каждой секции: суммируем, выбираем представителя, классифицируем
        sections = []
        for section_id, section_cands in by_section.items():
            total_v = sum(c["total_votes"] for c in section_cands)
            primary_v = sum(c["primary_votes"] for c in section_cands)
            secondary_v = sum(c["secondary_votes"] for c in section_cands)
            all_types = set()
            for c in section_cands:
                all_types.update(c["chunk_types_found"])

            representative = choose_representative(section_cands)

            # Классификация
            if primary_v >= 2 or (primary_v >= 1 and total_v >= 2):
                category = "основной"
            elif primary_v >= 1 or total_v >= 2:
                category = "дополнительный"
            else:
                category = "неочевидный"

            sections.append({
                "section_id": section_id,
                "category": category,
                "representative_id": representative["id"],
                "representative_text": representative["chunk_text_plain"][:300],
                "total_votes": total_v,
                "primary_votes": primary_v,
                "secondary_votes": secondary_v,
                "chunk_types": sorted(all_types),
                "all_candidates_in_section": [c["id"] for c in section_cands],
                "explanations": representative["explanations"][:3],
            })

        # Сортировка: основной → дополнительный → неочевидный → по голосам
        cat_order = {"основной": 0, "дополнительный": 1, "неочевидный": 2}
        sections.sort(key=lambda s: (cat_order[s["category"]], -s["total_votes"]))

        consolidated.append({
            "pair_id": pair_id,
            "operation": operation,
            "bp_name": bp_name,
            "total_unique_candidates": len(all_cands),
            "total_sections": len(sections),
            "summary": {
                "основной": [s["representative_id"] for s in sections if s["category"] == "основной"],
                "дополнительный": [s["representative_id"] for s in sections if s["category"] == "дополнительный"],
                "неочевидный": [s["representative_id"] for s in sections if s["category"] == "неочевидный"],
            },
            "sections": sections,
        })

    return consolidated


def print_consolidated(consolidated: list[dict]):
    """Красивый вывод в консоль."""
    for item in consolidated:
        pid = item["pair_id"]
        op = item["operation"][:80]
        n_cands = item["total_unique_candidates"]
        n_sects = item["total_sections"]

        print(f"\n{'='*70}")
        print(f"{pid}: {op}")
        print(f"  {n_cands} кандидатов → {n_sects} секций")

        for s in item["sections"]:
            cat = s["category"].upper()
            rep = s["representative_id"]
            tv = s["total_votes"]
            pv = s["primary_votes"]
            sv = s["secondary_votes"]
            types = ", ".join(s["chunk_types"])
            all_in = ", ".join(s["all_candidates_in_section"])

            if cat == "ОСНОВНОЙ":
                marker = "●"
            elif cat == "ДОПОЛНИТЕЛЬНЫЙ":
                marker = "◐"
            else:
                marker = "○"

            print(f"  {marker} [{cat}] секция {s['section_id']}: {rep} "
                  f"({tv} голосов: {pv}P+{sv}S | типы: {types})")

            # Если в секции >1 кандидата — показать все
            if len(s["all_candidates_in_section"]) > 1:
                print(f"    все в секции: {all_in}")

            # Одно объяснение от LLM
            for expl in s["explanations"][:1]:
                print(f"    LLM: {expl[:120]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="results_clustered.json от 2-step пайплайна")
    parser.add_argument("--output", default=None, help="Путь для consolidated.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: {input_path}")
        return

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Загружено: {len(results)} результатов")

    step_results = [r for r in results if r.get("chunk_meta", {}).get("pair_id", "")]
    print(f"С pair_id: {len(step_results)}")

    consolidated = consolidate(step_results)
    print_consolidated(consolidated)

    output_path = Path(args.output) if args.output else input_path.parent / "consolidated.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {"source": str(input_path), "total_steps": len(consolidated)},
            "consolidated": consolidated,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nСохранено: {output_path}")


if __name__ == "__main__":
    main()