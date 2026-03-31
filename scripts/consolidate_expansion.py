"""
consolidate_expansion.py — Консолидация результатов раскрытия из всех направлений.

Объединяет результаты из:
  - expansion_d1.json (направление 1: динамические кластеры)
  - expansion_d2.json (направление 2: статические кластеры банка)
  - expansion_d3.json (направление 3: декомпозиция формулировки) [когда появится]

АЛГОРИТМ:
  1. Для каждого подтверждённого маппинга (pair_id → bank_id):
     - Собираем всех кандидатов из всех направлений
     - Дедупликация по bank_id
     - Агрегация вердиктов: если "нужен" хотя бы в одном направлении → "нужен"
     - Подсчёт: в скольких направлениях кандидат найден, сколько "нужен"/"контекст"
  2. Ранжирование: нужен (по кол-ву направлений) > контекст > не нужен
  3. Исключаем сам подтверждённый пункт из раскрытия

ЗАПУСК:
  python scripts/consolidate_expansion.py \\
    --d1 output/expansion_d1.json \\
    --d2 output/expansion_d2.json \\
    --output output/expansion_consolidated.json

  # Только одно направление (остальные опциональны):
  python scripts/consolidate_expansion.py \\
    --d1 output/expansion_d1.json \\
    --output output/expansion_consolidated.json
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict


def extract_l3(bank_id: str) -> str:
    """FB-4.7.3 → 4.7.3"""
    clean = bank_id.replace("FB-", "")
    clean = re.sub(r'[а-ед]\)$', '', clean)
    parts = clean.split(".")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    elif len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def load_direction(path: str, direction_name: str) -> list[dict]:
    """
    Загружает результаты одного направления.
    Возвращает список {pair_id, confirmed_id, confirmed_desc, operation, neighbors[]}.
    Нормализует формат — d1 и d2 хранят данные чуть по-разному.
    """
    if not path or not Path(path).exists():
        return []

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # d1 — это список напрямую
    # d2 — это {"metadata": {...}, "expansions": [...]}
    if isinstance(raw, list):
        expansions = raw
    elif isinstance(raw, dict) and "expansions" in raw:
        expansions = raw["expansions"]
    else:
        print(f"  Неизвестный формат {path}")
        return []

    # Добавляем имя направления к каждому соседу
    for exp in expansions:
        for n in exp.get("neighbors", []):
            n["_direction"] = direction_name

    print(f"  {direction_name}: {len(expansions)} маппингов, "
          f"{sum(len(e.get('neighbors', [])) for e in expansions)} соседей")
    return expansions


def consolidate(all_directions: dict[str, list[dict]]) -> list[dict]:
    """
    Объединяет результаты из всех направлений.

    Возвращает список consolidated:
    [
      {
        "pair_id": "BP-06.01.01",
        "confirmed_id": "FB-4.7",
        "confirmed_desc": "...",
        "operation": "...",
        "total_candidates": N,
        "candidates": [
          {
            "id": "FB-4.7.1",
            "l3": "4.7.1",
            "text": "...",
            "verdict": "нужен",       # лучший вердикт из всех направлений
            "n_directions": 2,         # в скольких направлениях найден
            "directions": ["d1", "d2"],
            "verdicts_by_direction": {"d1": "нужен", "d2": "контекст"},
            "explanations_by_direction": {"d1": "...", "d2": "..."},
          },
          ...
        ]
      }
    ]
    """
    # Группируем по (pair_id, confirmed_id) из всех направлений
    # Ключ: (pair_id, confirmed_id) → {meta + neighbors по направлениям}
    mapping_key_to_data = {}

    for dir_name, expansions in all_directions.items():
        for exp in expansions:
            key = (exp["pair_id"], exp["confirmed_id"])
            if key not in mapping_key_to_data:
                mapping_key_to_data[key] = {
                    "pair_id": exp["pair_id"],
                    "confirmed_id": exp["confirmed_id"],
                    "confirmed_desc": exp.get("confirmed_desc", ""),
                    "operation": exp.get("operation", ""),
                    "neighbors_by_direction": {},
                }
            mapping_key_to_data[key]["neighbors_by_direction"][dir_name] = exp.get("neighbors", [])

    # Для каждого маппинга: дедупликация и агрегация
    verdict_priority = {"нужен": 3, "контекст": 2, "не нужен": 1, "": 0}
    results = []

    for key, data in sorted(mapping_key_to_data.items()):
        pair_id = data["pair_id"]
        confirmed_id = data["confirmed_id"]

        # Собираем всех кандидатов по bank_id
        candidates_map = {}  # bank_id → aggregated data

        for dir_name, neighbors in data["neighbors_by_direction"].items():
            for n in neighbors:
                nid = n.get("id", "")
                if not nid or nid == confirmed_id:
                    continue  # пропускаем сам подтверждённый пункт

                if nid not in candidates_map:
                    candidates_map[nid] = {
                        "id": nid,
                        "l3": extract_l3(nid),
                        "text": n.get("text", ""),
                        "directions": [],
                        "verdicts_by_direction": {},
                        "explanations_by_direction": {},
                    }

                c = candidates_map[nid]
                c["directions"].append(dir_name)

                verdict = n.get("llm_verdict", "")
                explanation = n.get("llm_explanation", "")

                c["verdicts_by_direction"][dir_name] = verdict
                if explanation:
                    c["explanations_by_direction"][dir_name] = explanation

                # Обновляем текст если пустой
                if not c["text"] and n.get("text", ""):
                    c["text"] = n["text"]

        # Агрегация: лучший вердикт, кол-во направлений
        candidates_list = []
        for nid, c in candidates_map.items():
            verdicts = list(c["verdicts_by_direction"].values())
            # Лучший вердикт: нужен > контекст > не нужен > пусто
            best_verdict = max(verdicts, key=lambda v: verdict_priority.get(v, 0)) if verdicts else ""
            # Считаем сколько направлений дали "нужен"
            n_needed = sum(1 for v in verdicts if v == "нужен")
            n_context = sum(1 for v in verdicts if v == "контекст")

            c["verdict"] = best_verdict
            c["n_directions"] = len(set(c["directions"]))
            c["n_needed"] = n_needed
            c["n_context"] = n_context
            candidates_list.append(c)

        # Ранжирование:
        # 1. verdict priority (нужен > контекст > не нужен)
        # 2. количество направлений с "нужен"
        # 3. общее количество направлений
        candidates_list.sort(key=lambda c: (
            -verdict_priority.get(c["verdict"], 0),
            -c["n_needed"],
            -c["n_directions"],
        ))

        results.append({
            "pair_id": pair_id,
            "confirmed_id": confirmed_id,
            "confirmed_desc": data["confirmed_desc"],
            "operation": data["operation"],
            "total_candidates": len(candidates_list),
            "n_needed": sum(1 for c in candidates_list if c["verdict"] == "нужен"),
            "n_context": sum(1 for c in candidates_list if c["verdict"] == "контекст"),
            "n_not_needed": sum(1 for c in candidates_list if c["verdict"] == "не нужен"),
            "n_no_verdict": sum(1 for c in candidates_list if c["verdict"] == ""),
            "candidates": candidates_list,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Консолидация результатов раскрытия из всех направлений"
    )
    parser.add_argument("--d1", default=None,
                        help="expansion_d1.json (динамические кластеры)")
    parser.add_argument("--d2", default=None,
                        help="expansion_d2.json (статические кластеры банка)")
    parser.add_argument("--d3", default=None,
                        help="expansion_d3.json (декомпозиция формулировки)")
    parser.add_argument("--output", default="output/expansion_consolidated.json")
    args = parser.parse_args()

    print("Загрузка направлений:")
    all_directions = {}
    if args.d1:
        all_directions["d1"] = load_direction(args.d1, "d1")
    if args.d2:
        all_directions["d2"] = load_direction(args.d2, "d2")
    if args.d3:
        all_directions["d3"] = load_direction(args.d3, "d3")

    if not all_directions:
        print("Ошибка: укажите хотя бы одно направление (--d1, --d2, --d3)")
        return

    print(f"\nКонсолидация...")
    results = consolidate(all_directions)

    # Сохранение
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "directions_used": list(all_directions.keys()),
            "n_mappings": len(results),
        },
        "consolidated": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Сохранено: {out_path}")

    # Сводка
    print(f"\n{'='*60}")
    print("СВОДКА консолидации раскрытия:")
    for r in results:
        print(f"\n  {r['pair_id']} → {r['confirmed_id']}: {r['confirmed_desc'][:40]}")
        print(f"    Кандидатов: {r['total_candidates']}, "
              f"нужен: {r['n_needed']}, контекст: {r['n_context']}, "
              f"не нужен: {r['n_not_needed']}, без оценки: {r['n_no_verdict']}")

        # Показываем "нужен"
        needed = [c for c in r["candidates"] if c["verdict"] == "нужен"]
        for c in needed[:5]:
            dirs = ",".join(sorted(c["verdicts_by_direction"].keys()))
            n_str = "/".join(c["verdicts_by_direction"].get(d, "?") for d in sorted(c["verdicts_by_direction"].keys()))
            print(f"      ✓ {c['id']}: [{dirs}]={n_str} — {c['text'][:60]}")


if __name__ == "__main__":
    main()