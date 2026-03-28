"""
dump_chunks.py — Выводит тексты трёх типов чанков для 7 эталонных БП.
  python scripts/dump_chunks.py
"""

import json
from collections import defaultdict

ETALON_PAIRS = [
    "BP-06.01.01", "BP-06.01.02", "BP-06.01.03",
    "BP-06.01.04", "BP-06.01.05", "BP-06.01.06", "BP-06.01.07"
]

ETALON_BANKS = {
    "BP-06.01.01": ["FB-4.7.1", "FB-4.16.1"],
    "BP-06.01.02": ["FB-4.7.3"],
    "BP-06.01.03": ["FB-4.3", "FB-4.7.2"],
    "BP-06.01.04": ["FB-4.6.2"],
    "BP-06.01.05": ["FB-5.13.1"],
    "BP-06.01.06": ["FB-4.10.1.2", "FB-4.6.8.4"],
    "BP-06.01.07": ["FB-4.10.1.1"],
}

ETALON_DESC = {
    "BP-06.01.01": "Формирование, согласование и утверждение годового номенклатурного плана производства товарной продукции",
    "BP-06.01.02": "Разработка сетевого графика выпуска товарной продукции на год",
    "BP-06.01.03": "Формирование прогнозного (годового) плана. Расчёт потребности в материальных ресурсах и производственных мощностях",
    "BP-06.01.04": "Выделение рабочих центров, планируемых с учётом ограничений мощности",
    "BP-06.01.05": "Определение процента выполнения производственного плана на год",
    "BP-06.01.06": "Расчёт загрузки рабочих центров согласно утвержденному справочнику группы рабочих центров. Определение собственной производственной возможности",
    "BP-06.01.07": "Оценка потребности в персонале. Оформление заявки на приём новых рабочих",
}

def main():
    with open("output/chunks_bp-06.json") as f:
        chunks = json.load(f)

    by_pair = defaultdict(dict)
    for c in chunks:
        pid = c["metadata"].get("pair_id", "")
        ct = c["metadata"].get("chunk_type", "")
        by_pair[pid][ct] = c

    for pid in ETALON_PAIRS:
        print("=" * 80)
        print(f"{pid}")
        print(f"Эталон: {ETALON_BANKS[pid]}")
        print(f"\n--- ОРИГИНАЛ (из документа обследования) ---")
        print(ETALON_DESC[pid])

        types = by_pair.get(pid, {})

        for ct in ["step_table", "step_enriched", "step_narrative"]:
            chunk = types.get(ct)
            if chunk:
                print(f"\n--- {ct} ---")
                print(chunk["search_text"])
            else:
                print(f"\n--- {ct} --- НЕТ")

        print()


if __name__ == "__main__":
    main()