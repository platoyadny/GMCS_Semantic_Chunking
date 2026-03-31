#!/usr/bin/env python3
"""
prepare_data.py — Сборка данных для ProcessScout AI UI

Использование:
    python prepare_data.py \
        --data_dir output/ \
        --bank bank_4-5.xlsx \
        --output processscout_data.json

Реальная структура data_dir:
    output/
    ├── bp_06_01/ ... bp_06_14/
    │   ├── chunks/chunks_3types.json
    │   ├── mapping/
    │   │   ├── consolidated_l3.json           ← {"metadata":{...}, "consolidated":[...]}
    │   │   └── consolidated_recs.json         ← {"metadata":{...}, "consolidated":[...]}
    │   └── expansion/
    │       ├── expansion_consolidated.json     ← {"metadata":{...}, "consolidated":[...]}
    │       └── expansion_recs_consolidated.json
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import openpyxl
except ImportError:
    print("ОШИБКА: pip install openpyxl --break-system-packages")
    sys.exit(1)


# ================================================================
# 1. БАНК ФУНКЦИОНАЛЬНОСТИ
# ================================================================

def load_bank(bank_path: str) -> dict:
    """
    Читает bank_4-5.xlsx → { "4.7": { "text": "...", "path": "[4]...→[4.7]..." } }
    Файл без заголовков: col 0 = ID, col 1 = текст.
    """
    wb = openpyxl.load_workbook(bank_path, read_only=True, data_only=True)
    ws = wb.active

    # Шаг 1: плоский словарь { id → text }
    raw = {}
    for row in ws.iter_rows(min_row=1, values_only=True):
        item_id = str(row[0]).strip() if row[0] else None
        item_text = str(row[1]).strip() if row[1] else ""
        if not item_id:
            continue
        raw[item_id] = item_text
    wb.close()

    # Шаг 2: иерархические пути
    bank = {}
    for item_id, text in raw.items():
        parts = item_id.split(".")
        ancestors = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
        path_parts = []
        for anc_id in ancestors:
            anc_text = raw.get(anc_id, "")
            short = anc_text[:60] + ("..." if len(anc_text) > 60 else "")
            path_parts.append(f"[{anc_id}] {short}")
        bank[item_id] = {"text": text, "path": " → ".join(path_parts)}

    print(f"  Банк: {len(bank)} пунктов из {bank_path}")
    return bank


# ================================================================
# 2. JSON УТИЛИТЫ
# ================================================================

def load_json(filepath: str):
    """Безопасно читает JSON. Возвращает None если файл не найден."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def find_bp_folders(data_dir: str) -> list[tuple[str, str]]:
    """Находит папки bp_XX_YY → [("06.01", "output/bp_06_01"), ...]"""
    folders = []
    for item in sorted(Path(data_dir).iterdir()):
        if not item.is_dir():
            continue
        match = re.match(r"bp_(\d{2})_(\d{2})", item.name)
        if match:
            folders.append((f"{match.group(1)}.{match.group(2)}", str(item)))
    return folders


# ================================================================
# 3. ПАРСИНГ CONSOLIDATED (шаги и рекомендации)
#
# Формат файла: {"metadata": {...}, "consolidated": [...]}
# consolidated — массив объектов, каждый = один чанк (pair_id).
# Кандидаты лежат в sections[].
# ================================================================

def parse_consolidated(raw_data) -> dict:
    """
    Парсит consolidated_l3.json или consolidated_recs.json.

    Вход: dict {"metadata": {...}, "consolidated": [...]}
      или list [...] (массив напрямую)

    consolidated = [
      { "pair_id": "BP-06.01.01", "operation": "...", "bp_name": "...",
        "sections": [{ section_id, category, representative_id, representative_text,
                       total_votes, chunk_types, explanations }] }, ...
    ]

    Выход: dict { pair_id → parsed_entry }
    """
    if not raw_data:
        return {}

    # Извлекаем массив из обёртки {"metadata": ..., "consolidated": [...]}
    if isinstance(raw_data, dict):
        raw_data = raw_data.get("consolidated", [])

    result = {}
    for entry in raw_data:
        pair_id = entry.get("pair_id", "")
        if not pair_id:
            continue

        operation = entry.get("operation", "")
        bp_name = entry.get("bp_name", "")
        chunk_type = entry.get("chunk_type", "")  # только у рекомендаций
        sections = entry.get("sections", [])

        candidates = []
        for sec in sections:
            rep_id = sec.get("representative_id", "")
            if not rep_id:
                continue

            # Извлекаем LLM комментарий из нескольких источников
            # Приоритет: explanations[] → step2_explanation → step1_explanation
            explanations = sec.get("explanations", [])
            llm_comment = ""
            
            # 1. Из массива explanations (формат "[enriched] текст" или просто "текст")
            for expl in explanations:
                if not expl or not expl.strip():
                    continue
                # Убираем теги вида [enriched], [clean], [narrative], [step2] и т.д.
                cleaned = re.sub(r"^\[[^\]]*\]\s*", "", expl.strip())
                if cleaned and len(cleaned) > 5:
                    llm_comment = cleaned
                    break
            
            # 2. Fallback: поле step2_explanation напрямую
            if not llm_comment:
                s2 = sec.get("step2_explanation", "")
                if s2 and s2.strip() and len(s2.strip()) > 5:
                    llm_comment = s2.strip()
            
            # 3. Fallback: поле step1_explanation
            if not llm_comment:
                s1 = sec.get("step1_explanation", sec.get("_step1_explanation", ""))
                if s1 and s1.strip() and len(s1.strip()) > 5:
                    llm_comment = s1.strip()

            total_votes = sec.get("total_votes", 0)
            # Формируем votes строку: "2/3" для шагов
            votes_str = f"{total_votes}/3" if total_votes and not chunk_type else ""

            candidates.append({
                "id": rep_id,
                "category": sec.get("category", "неочевидный"),
                "votes": votes_str,
                "bank_text": sec.get("representative_text", ""),
                "bank_path": "",  # будет обогащён из банка ниже
                "llm_comment": llm_comment,
                "chunk_types": sec.get("chunk_types", []),
            })

        result[pair_id] = {
            "operation": operation,
            "bp_name": bp_name,
            "chunk_type": chunk_type,
            "candidates": candidates,
        }

    return result


# ================================================================
# 4. ПАРСИНГ EXPANSION
#
# Формат файла: {"metadata": {...}, "consolidated": [...]}
# consolidated — массив объектов, каждый = раскрытие одного confirmed_id.
# ================================================================

def parse_expansion(raw_data) -> dict:
    """
    Парсит expansion_consolidated.json или expansion_recs_consolidated.json.

    Вход: dict {"metadata": {...}, "consolidated": [...]}
      или list [...] (массив напрямую)

    consolidated = [
      { "pair_id": "BP-06.01.01", "confirmed_id": "FB-4.7",
        "candidates": [{ id, text, verdict, explanations_by_direction }] }
    ]

    Выход: двухуровневый dict { pair_id → { confirmed_id → [items] } }
    """
    if not raw_data:
        return {}

    # Извлекаем массив из обёртки {"metadata": ..., "consolidated": [...]}
    if isinstance(raw_data, dict):
        raw_data = raw_data.get("consolidated", [])

    result = {}
    for entry in raw_data:
        pair_id = entry.get("pair_id", "")
        confirmed_id = entry.get("confirmed_id", "")
        if not pair_id or not confirmed_id:
            continue

        if pair_id not in result:
            result[pair_id] = {}

        items = []
        for cand in entry.get("candidates", []):
            item_id = cand.get("id", "")
            verdict = cand.get("verdict", "")
            if not item_id or not verdict:
                continue

            direction = map_verdict_to_direction(verdict)
            if direction is None:
                continue  # пропускаем "не нужен"

            # Собираем лучший llm_comment из explanations_by_direction
            explanations = cand.get("explanations_by_direction", {})
            llm_comment = ""
            for d_key in ["d1", "d2", "d3"]:
                if explanations.get(d_key, "").strip():
                    llm_comment = explanations[d_key].strip()
                    break

            items.append({
                "id": item_id,
                "direction": direction,
                "bank_text": cand.get("text", ""),
                "bank_path": "",  # обогащается из банка ниже
                "llm_comment": llm_comment,
            })

        if items:
            result[pair_id][confirmed_id] = items

    return result


def map_verdict_to_direction(verdict: str) -> str | None:
    """нужен → need, контекст → context, не нужен → None"""
    v = verdict.lower().strip()
    if v in ("нужен", "нужно", "need"):
        return "need"
    elif v in ("контекст", "context"):
        return "context"
    elif v in ("не нужен", "не нужно", "not needed"):
        return None
    return "context"


# ================================================================
# 5. ОБОГАЩЕНИЕ ИЗ БАНКА
# ================================================================

def _trim_bank_text(text: str) -> str:
    """
    Обрезает текст банка до сути — убирает перечисления подпунктов.
    Примеры того что обрезается:
      "... Включает подпункты: - [4.7.1] ..."
      "... - [4.1.2.1] Описание - [4.1.2.2] ..."
      "...) - [4.7.1] Формирование ..."
      "...) [4.1.2.1] Поддержка ..."
    """
    # 1. "Включает подпункты" и всё после
    text = re.split(r'\s*Включает подпункты', text)[0].strip()
    # 2. "- [X.X.X" или "– [X.X.X" (начало перечисления детей через тире)
    text = re.split(r'\s*[-–]\s*\[\d+\.\d+', text)[0].strip()
    # 3. Просто "[X.X.X.X]" без тире (ссылка на подпункт в тексте)
    text = re.split(r'\s*\[\d+\.\d+\.\d+\]', text)[0].strip()
    # 4. Убираем хвостовые скобки/тире если остались
    text = text.rstrip(' -–,;:')
    return text


def enrich_from_bank(candidates: list, bank: dict) -> list:
    """
    Для каждого кандидата ВСЕГДА подставляет bank_path и bank_text из банка.
    Текст обрезается до первого упоминания подпунктов — они дублируют раскрытие.
    """
    for cand in candidates:
        bank_key = cand["id"].replace("FB-", "")
        bank_key = re.sub(r'[а-ед]\)$', '', bank_key)
        bank_entry = bank.get(bank_key, {})

        if bank_entry.get("path"):
            cand["bank_path"] = bank_entry["path"]
        if bank_entry.get("text"):
            cand["bank_text"] = _trim_bank_text(bank_entry["text"])

    return candidates


def enrich_expansion_from_bank(expansion: dict, bank: dict) -> dict:
    """Обогащает пункты раскрытия данными из банка. ВСЕГДА заменяет на короткий текст."""
    for cand_id, items in expansion.items():
        for item in items:
            bank_key = item["id"].replace("FB-", "")
            bank_key = re.sub(r'[а-ед]\)$', '', bank_key)
            bank_entry = bank.get(bank_key, {})
            if bank_entry.get("path"):
                item["bank_path"] = bank_entry["path"]
            if bank_entry.get("text"):
                item["bank_text"] = _trim_bank_text(bank_entry["text"])
    return expansion


# ================================================================
# 6. ОБРАБОТКА ОДНОГО ПОДПРОЦЕССА
# ================================================================

def process_subprocess(sp_id: str, sp_dir: str, bank: dict) -> dict | None:
    """
    Обрабатывает bp_06_XX/.
    Шаги:           consolidated_l3.json + expansion_consolidated.json
    Рекомендации:   consolidated_recs.json + expansion_recs_consolidated.json
    Тексты чанков:  chunks/chunks_3types.json
    """

    # ── Пути ──
    chunks_path         = os.path.join(sp_dir, "chunks", "chunks_3types.json")
    consolidated_path   = os.path.join(sp_dir, "mapping", "consolidated_l3.json")
    consol_recs_path    = os.path.join(sp_dir, "mapping", "consolidated_recs.json")
    expansion_path      = os.path.join(sp_dir, "expansion", "expansion_consolidated.json")
    expansion_recs_path = os.path.join(sp_dir, "expansion", "expansion_recs_consolidated.json")

    # ── Загрузка ──
    chunks_raw = load_json(chunks_path)
    if chunks_raw is None:
        print(f"  {sp_id}: chunks_3types.json не найден — пропускаю")
        return None

    consolidated    = parse_consolidated(load_json(consolidated_path))
    consol_recs     = parse_consolidated(load_json(consol_recs_path))
    expansion_steps = parse_expansion(load_json(expansion_path))
    expansion_recs  = parse_expansion(load_json(expansion_recs_path))

    # ── Имя подпроцесса ──
    sp_name = ""
    for entry in consolidated.values():
        if entry.get("bp_name"):
            sp_name = entry["bp_name"]
            break
    if not sp_name:
        for entry in consol_recs.values():
            if entry.get("bp_name"):
                sp_name = entry["bp_name"]
                break
    if not sp_name and isinstance(chunks_raw, list) and chunks_raw:
        sp_name = chunks_raw[0].get("metadata", {}).get("bp_name", "")

    # ── Индекс чанков для asis_text ──
    step_chunks = {}
    rec_chunks = {}

    for chunk in (chunks_raw if isinstance(chunks_raw, list) else []):
        meta = chunk.get("metadata", {})
        ctype = meta.get("chunk_type", "")
        cid = chunk.get("chunk_id", chunk.get("id", ""))
        pid = meta.get("pair_id", "")

        if ctype.startswith("step"):
            key = pid if pid and pid != "?" else cid
            if key not in step_chunks:
                step_chunks[key] = chunk
            else:
                priority = {"step_enriched": 3, "step_narrative": 2, "step_clean": 1}
                if priority.get(ctype, 0) > priority.get(
                        step_chunks[key].get("metadata", {}).get("chunk_type", ""), 0):
                    step_chunks[key] = chunk
        elif ctype.startswith("rec"):
            rec_chunks[cid] = chunk

    # ── ШАГИ ──
    steps = []
    for pair_id, entry in sorted(consolidated.items()):
        source = step_chunks.get(pair_id)

        asis_text = entry.get("operation", "")
        if not asis_text and source:
            asis_text = source.get("search_text",
                        source.get("chunk_text_plain",
                        source.get("metadata", {}).get("operation", "")))

        title = (entry.get("operation", "") or pair_id)[:100]

        candidates = enrich_from_bank(entry["candidates"], bank)

        pair_expansion = expansion_steps.get(pair_id, {})
        pair_expansion = enrich_expansion_from_bank(pair_expansion, bank)

        steps.append({
            "chunk_id": pair_id,
            "title": title,
            "asis_text": asis_text,
            "type": "step",
            "candidates": candidates,
            "expansion": pair_expansion,
        })

    # ── РЕКОМЕНДАЦИИ ──
    recommendations = []
    for pair_id, entry in sorted(consol_recs.items()):
        source = rec_chunks.get(pair_id)

        asis_text = entry.get("operation", "")
        if not asis_text and source:
            asis_text = source.get("search_text",
                        source.get("chunk_text_plain",
                        source.get("metadata", {}).get("text", "")))

        title = (entry.get("operation", "") or pair_id)[:120]
        if not title.strip() or title == pair_id:
            if source:
                title = source.get("search_text",
                        source.get("chunk_text_plain",
                        source.get("metadata", {}).get("text", pair_id)))[:120]

        chunk_type = entry.get("chunk_type", "")
        if not chunk_type and source:
            chunk_type = source.get("metadata", {}).get("chunk_type", "rec_process")
        if not chunk_type:
            chunk_type = "rec_process"

        candidates = enrich_from_bank(entry["candidates"], bank)

        pair_expansion = expansion_recs.get(pair_id, {})
        pair_expansion = enrich_expansion_from_bank(pair_expansion, bank)

        recommendations.append({
            "chunk_id": pair_id,
            "title": title,
            "asis_text": asis_text,
            "type": chunk_type,
            "candidates": candidates,
            "expansion": pair_expansion,
        })

    # ── Лог ──
    exp_s = sum(len(v) for v in expansion_steps.values())
    exp_r = sum(len(v) for v in expansion_recs.values())
    print(f"  {sp_id}: {len(steps)} шагов, {len(recommendations)} рек.  | exp: steps={exp_s} recs={exp_r}")

    return {
        "id": sp_id,
        "name": sp_name,
        "steps": steps,
        "recommendations": recommendations,
    }


# ================================================================
# 7. MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Сборка данных ProcessScout AI для UI")
    parser.add_argument("--data_dir", required=True, help="Папка с bp_06_01/ и т.д.")
    parser.add_argument("--bank", required=True, help="Путь к bank_4-5.xlsx")
    parser.add_argument("--output", default="processscout_data.json", help="Выходной JSON")
    parser.add_argument("--document_title", default="БП-06 Производство", help="Название документа")
    args = parser.parse_args()

    print("=" * 60)
    print("ProcessScout AI — prepare_data.py v3 (fixed)")
    print("=" * 60)

    print(f"\n1. Загрузка банка: {args.bank}")
    bank = load_bank(args.bank)

    print(f"\n2. Сканирование {args.data_dir}")
    bp_folders = find_bp_folders(args.data_dir)
    print(f"  Найдено {len(bp_folders)} подпроцессов: {[f[0] for f in bp_folders]}")

    print(f"\n3. Обработка")
    subprocesses = []
    for sp_id, sp_dir in bp_folders:
        result = process_subprocess(sp_id, sp_dir, bank)
        if result:
            subprocesses.append(result)
        else:
            subprocesses.append({
                "id": sp_id,
                "name": f"Подпроцесс {sp_id}",
                "steps": [],
                "recommendations": [],
            })

    output = {
        "meta": {
            "document_id": "BP-06",
            "document_title": args.document_title,
            "generated_at": datetime.now().isoformat(),
            "version": "3.0",
            "total_subprocesses": len(subprocesses),
            "total_steps": sum(len(sp["steps"]) for sp in subprocesses),
            "total_recommendations": sum(len(sp["recommendations"]) for sp in subprocesses),
        },
        "subprocesses": subprocesses,
    }

    print(f"\n4. Запись: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Размер:        {size_kb:.1f} KB")
    print(f"  Подпроцессов:  {len(subprocesses)}")
    print(f"  Шагов:         {output['meta']['total_steps']}")
    print(f"  Рекомендаций:  {output['meta']['total_recommendations']}")
    print("\n" + "=" * 60)
    print("✓ Готово! Загрузите файл в processscout_ui.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
