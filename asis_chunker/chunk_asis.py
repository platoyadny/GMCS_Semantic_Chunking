"""
chunk_asis.py — Универсальный чанкинг документов обследования AS-IS
               для маппинга на банк функциональности PLM/ERP.

Вход:  .docx файл с заголовками H2 (начало каждого БП) и H3 (секции внутри БП).
Выход: JSON файл с чанками, каждый содержит search_text (для BM25 + embedding)
       и full_context (для LLM и формирования ФТТ).

Стандарт разметки документа (4 обязательных H3 + 1 опциональный):
  [H1] БП XX. Название документа
  [H2] БП XX.YY Название процесса
    [H3] Карточка процесса
    [H3] Описание хода процесса
    [H3] Описание шагов процесса      ← перед таблицей шагов
    [H3] Особенности реализации AS-IS
    [H3] Рекомендации                  ← опциональный

Стратегия чанкинга:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Таблица шагов → 3 чанка на строку:                                     │
  │   A (step_table)    — название операции + контекст БП (для BM25)        │
  │   B (step_narrative)— абзац нарратива (если ссылка XX.YY.ZZ найдена)   │
  │   C (step_enriched) — операция + исполнитель + вход/выход + система     │
  │                       (программно или через LLM с --use-llm)            │
  │                                                                          │
  │ Нарратив      → абзацы С ссылкой XX.YY.ZZ → чанк B (привязан к шагу)  │
  │                 абзацы БЕЗ ссылки          → самостоятельные чанки      │
  │                                                                          │
  │ Рекомендации  → по пунктам (одна мысль = всё между цифрами нумерации)  │
  │                 подтипы: rec_automation / rec_process / rec_consulting   │
  │                 search_text = только функциональная часть (после         │
  │                 "Рекомендуется:"), без описания проблемы                 │
  │                                                                          │
  │ Карточка      → целиком (контекст)                                      │
  │ AS-IS         → целиком (факты)                                         │
  │ Intro         → целиком                                                  │
  └──────────────────────────────────────────────────────────────────────────┘

Использование:
  python chunk_asis.py --input БП_8.docx
  python chunk_asis.py --input БП_8.docx --output output/chunks_bp08.json
  python chunk_asis.py --input БП_8.docx --use-llm    # LLM-переформулировка step_enriched
"""

import json
import re
import argparse
import os
from pathlib import Path

# Namespace для работы с XML элементами Word
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

# Маркер для элементов нумерованного списка Word (нумерация в стиле, не в тексте)
LISTITEM_MARKER = "#LISTITEM#"

def strip_marker(text: str) -> str:
    """Убирает маркер #LISTITEM# если есть."""
    return text.replace(LISTITEM_MARKER, "", 1).strip() if LISTITEM_MARKER in text else text.strip()


# Регулярка для поиска ссылок на шаги вида XX.YY.ZZ (6+ цифр с точками)
# Примеры: "08.04.08", "06.01.03", "БП 06.01.03"
STEP_REF_PATTERN = re.compile(r'\b(\d{2}\.\d{2}\.\d{2,})\b')


# ============================================================
# 1. ПАРСИНГ ДОКУМЕНТА
# ============================================================

def get_style_name(element) -> str:
    """
    Извлекает имя стиля из XML-элемента параграфа.
    Возвращает строку вида 'Heading2', 'Heading3' или '' для обычного текста.
    """
    pPr = element.find(f'{{{W}}}pPr')
    if pPr is not None:
        pStyle = pPr.find(f'{{{W}}}pStyle')
        if pStyle is not None:
            return pStyle.get(f'{{{W}}}val', '')
    return ''


def get_text(element) -> str:
    """Извлекает весь текст из XML-элемента (параграф или ячейка)."""
    return ''.join(
        node.text or ''
        for node in element.iter(f'{{{W}}}t')
    ).strip()


def extract_table_rows_xml(table_element) -> list[dict]:
    """
    Извлекает строки таблицы из XML-элемента <w:tbl>.
    Первая строка — заголовок, остальные — данные.
    Не зависит от python-docx, работает напрямую с XML.
    """
    rows = table_element.findall(f'.//{{{W}}}tr')
    if not rows:
        return []

    # Извлекаем заголовки из первой строки
    header_cells = rows[0].findall(f'{{{W}}}tc')
    headers = [get_text(cell) for cell in header_cells]

    # Извлекаем данные из остальных строк
    data_rows = []
    for row in rows[1:]:
        cells = row.findall(f'{{{W}}}tc')
        cell_texts = [get_text(cell) for cell in cells]
        if not any(ct.strip() for ct in cell_texts):
            continue
        row_dict = {}
        for i, header in enumerate(headers):
            row_dict[header] = cell_texts[i].strip() if i < len(cell_texts) else ""
        data_rows.append(row_dict)

    return data_rows


def is_steps_table(table_rows: list[dict]) -> bool:
    """
    Определяет, является ли таблица таблицей шагов процесса.
    Эвристика: в заголовках есть «Операция» и «Исполнитель»/«Система».
    """
    if not table_rows:
        return False
    headers_lower = {k.lower() for k in table_rows[0].keys()}
    # Должны быть минимум колонки "Операция" и хотя бы одна из характерных
    has_operation = any('операци' in h for h in headers_lower)
    has_other = any(kw in h for h in headers_lower for kw in ('исполнит', 'систем', 'входящ'))
    return has_operation and has_other


def parse_document(filepath: str) -> list[dict]:
    """
    Парсит .docx файл напрямую через XML (обходит баги python-docx с SVG).
    Возвращает список БП, каждый с секциями по H3-заголовкам.
    
    Каждая секция содержит:
      - paragraphs: список текстов абзацев
      - tables: список таблиц (каждая — список словарей строк)
    """
    import zipfile
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(filepath) as z:
        tree = ET.parse(z.open('word/document.xml'))
    body = tree.getroot().find(f'.//{{{W}}}body')

    business_processes = []
    current_bp = None
    current_section_name = None
    current_section = None

    for child in body:
        tag = child.tag.split('}')[-1]

        if tag == 'p':
            text = get_text(child)
            if not text:
                continue

            style = get_style_name(child)

            # Проверяем: параграф является элементом нумерованного списка Word?
            # Word хранит нумерацию в <w:numPr>, а не в тексте.
            # Добавляем маркер "#LISTITEM#" чтобы чанкер мог это учесть.
            pPr = child.find(f'{{{W}}}pPr')
            has_numPr = False
            if pPr is not None and pPr.find(f'{{{W}}}numPr') is not None:
                has_numPr = True

            # H2 = начало нового бизнес-процесса
            if 'Heading2' in style:
                bp_match = re.match(r'БП\s+(\d+\.\d+)\s+(.*)', text)
                if bp_match:
                    current_bp = {
                        "bp_id": bp_match.group(1),
                        "bp_name": bp_match.group(2).strip(),
                        "intro_text": "",
                        "sections": {}
                    }
                    business_processes.append(current_bp)
                    current_section_name = None
                    current_section = None
                continue

            # H3 = начало новой секции внутри БП
            if 'Heading3' in style and current_bp is not None:
                current_section_name = text.strip().rstrip(':')
                current_section = {"paragraphs": [], "tables": []}
                current_bp["sections"][current_section_name] = current_section
                continue

            # Обычный параграф
            if current_bp is not None:
                # Помечаем элементы нумерованного списка Word маркером
                stored_text = f"{LISTITEM_MARKER} {text}" if has_numPr else text
                if current_section is not None:
                    current_section["paragraphs"].append(stored_text)
                else:
                    current_bp["intro_text"] += text + "\n"

        elif tag == 'tbl' and current_bp is not None:
            table_rows = extract_table_rows_xml(child)
            if table_rows:
                if current_section is not None:
                    current_section["tables"].append(table_rows)
                # Таблица до первого H3 — сохраняем в intro (для контекста)

    return business_processes


# ============================================================
# 2. КЛАССИФИКАЦИЯ СЕКЦИЙ ПО КОНТЕНТУ
# ============================================================

def classify_section(section_name: str, section_data: dict) -> str:
    """
    Определяет тип секции.
    Приоритет 1: по названию H3-заголовка.
    Приоритет 2 (fallback): по содержимому, если заголовок не совпал с известными.
    """
    name_lower = section_name.lower()

    # --- Приоритет 1: по названию H3 ---
    if any(kw in name_lower for kw in ("шагов", "шаги процесса")):
        return "step_table"
    if "рекоменда" in name_lower:
        return "recommendations"
    if "карточка" in name_lower:
        return "card"
    if any(kw in name_lower for kw in ("хода процесса", "ход процесса", "обобщенное описание")):
        return "narrative"
    if any(kw in name_lower for kw in ("особенности", "as-is")):
        return "asis_notes"

    # --- Приоритет 2: по содержимому (fallback для нестандартных заголовков) ---
    paragraphs = section_data.get("paragraphs", [])
    full_text = " ".join(paragraphs[:15]).lower()
    tables = section_data.get("tables", [])

    # Есть таблица с колонками «Операция» + «Исполнитель»/«Система» → шаги
    for tbl in tables:
        if is_steps_table(tbl):
            return "step_table"

    # Есть «владелец процесса» + «участники» → карточка
    if "владелец процесса" in full_text and "участники" in full_text:
        return "card"

    # Есть «рекомендуется» или «автоматизац» → рекомендации
    if full_text.count("рекомендуется") >= 2 or "автоматизац" in full_text:
        return "recommendations"

    # Есть «as-is» или «расхождени» → особенности
    if "as-is" in full_text or "расхождени" in full_text:
        return "asis_notes"

    # Дефолт: нарратив (ничего не теряется)
    return "narrative"


# ============================================================
# 3. ПРИВЯЗКА НАРРАТИВНЫХ АБЗАЦЕВ К ШАГАМ ТАБЛИЦЫ
# ============================================================

def build_narrative_step_map(paragraphs: list[str], bp_id: str) -> dict:
    """
    Строит маппинг step_id → [список абзацев нарратива].
    
    Правило привязки СТРОГОЕ: только по явной числовой ссылке XX.YY.ZZ.
    
    Логика:
    - Если абзац содержит XX.YY.ZZ, он НАЧИНАЕТ блок для этого шага.
    - Последующие абзацы без ссылки — ПРОДОЛЖЕНИЕ текущего блока.
    - Абзац с новой ссылкой — начало нового блока.
    - Абзацы до первой ссылки и после «разрыва» → не привязаны (None).
    
    Возвращает:
      {
        "06.01.03": ["Абзац с описанием шага", "Продолжение описания"],
        None: ["Абзац без привязки к шагу", ...]
      }
    """
    step_map = {}     # step_id → [paragraphs]
    current_step = None
    bp_prefix = bp_id + "."  # "06.01." — для фильтрации ссылок на СВОИ шаги

    for para in paragraphs:
        clean_para = strip_marker(para)
        refs = STEP_REF_PATTERN.findall(clean_para)
        # Фильтруем: оставляем только ссылки на шаги ТЕКУЩЕГО БП
        own_refs = [r for r in refs if r.startswith(bp_prefix)]

        if own_refs:
            current_step = own_refs[0]
            if current_step not in step_map:
                step_map[current_step] = []
            step_map[current_step].append(clean_para)
        elif current_step is not None:
            step_map[current_step].append(clean_para)
        else:
            if None not in step_map:
                step_map[None] = []
            step_map[None].append(clean_para)

    return step_map


# ============================================================
# 4. ФОРМИРОВАНИЕ ЧАНКОВ
# ============================================================

def chunk_step_table(bp_id: str, bp_name: str, section_data: dict,
                     narrative_step_map: dict) -> list[dict]:
    """
    Чанкинг таблицы шагов. Для каждой строки создаём до 3 чанков:
    
    Чанк A (step_table):    search_text = название операции + контекст БП
                            → точный, для BM25 по ключевым словам
    Чанк B (step_narrative): search_text = абзац нарратива (если привязка есть)
                            → семантический, для случаев несовпадения лексики
    Чанк C (step_enriched): search_text = операция + исполнитель + вход + выход + система
                            → контекстный, ловит маппинги по ролям и документам
    
    Все чанки ссылаются на один pair_id и full_context.
    """
    chunks = []

    for tbl in section_data.get("tables", []):
        if not is_steps_table(tbl):
            continue

        for row in tbl:
            step_num = row.get("№", "").strip().rstrip('.')
            operation = row.get("Операция", "").strip()
            executor = row.get("Исполнитель", "").strip()
            input_info = row.get("Входящая информация", "").strip()
            output_info = row.get("Исходящая информация", "").strip()
            # Колонка "Система" или "Системы" — ищем обе
            system = row.get("Система", row.get("Системы", "")).strip()

            if not operation:
                continue

            # Извлекаем step_id из текста операции
            step_id_match = re.search(r'(\d{2}\.\d{2}\.\d{2,})', operation)
            if step_id_match:
                step_id = step_id_match.group(1)
                # Чистое название операции: убираем "БП XX.YY.ZZ" и нумерацию
                operation_clean = re.sub(
                    r'(БП\s+)?\d{2}\.\d{2}\.\d{2,}\s*', '', operation
                ).strip().lstrip('«').rstrip('»').strip()
            else:
                step_id = f"{bp_id}.{step_num.zfill(2)}" if step_num else ""
                operation_clean = operation

            pair_id = f"BP-{step_id}"

            # full_context: ВСЕ данные строки таблицы + нарратив если есть
            context_lines = [
                f"Шаг {step_num}: {operation}",
                f"Исполнитель: {executor}" if executor else None,
                f"Входящая информация: {input_info}" if input_info else None,
                f"Исходящая информация: {output_info}" if output_info else None,
                f"Система: {system}" if system else None,
            ]

            # Добавляем нарративное описание если найдено
            narr_paras = narrative_step_map.get(step_id, [])
            if narr_paras:
                context_lines.append("---")
                context_lines.extend(narr_paras)

            full_context = "\n".join(line for line in context_lines if line)

            base_metadata = {
                "bp_id": bp_id,
                "bp_name": bp_name,
                "step_id": step_id,
                "step_num": step_num,
                "operation": operation_clean,
                "executor": executor,
                "system": system,
                "input_docs": input_info,
                "output_docs": output_info,
                "pair_id": pair_id,
            }

            # --- Чанк A: табличный (всегда) ---
            # Чистое название операции + контекст БП. Точный, для BM25.
            search_a = f"В рамках процесса «{bp_name}»: {operation_clean}"

            chunks.append({
                "chunk_id": f"{pair_id}-table",
                "search_text": search_a,
                "full_context": full_context,
                "metadata": {
                    **base_metadata,
                    "chunk_type": "step_table",
                    "mapping_priority": "high",
                }
            })

            # --- Чанк B: нарративный (только если привязка найдена) ---
            if narr_paras:
                narr_text = " ".join(narr_paras)
                # Ограничиваем ~250 символов для embedding
                if len(narr_text) > 250:
                    sentences = re.split(r'(?<=[.!?])\s+', narr_text)
                    search_b = ""
                    for s in sentences:
                        if len(search_b) + len(s) < 250:
                            search_b += s + " "
                        else:
                            break
                    search_b = search_b.strip()
                else:
                    search_b = narr_text

                chunks.append({
                    "chunk_id": f"{pair_id}-narr",
                    "search_text": search_b,
                    "full_context": full_context,
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "step_narrative",
                        "mapping_priority": "high",
                    }
                })

            # --- Чанк C: обогащённый (всегда) ---
            # Операция + данные из таблицы: кто делает, что на входе/выходе, в какой системе.
            # Ловит маппинги, где в банке упоминается роль или документ.
            enriched_parts = [operation_clean]
            if executor:
                enriched_parts.append(f"Исполнитель: {executor}")
            if input_info:
                enriched_parts.append(f"Вход: {input_info}")
            if output_info:
                enriched_parts.append(f"Выход: {output_info}")
            if system:
                enriched_parts.append(f"Система: {system}")
            search_c = ". ".join(enriched_parts)

            chunks.append({
                "chunk_id": f"{pair_id}-enriched",
                "search_text": search_c,
                "full_context": full_context,
                "metadata": {
                    **base_metadata,
                    "chunk_type": "step_enriched",
                    "mapping_priority": "high",
                }
            })

    return chunks


def chunk_narrative_independent(bp_id: str, bp_name: str,
                                paragraphs: list[str],
                                narrative_step_map: dict) -> list[dict]:
    """
    Чанкинг нарративных абзацев, НЕ привязанных к шагам таблицы.
    Каждый абзац — отдельный чанк.
    """
    chunks = []
    # Абзацы без привязки
    unlinked = narrative_step_map.get(None, [])
    # Если narrative_step_map пуст (нет ни одной ссылки XX.YY.ZZ) — все абзацы
    if not narrative_step_map:
        unlinked = [strip_marker(p) for p in paragraphs if len(strip_marker(p)) >= 30]

    para_num = 0
    for para_text in unlinked:
        stripped = strip_marker(para_text)
        if len(stripped) < 30:
            continue
        para_num += 1
        chunks.append({
            "chunk_id": f"BP-{bp_id}-narr-{para_num:02d}",
            "search_text": stripped,
            "full_context": stripped,
            "metadata": {
                "bp_id": bp_id,
                "bp_name": bp_name,
                "chunk_type": "narrative",
                "para_num": para_num,
                "mapping_priority": "low",
            }
        })

    return chunks


def classify_recommendation_subtype(text: str) -> str:
    """
    Определяет подтип рекомендации по тексту (название секции или заголовок внутри).
    Возвращает: rec_automation, rec_process, rec_consulting, rec_general.
    """
    text_lower = text.lower()
    if "автоматизац" in text_lower:
        return "rec_automation"
    if "консалтинг" in text_lower:
        return "rec_consulting"
    if any(kw in text_lower for kw in ("ведени", "процесс", "регламент", "kpi")):
        return "rec_process"
    return "rec_general"


def chunk_recommendations(bp_id: str, bp_name: str, section_name: str,
                          section_data: dict) -> list[dict]:
    """
    Чанкинг рекомендаций.
    
    Принцип: одна мысль = всё между двумя цифрами нумерации.
    
    Структура документа (пример БП_06):
      "1. Рекомендации по ведению процесса"    ← заголовок подраздела → переключение подтипа
      "  1. Устранение расхождений"             ← начало рекомендации (цифра)
      "  - Провести формальное уточнение..."    ← продолжение (буллет)
      "  2. Детализация подпроцессов"            ← начало следующей рекомендации (цифра)
      "  - Разработать подробные..."             ← продолжение (буллет)
      "2. Рекомендации по автоматизации"        ← заголовок подраздела → переключение подтипа
      "1. Интеграция систем..."                  ← начало рекомендации (нумерация сбросилась)
    """
    chunks = []
    paragraphs = section_data.get("paragraphs", [])

    # Определяем дефолтный подтип из названия H3-секции
    default_subtype = classify_recommendation_subtype(section_name)
    current_subtype = default_subtype

    grouped_recs = []     # [(subtype, full_text)]
    current_rec = ""      # аккумулятор текущей рекомендации
    last_started_by_number = False  # текущий пункт начат с цифры? ("1. Проблема" → True)

    for para_text in paragraphs:
        stripped = para_text.strip()
        if not stripped:
            continue

        # Определяем: это элемент нумерованного списка Word (нумерация в стиле, не в тексте)?
        is_word_list_item = stripped.startswith("#LISTITEM#")
        if is_word_list_item:
            stripped = stripped.replace("#LISTITEM#", "", 1).strip()

        # --- Проверка 1: это заголовок подраздела рекомендаций? ---
        # Паттерн: "1. Рекомендации по ведению", "2. Рекомендации по автоматизации"
        # или просто "Рекомендации по автоматизации процесса"
        # Маркер #LISTITEM# уже снят — проверяем чистый текст
        if re.match(r'^(#LISTITEM#\s*)?\d*\.?\s*Рекомендации\s', para_text.strip(), re.IGNORECASE):
            if current_rec:
                grouped_recs.append((current_subtype, current_rec))
                current_rec = ""
                last_started_by_number = False
            current_subtype = classify_recommendation_subtype(stripped)
            continue

        # --- Проверка 2: это начало нового пункта? ---
        # Способ A: явная нумерация в тексте
        # Поддерживаемые форматы: "1. Текст", "2) Текст", "1.1 Текст", "2.3 Текст", "1.1.1 Текст"
        # Regex: одна или несколько групп цифр через точку, затем опц. точка/скобка, затем пробел
        num_match = re.match(r'^(\d+(?:\.\d+)*)[\.\)]?\s+(.*)', stripped)
        
        # Способ A2: Word-нумерация без цифры в тексте (numPr в XML)
        # Но НЕ КАЖДЫЙ элемент списка — новый пункт. Подпункты-буллеты тоже имеют numPr.
        # Эвристика: новый пункт = заглавная буква + длина > 60 символов.
        # Подпункт = строчная буква или короткий текст (обычно "график плановых ТО;")
        is_word_top_level_item = (
            is_word_list_item and
            len(stripped) > 60 and
            stripped[0].isupper()
        )
        is_numbered = bool(num_match) or is_word_top_level_item
        
        # Способ B: Word-список без видимой нумерации — определяем по содержимому.
        # НО: "Рекомендуется:" после нумерованного пункта — это ПРОДОЛЖЕНИЕ, не новый пункт.
        # Паттерн документа: "N. Описание проблемы" → "Рекомендуется: решение" = одна мысль.
        content_new_item = (
            not is_numbered and
            not last_started_by_number and  # ← не разрываем пару "проблема + решение"
            len(stripped) > 60 and
            (
                stripped.startswith("Рекомендуется") or
                stripped.startswith("Внедрить") or
                stripped.startswith("Настроить") or
                stripped.startswith("Разработать") or
                stripped.startswith("Перевести") or
                stripped.startswith("Автоматизация") or
                stripped.startswith("Интеграция") or
                stripped.startswith("Оцифровка") or
                stripped.startswith("Провести") or
                stripped.startswith("Обеспечить") or
                stripped.startswith("Ввести единый") or
                stripped.startswith("Создать единый") or
                stripped.startswith("Реализовать")
            )
        )

        if is_numbered or content_new_item:
            # Сохраняем предыдущую рекомендацию
            if current_rec:
                grouped_recs.append((current_subtype, current_rec))
            # Начинаем новую:
            current_rec = num_match.group(2) if num_match else stripped
            last_started_by_number = is_numbered
            continue

        # --- Всё остальное: продолжение текущего пункта ---
        if current_rec:
            clean = re.sub(r'^[-–—•]\s*', '', stripped)
            separator = "; " if current_rec and not current_rec.endswith(":") else " "
            current_rec += separator + clean
        else:
            current_rec = stripped

    # Не забываем последний пункт
    if current_rec:
        grouped_recs.append((current_subtype, current_rec))

    # --- Формируем чанки ---
    subtype_counters = {}
    for subtype, rec_text in grouped_recs:
        subtype_counters[subtype] = subtype_counters.get(subtype, 0) + 1
        rec_num = subtype_counters[subtype]
        priority = "high" if subtype == "rec_automation" else "medium"

        type_suffix = {"rec_automation": "auto", "rec_process": "proc",
                       "rec_consulting": "cons", "rec_general": "gen"}
        suffix = type_suffix.get(subtype, "gen")

        # search_text: только функциональная часть (решение), без описания проблемы.
        # Банк содержит возможности системы — вектор search_text должен быть
        # максимально близок к функциональным формулировкам банка.
        #
        # Ищем маркер "Рекомендуется:" — всё после него = чистая функция.
        # Если маркера нет — текст уже является функциональной рекомендацией целиком.
        rec_marker = re.search(r'Рекомендуется[:\s]+', rec_text)
        if rec_marker:
            # Берём только текст после "Рекомендуется:"
            functional_part = rec_text[rec_marker.end():].strip()
        else:
            functional_part = rec_text

        # Обрезаем до первого предложения для компактного вектора
        first_sentence = functional_part.split('.')[0] + '.' if '.' in functional_part else functional_part
        if len(first_sentence) > 200:
            first_sentence = first_sentence[:200] + "..."

        search_text = f"Рекомендация по процессу «{bp_name}»: {first_sentence}"

        chunks.append({
            "chunk_id": f"BP-{bp_id}-rec-{suffix}-{rec_num:02d}",
            "search_text": search_text,
            "full_context": rec_text,  # полный текст: проблема + решение
            "metadata": {
                "bp_id": bp_id,
                "bp_name": bp_name,
                "chunk_type": subtype,
                "rec_num": rec_num,
                "mapping_priority": priority,
            }
        })

    return chunks


def chunk_whole_section(bp_id: str, bp_name: str, section_name: str,
                        section_data: dict, chunk_type: str) -> list[dict]:
    """Чанкинг секции целиком: один чанк на всю секцию."""
    paragraphs = section_data.get("paragraphs", [])
    full_text = "\n".join(strip_marker(p) for p in paragraphs if strip_marker(p))
    if not full_text:
        return []

    search_text = f"{section_name} процесса «{bp_name}»: {full_text[:200]}"
    priority = "medium" if chunk_type == "card" else "low"

    return [{
        "chunk_id": f"BP-{bp_id}-{chunk_type}",
        "search_text": search_text,
        "full_context": full_text,
        "metadata": {
            "bp_id": bp_id,
            "bp_name": bp_name,
            "chunk_type": chunk_type,
            "section_name": section_name,
            "mapping_priority": priority,
        }
    }]


# ============================================================
# 5. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def chunk_document(filepath: str) -> list[dict]:
    """
    Главная функция: парсит документ → формирует все чанки.
    """
    business_processes = parse_document(filepath)
    all_chunks = []

    for bp in business_processes:
        bp_id = bp["bp_id"]
        bp_name = bp["bp_name"]

        # Intro (текст до первого H3)
        if bp["intro_text"].strip():
            all_chunks.append({
                "chunk_id": f"BP-{bp_id}-intro",
                "search_text": f"Описание процесса «{bp_name}»: {bp['intro_text'].strip()[:200]}",
                "full_context": bp["intro_text"].strip(),
                "metadata": {
                    "bp_id": bp_id, "bp_name": bp_name,
                    "chunk_type": "intro", "mapping_priority": "low",
                }
            })

        # Сначала строим карту привязки нарратива к шагам
        # Ищем секцию нарратива
        narrative_paragraphs = []
        for sec_name, sec_data in bp["sections"].items():
            sec_type = classify_section(sec_name, sec_data)
            if sec_type == "narrative":
                narrative_paragraphs = sec_data.get("paragraphs", [])
                break

        narrative_step_map = build_narrative_step_map(narrative_paragraphs, bp_id)

        # Подсчёт привязанных шагов (для отладки)
        linked_steps = [k for k in narrative_step_map if k is not None]

        # Обработка каждой H3-секции
        for sec_name, sec_data in bp["sections"].items():
            sec_type = classify_section(sec_name, sec_data)

            if sec_type == "step_table":
                chunks = chunk_step_table(bp_id, bp_name, sec_data, narrative_step_map)
                all_chunks.extend(chunks)

            elif sec_type == "narrative":
                # Самостоятельные нарративные чанки (непривязанные абзацы)
                chunks = chunk_narrative_independent(
                    bp_id, bp_name, narrative_paragraphs, narrative_step_map
                )
                all_chunks.extend(chunks)

            elif sec_type == "recommendations":
                chunks = chunk_recommendations(bp_id, bp_name, sec_name, sec_data)
                all_chunks.extend(chunks)

            elif sec_type in ("card", "asis_notes"):
                chunks = chunk_whole_section(bp_id, bp_name, sec_name, sec_data, sec_type)
                all_chunks.extend(chunks)

            else:
                # Неизвестный тип → целиком
                chunks = chunk_whole_section(bp_id, bp_name, sec_name, sec_data, "unknown")
                all_chunks.extend(chunks)

    return all_chunks


# ============================================================
# 6. LLM-ПЕРЕФОРМУЛИРОВКА step_enriched
# ============================================================

# Системный промпт для переформулировки step_enriched чанков.
# Задача: превратить механическую склейку полей таблицы в одно связное предложение,
# которое embedding-модель обработает как естественный текст.
ENRICHED_SYSTEM_PROMPT = """Переформулируй данные строки таблицы бизнес-процесса в одно связное предложение (30-60 слов).

Правила:
- Сохрани ВСЕ ключевые сущности: кто делает, что делает, на основании чего, что выдаёт, в какой системе
- Раскрой аббревиатуры при первом упоминании, сохраняя сокращение в скобках: «шеф-монтажных работ (ШМР)»
- Убери номера форм и приложений (Ф11, Ф14, РП 225-36.011...) — это шум для поиска
- Не добавляй ничего от себя — только то что есть в данных
- Результат: ОДНО предложение, без списков, без «Вход:», «Выход:» и прочих меток"""


def enhance_enriched_with_llm(chunks: list[dict], llm_client) -> list[dict]:
    """
    LLM-переформулировка search_text для чанков типа step_enriched.

    Берёт механическую склейку полей:
      "Контроль за проведением ШМР. Исполнитель: Шеф-инженер.
       Вход: Распоряжение Ф11. Выход: Еженедельный отчет Ф14..."

    И превращает в связное предложение:
      "Шеф-инженер контролирует проведение шеф-монтажных работ (ШМР),
       на основании распоряжения о направлении специалиста формирует
       еженедельный отчёт, табель рабочего времени и ведёт журнал
       ШМР и ПНР в системе Directum."

    Программный вариант сохраняется в search_text_programmatic как fallback.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    enriched_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "step_enriched"]
    total = len(enriched_chunks)
    print(f"\n  LLM-переформулировка {total} step_enriched чанков...")

    success = 0
    errors = 0

    for i, chunk in enumerate(enriched_chunks):
        # Формируем промпт из полей таблицы (чистые данные, без склейки)
        meta = chunk["metadata"]
        user_lines = [
            f"Процесс: {meta['bp_name']}",
            f"Операция: {meta['operation']}",
        ]
        if meta.get("executor"):
            user_lines.append(f"Исполнитель: {meta['executor']}")
        if meta.get("input_docs"):
            user_lines.append(f"Входящая информация: {meta['input_docs']}")
        if meta.get("output_docs"):
            user_lines.append(f"Исходящая информация: {meta['output_docs']}")
        if meta.get("system"):
            user_lines.append(f"Система: {meta['system']}")

        user_text = "\n".join(user_lines)

        try:
            response = llm_client.invoke([
                SystemMessage(ENRICHED_SYSTEM_PROMPT),
                HumanMessage(user_text)
            ])
            # Сохраняем программный вариант как fallback
            chunk["search_text_programmatic"] = chunk["search_text"]
            # Заменяем search_text на LLM-переформулировку
            chunk["search_text"] = response.content.strip()
            success += 1

            # Прогресс каждые 10 чанков
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"    [{i+1}/{total}] обработано")

        except Exception as e:
            errors += 1
            print(f"    [{i+1}/{total}] ОШИБКА {meta['step_id']}: {e}")
            # При ошибке search_text остаётся программным — не ломаем результат

    print(f"  Готово: {success} успешно, {errors} ошибок")
    return chunks


# ============================================================
# 7. ТОЧКА ВХОДА
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Чанкинг документа AS-IS для маппинга на банк функциональности")
    parser.add_argument("--input", required=True, help="Путь к .docx файлу")
    parser.add_argument("--output", default=None, help="Путь к выходному .json")
    parser.add_argument("--use-llm", action="store_true",
                        help="Использовать LLM для переформулировки step_enriched чанков")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or f"output/chunks_{Path(input_path).stem}.json"

    print(f"\n{'='*60}")
    print(f"Чанкинг документа: {input_path}")
    print(f"{'='*60}\n")

    # 1. Чанкуем документ
    chunks = chunk_document(input_path)
    print(f"Создано чанков: {len(chunks)}")

    # 2. Опциональная LLM-переформулировка step_enriched
    if args.use_llm:
        try:
            from dotenv import load_dotenv
            from langchain_openai import ChatOpenAI
            load_dotenv()

            llm = ChatOpenAI(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.0,
                seed=47
            )
            chunks = enhance_enriched_with_llm(chunks, llm)
        except ImportError:
            print("  ОШИБКА: не установлены langchain_openai / dotenv")
            print("  pip install langchain-openai python-dotenv")
        except KeyError as e:
            print(f"  ОШИБКА: не задана переменная окружения {e}")
        except Exception as e:
            print(f"  ОШИБКА LLM: {e}")
            print("  search_text остаётся программным")

    # 3. Статистика
    type_counts = {}
    bp_stats = {}
    linked_steps = set()
    llm_enhanced = 0
    for c in chunks:
        ct = c["metadata"]["chunk_type"]
        bp = c["metadata"]["bp_id"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
        if bp not in bp_stats:
            bp_stats[bp] = {}
        bp_stats[bp][ct] = bp_stats[bp].get(ct, 0) + 1
        if ct == "step_narrative":
            linked_steps.add(c["metadata"]["step_id"])
        if "search_text_programmatic" in c:
            llm_enhanced += 1

    print(f"\nПо типам:")
    for ct, n in sorted(type_counts.items()):
        print(f"  {ct:25s}: {n}")

    print(f"\nПо бизнес-процессам:")
    for bp in sorted(bp_stats):
        total = sum(bp_stats[bp].values())
        details = ", ".join(f"{ct}={n}" for ct, n in sorted(bp_stats[bp].items()))
        print(f"  БП {bp}: {total:3d}  ({details})")

    if linked_steps:
        print(f"\nШаги с нарративной привязкой: {len(linked_steps)}")
    if llm_enhanced:
        print(f"Чанков переформулированных LLM: {llm_enhanced}")

    # 4. Добавляем source_doc
    for c in chunks:
        c["metadata"]["source_doc"] = Path(input_path).name

    # 5. Сохраняем
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nСохранено: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
