"""
llm_judge.py — LLM-as-a-Judge: валидация маппингов через Qwen 32B.

Берёт результаты hybrid_search.py (пары-кандидаты) и для каждой пары
отправляет полный контекст в Qwen для оценки функционального соответствия.

Qwen получает:
  - Полный контекст шага AS-IS (full_context: операция, исполнитель,
    вход/выход, система, нарратив)
  - Полный текст пункта банка (chunk_text_context + parent_chain)

Qwen выдаёт структурированную оценку:
  - match: full / partial / none
  - coverage_percent: 0-100
  - covered: что покрыто
  - not_covered: что не покрыто
  - rationale: обоснование для ФТТ
  - color: green / yellow / orange

Использование:
  # Только high-confidence (быстрый прогон)
  python scripts/llm_judge.py \
    --results output/mapping_results.json \
    --asis output/chunks_bp-06.json \
    --bank output/chunks_bank.json \
    --confidence high

  # Все mutual-confirmed пары
  python scripts/llm_judge.py \
    --results output/mapping_results.json \
    --asis output/chunks_bp-06.json \
    --bank output/chunks_bank.json \
    --mutual-only

  # Все пары (долго)
  python scripts/llm_judge.py \
    --results output/mapping_results.json \
    --asis output/chunks_bp-06.json \
    --bank output/chunks_bank.json
"""

import json
import re
import argparse
import time
import sys
from pathlib import Path

# Qwen API (OpenAI-совместимый, vLLM на 10.40.1.102)
QWEN_URL = "http://10.40.1.102:8000/v1/chat/completions"
QWEN_MODEL = "qdzzzxc/RuadaptQwen3-32B-Instruct-AWQ"

# Системный промпт для оценки маппинга
SYSTEM_PROMPT = """Ты эксперт по внедрению PLM/ERP систем на промышленных предприятиях.

Задача: оценить, насколько пункт банка функциональности PLM/ERP покрывает шаг бизнес-процесса AS-IS.

Правила оценки:
- "full" (green) — пункт банка напрямую покрывает описанную операцию: та же функция, те же сущности (документы, роли, системы)
- "partial" (yellow) — пункт банка покрывает ЧАСТЬ операции: совпадает основная функция, но часть деталей (конкретная система, документ, роль) не покрыта или отличается
- "none" (orange) — пункт банка НЕ покрывает операцию: общая тематическая область может совпадать, но функционально это разные вещи

ВАЖНО:
- Оценивай ФУНКЦИОНАЛЬНОЕ соответствие, а не лексическое. Совпадение слов (ТМЦ, ПКИ, упаковка) не означает совпадение функций.
- Учитывай ИЕРАРХИЮ банка (parent chain): пункт 5.7.4 имеет контекст "5 → 5.7 → 5.7.4", каждый уровень уточняет смысл.
- Если шаг AS-IS про "подготовку информации о стоимости", а банк про "стратегии распределения материалов" — это РАЗНЫЕ функции, даже если обе упоминают ТМЦ.

Ответь СТРОГО в формате JSON (без markdown, без ```):
{
  "match": "full" или "partial" или "none",
  "coverage_percent": число от 0 до 100,
  "covered": "что именно из шага AS-IS покрыто пунктом банка (или пусто если none)",
  "not_covered": "что из шага AS-IS НЕ покрыто (или пусто если full)",
  "rationale": "обоснование в 1-3 предложения для включения в ФТТ",
  "color": "green" или "yellow" или "orange"
}"""


def call_qwen(system: str, user: str, max_retries: int = 3) -> str:
    """
    Вызов Qwen через OpenAI-совместимый API.
    Обработка <think>...</think> тегов Qwen3 — вырезаем thinking,
    оставляем только ответ.
    """
    import requests

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": 1024,
        "temperature": 0.1    # почти детерминированный ответ
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(QWEN_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            raw_content = data["choices"][0]["message"]["content"]

            # Qwen3 использует thinking mode: <think>рассуждения</think>ответ
            # Вырезаем <think>...</think> блок, оставляем чистый ответ
            clean = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

            return clean

        except requests.exceptions.Timeout:
            print(f"      Timeout (попытка {attempt + 1}/{max_retries})")
            time.sleep(5)
        except Exception as e:
            print(f"      Ошибка (попытка {attempt + 1}/{max_retries}): {e}")
            time.sleep(3)

    return ""


def parse_json_response(text: str) -> dict:
    """
    Парсит JSON из ответа LLM.
    Обрабатывает случаи когда LLM оборачивает JSON в markdown-блок
    или добавляет текст до/после JSON.
    """
    # Убираем markdown-обёртку если есть
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Ищем JSON-объект в тексте
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Пробуем распарсить весь текст как JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}


def build_user_prompt(asis_context: str, bank_plain: str, bank_context: str,
                      bank_id: str, asis_meta: dict) -> str:
    """
    Формирует user-промпт с полным контекстом обеих сторон.
    """
    # AS-IS сторона
    bp_name = asis_meta.get('bp_name', '')
    operation = asis_meta.get('operation', '')
    executor = asis_meta.get('executor', '')
    system = asis_meta.get('system', '')
    step_id = asis_meta.get('step_id', '')

    prompt = f"""== ШАГ AS-IS ==
Процесс: {bp_name}
Шаг: {step_id} — {operation}
Исполнитель: {executor}
Система: {system}

Полный контекст шага:
{asis_context}

== ПУНКТ БАНКА ФУНКЦИОНАЛЬНОСТИ ==
ID: {bank_id}

Иерархия (plain):
{bank_plain}

Описание (context):
{bank_context}

== ЗАДАЧА ==
Оцени, покрывает ли пункт банка этот шаг AS-IS. Ответь в JSON."""

    return prompt


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge: валидация маппингов через Qwen")
    parser.add_argument("--results", required=True, help="JSON из hybrid_search.py")
    parser.add_argument("--asis", required=True, help="JSON с AS-IS чанками")
    parser.add_argument("--bank", required=True, help="JSON с чанками банка")
    parser.add_argument("--output", default="output/judge_results.json", help="Выходной JSON")
    parser.add_argument("--confidence", default=None,
                        help="Фильтр по confidence: high, medium, low")
    parser.add_argument("--mutual-only", action="store_true",
                        help="Только взаимно подтверждённые пары")
    parser.add_argument("--limit", type=int, default=None,
                        help="Максимум пар для обработки (для тестирования)")
    args = parser.parse_args()

    # Загружаем данные
    with open(args.results) as f:
        search_results = json.load(f)
    with open(args.asis) as f:
        asis_all = {c['chunk_id']: c for c in json.load(f)}
    with open(args.bank) as f:
        bank_all = {}
        for c in json.load(f):
            cid = c.get('metadata', {}).get('chunk_id', c.get('chunk_id', ''))
            bank_all[cid] = c

    mappings = search_results['mappings']

    # Фильтрация
    if args.confidence:
        mappings = [m for m in mappings if m['confidence'] == args.confidence]
    if args.mutual_only:
        mappings = [m for m in mappings if m['mutual_confirmation']]
    if args.limit:
        mappings = mappings[:args.limit]

    print(f"Пар для оценки: {len(mappings)}")
    print(f"Qwen: {QWEN_MODEL}")
    print()

    # Группируем AS-IS чанки по pair_id для полного контекста
    pair_to_chunks = {}
    for cid, chunk in asis_all.items():
        pid = chunk['metadata'].get('pair_id', cid)
        if pid not in pair_to_chunks:
            pair_to_chunks[pid] = []
        pair_to_chunks[pid].append(chunk)

    # Обработка каждой пары
    results = []
    total = len(mappings)
    stats = {"full": 0, "partial": 0, "none": 0, "error": 0}

    for i, mapping in enumerate(mappings):
        pair_id = mapping['pair_id']
        bank_id = mapping['bank_id']

        # Собираем полный контекст AS-IS из всех чанков с этим pair_id
        asis_chunks = pair_to_chunks.get(pair_id, [])
        if not asis_chunks:
            # Пробуем найти по chunk_id напрямую
            if pair_id in asis_all:
                asis_chunks = [asis_all[pair_id]]

        if not asis_chunks:
            stats["error"] += 1
            continue

        # Берём enriched-чанк как основной (он самый полный),
        # fallback на любой доступный
        enriched = None
        for ac in asis_chunks:
            if ac['metadata'].get('chunk_type') == 'step_enriched':
                enriched = ac
                break
        if not enriched:
            enriched = asis_chunks[0]

        asis_context = enriched.get('full_context', enriched.get('search_text', ''))
        asis_meta = enriched.get('metadata', {})

        # Банк
        bank_chunk = bank_all.get(bank_id)
        if not bank_chunk:
            stats["error"] += 1
            continue

        bank_plain = bank_chunk.get('chunk_text_plain', '')
        bank_context = bank_chunk.get('chunk_text_context', '')

        # Формируем промпт и вызываем Qwen
        user_prompt = build_user_prompt(
            asis_context, bank_plain, bank_context, bank_id, asis_meta
        )

        raw_response = call_qwen(SYSTEM_PROMPT, user_prompt)
        evaluation = parse_json_response(raw_response)

        if not evaluation:
            stats["error"] += 1
            print(f"  [{i+1}/{total}] {pair_id} <-> {bank_id}: PARSE ERROR")
            results.append({
                **mapping,
                "evaluation": {"error": "parse_failed", "raw": raw_response[:500]}
            })
            continue

        match_type = evaluation.get("match", "none")
        stats[match_type] = stats.get(match_type, 0) + 1

        results.append({
            **mapping,
            "evaluation": evaluation
        })

        # Прогресс
        color_emoji = {"full": "G", "partial": "Y", "none": "O"}
        indicator = color_emoji.get(match_type, "?")
        coverage = evaluation.get("coverage_percent", "?")
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] full={stats['full']} partial={stats['partial']} none={stats['none']} err={stats['error']}")
        else:
            print(f"  [{i+1}/{total}] [{indicator}] {pair_id} <-> {bank_id}: {match_type} ({coverage}%)")

    # Итоги
    print(f"\n{'='*60}")
    print(f"Результаты LLM-as-a-Judge")
    print(f"{'='*60}")
    print(f"  Всего оценено:  {total}")
    print(f"  full (green):   {stats['full']}")
    print(f"  partial (yellow): {stats['partial']}")
    print(f"  none (orange):  {stats['none']}")
    print(f"  Ошибки парсинга: {stats['error']}")
    if stats['full'] + stats['partial'] > 0:
        useful = stats['full'] + stats['partial']
        print(f"  Полезных маппингов: {useful} ({useful/total*100:.0f}%)")
    print(f"{'='*60}\n")

    # Сохранение
    output = {
        "metadata": {
            "model": QWEN_MODEL,
            "total_evaluated": total,
            "stats": stats,
            "filters": {
                "confidence": args.confidence,
                "mutual_only": args.mutual_only,
                "limit": args.limit
            }
        },
        "evaluations": results
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Сохранено: {args.output}")


if __name__ == "__main__":
    main()
