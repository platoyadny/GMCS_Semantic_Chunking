"""
run_demo.py — Демонстрационный пайплайн ProcessScout AI.

  python scripts/run_demo.py --limit 50
  python scripts/run_demo.py --skip-judge
"""

import json
import re
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

MAPPING_RESULTS = "output/mapping_results.json"
ASIS_CHUNKS = "output/chunks_bp-06.json"
BANK_CHUNKS = "output/chunks_bank.json"
JUDGE_OUTPUT = "output/judge_results_demo.json"
REPORT_OUTPUT = "output/report_demo.html"

QWEN_URL = "http://10.40.1.102:8000/v1/chat/completions"
QWEN_MODEL = "qdzzzxc/RuadaptQwen3-32B-Instruct-AWQ"

SYSTEM_PROMPT = """Ты эксперт по внедрению PLM/ERP систем на промышленных предприятиях.

Задача: оценить, насколько пункт банка функциональности PLM/ERP покрывает шаг бизнес-процесса AS-IS.

СТРОГИЕ правила оценки:

"full" (green, 90-100%) — ТОЛЬКО если выполнены ОБА условия:
  1. Та же ФУНКЦИЯ (не просто тематическая область, а конкретная операция)
  2. Те же СУЩНОСТИ (документы, роли, типы данных совпадают или являются прямыми аналогами)

"partial" (yellow, 30-89%) — функция ЧАСТИЧНО совпадает:
  - Основная операция похожа, но детали отличаются
  - Банк покрывает ЧАСТЬ того, что описано в шаге AS-IS

"none" (orange, 0%) — функция НЕ совпадает:
  - Совпадение только по общим словам
  - Тематическая область может совпадать, но операции разные

ВАЖНО: оценивай ФУНКЦИОНАЛЬНОЕ соответствие, не лексическое. Учитывай ИЕРАРХИЮ банка.

Ответь СТРОГО в JSON (без markdown, без ```):
{
  "match": "full" или "partial" или "none",
  "coverage_percent": число от 0 до 100,
  "covered": "что покрыто",
  "not_covered": "что не покрыто",
  "rationale": "обоснование 1-3 предложения",
  "color": "green" или "yellow" или "orange"
}"""


def call_qwen(system, user, max_retries=3):
    import requests
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 1024, "temperature": 0.1
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(QWEN_URL, json=payload, timeout=120)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"      Ошибка Qwen (попытка {attempt+1}): {e}")
            time.sleep(3)
    return ""


def parse_json_response(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: pass
    try: return json.loads(text.strip())
    except: return {}


def build_user_prompt(asis_context, bank_plain, bank_context, bank_id, asis_meta):
    return f"""== ШАГ AS-IS ==
Процесс: {asis_meta.get('bp_name', '')}
Шаг: {asis_meta.get('step_id', '')} — {asis_meta.get('operation', '')}
Исполнитель: {asis_meta.get('executor', '')}
Система: {asis_meta.get('system', '')}

Полный контекст:
{asis_context}

== ПУНКТ БАНКА ==
ID: {bank_id}

Иерархия:
{bank_plain}

Описание:
{bank_context}

== ЗАДАЧА ==
Оцени покрытие. Ответь в JSON."""


def run_judge(mappings, asis_all, bank_all, pair_to_chunks, limit=None):
    results = []
    total = min(limit, len(mappings)) if limit else len(mappings)
    stats = {"full": 0, "partial": 0, "none": 0, "error": 0}

    print(f"\nLLM-as-a-Judge: {total} пар")
    print(f"Qwen: {QWEN_MODEL}\n")

    for i, mapping in enumerate(mappings[:total]):
        pair_id = mapping['pair_id']
        bank_id = mapping['bank_id']

        asis_chunks = pair_to_chunks.get(pair_id, [])
        if not asis_chunks and pair_id in asis_all:
            asis_chunks = [asis_all[pair_id]]
        if not asis_chunks:
            stats["error"] += 1; continue

        enriched = next((c for c in asis_chunks if c['metadata'].get('chunk_type') == 'step_enriched'), asis_chunks[0])
        asis_context = enriched.get('full_context', enriched.get('search_text', ''))
        asis_meta = enriched.get('metadata', {})

        bank_chunk = bank_all.get(bank_id)
        if not bank_chunk:
            stats["error"] += 1; continue

        prompt = build_user_prompt(asis_context, bank_chunk.get('chunk_text_plain', ''),
                                   bank_chunk.get('chunk_text_context', ''), bank_id, asis_meta)
        raw = call_qwen(SYSTEM_PROMPT, prompt)
        evaluation = parse_json_response(raw)

        if not evaluation:
            stats["error"] += 1
            results.append({**mapping, "evaluation": {"error": "parse_failed", "raw": raw[:300]}})
            continue

        match_type = evaluation.get("match", "none")
        stats[match_type] = stats.get(match_type, 0) + 1
        results.append({**mapping, "evaluation": evaluation})

        coverage = evaluation.get("coverage_percent", "?")
        indicator = {"full": "G", "partial": "Y", "none": "O"}.get(match_type, "?")
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] full={stats['full']} partial={stats['partial']} none={stats['none']} err={stats['error']}")
        else:
            print(f"  [{i+1}/{total}] [{indicator}] {pair_id} <-> {bank_id}: {match_type} ({coverage}%)")

    print(f"\n  Итого: full={stats['full']} partial={stats['partial']} none={stats['none']} err={stats['error']}")
    return results, stats


def expand_subpoints(bank_id, bank_all):
    prefix = bank_id.replace("FB-", "")
    subpoints = []
    for cid, chunk in bank_all.items():
        clean_cid = cid.replace("FB-", "")
        if clean_cid.startswith(prefix + ".") or (clean_cid.startswith(prefix) and re.search(r'[а-яё]\)$', clean_cid)):
            if clean_cid != prefix:
                subpoints.append({
                    "id": cid,
                    "plain": chunk.get("chunk_text_plain", ""),
                    "context": chunk.get("chunk_text_context", "")
                })
    subpoints.sort(key=lambda x: x["id"])
    return subpoints


def generate_report(judge_results, asis_all, bank_all, pair_to_chunks, stats, output_path, min_coverage):
    good = [item for item in judge_results
            if isinstance(item.get("evaluation", {}).get("coverage_percent", 0), (int, float))
            and item["evaluation"]["coverage_percent"] >= min_coverage]

    bad = [item for item in judge_results
           if item.get("evaluation", {}).get("match") == "none"]

    by_bp = defaultdict(list)
    for item in good:
        by_bp[item.get("bp_id", "unknown")].append(item)

    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    total_evaluated = len(judge_results)
    total_good = len(good)
    total_none = len(bad)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ProcessScout AI — Результаты маппинга</title>
<style>
:root{{--bg:#FAFAF8;--text:#1A1A18;--text2:#6B6A65;--border:#E5E4DF;--green:#40916C;--green-bg:#D8F3DC;--yellow:#E9A820;--yellow-bg:#FFF3CD;--orange:#D85A30;--orange-bg:#FFEBEE;--card:#FFF;--muted:#F3F2EE;--blue:#2E86AB;--blue-bg:#E4F1F7}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;font-size:14px}}
.container{{max-width:1100px;margin:0 auto;padding:32px 24px 80px}}
header{{margin-bottom:40px;padding-bottom:24px;border-bottom:2px solid var(--border)}}
header h1{{font-size:28px;font-weight:700;margin-bottom:4px}}
header .meta{{color:var(--text2);font-size:13px}}
.summary{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:32px}}
.summary-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center}}
.summary-card .num{{font-size:28px;font-weight:700}}
.summary-card .label{{font-size:11px;color:var(--text2);margin-top:4px}}
.pipeline-desc{{background:var(--blue-bg);border:1px solid var(--blue);border-radius:10px;padding:20px;margin-bottom:32px;font-size:13px;color:#1a3a4a}}
.pipeline-desc h3{{font-size:15px;margin-bottom:8px;color:var(--blue)}}
.pipeline-desc ol{{padding-left:20px;margin:8px 0}}
.pipeline-desc li{{margin-bottom:4px}}
.section-title{{font-size:20px;font-weight:700;margin:32px 0 16px;padding-bottom:8px;border-bottom:2px solid var(--border)}}
.bp-section{{margin-bottom:24px}}
.bp-header{{font-size:16px;font-weight:700;padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:12px;color:var(--text2)}}
.mapping{{background:var(--card);border:1px solid var(--border);border-radius:10px;margin-bottom:12px;overflow:hidden}}
.mapping-header{{padding:12px 16px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border)}}
.mapping-header .pair{{font-weight:600;font-size:14px}}
.badge{{padding:3px 10px;border-radius:100px;font-size:11px;font-weight:600}}
.badge-green{{background:var(--green-bg);color:var(--green)}}
.badge-yellow{{background:var(--yellow-bg);color:var(--yellow)}}
.badge-orange{{background:var(--orange-bg);color:var(--orange)}}
.mapping-body{{padding:12px 16px;font-size:13px}}
.row{{display:grid;grid-template-columns:90px 1fr;gap:6px;margin-bottom:6px}}
.row .lbl{{color:var(--text2);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.5px;padding-top:2px}}
.subpoints{{margin-top:10px;padding:10px 14px;background:var(--muted);border-radius:8px}}
.subpoints h4{{font-size:12px;font-weight:600;margin-bottom:6px;color:var(--text2)}}
.subpoint{{padding:4px 0;border-bottom:1px solid var(--border);font-size:12px}}
.subpoint:last-child{{border-bottom:none}}
.sp-id{{font-weight:600;color:var(--green)}}
.none-item{{background:var(--card);border:1px solid var(--border);border-left:3px solid var(--orange);border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;font-size:13px}}
.none-pair{{font-weight:600}}
.none-reason{{color:var(--text2);margin-top:4px}}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>ProcessScout AI — Результаты маппинга</h1>
  <div class="meta">{now} · БП_ASIS-06 (14 бизнес-процессов) · Банк секции 4-5 (392 пункта) · Порог ≥{min_coverage}%</div>
</header>

<div class="pipeline-desc">
  <h3>Как работает пайплайн</h3>
  <ol>
    <li><strong>Чанкинг</strong> — документ AS-IS (754 чанка) и банк функциональности (392 чанка) нарезаются с LLM-переформулировкой</li>
    <li><strong>Эмбеддинги</strong> — BGE-M3 (1024 измерений) генерирует семантические векторы для каждого чанка</li>
    <li><strong>Двухпроходный поиск</strong> — BM25 (лексика) + KNN (семантика) через OpenSearch RRF. Проход 1: AS-IS → банк. Проход 2: банк → AS-IS. Пары с взаимным подтверждением получают высший приоритет</li>
    <li><strong>Chunk-type фильтр</strong> — кандидат, найденный step_table + step_narrative + step_enriched (3 разных формулировки) → сильный сигнал. Найден только enriched → шум, отсечён</li>
    <li><strong>LLM-as-a-Judge (Qwen 32B)</strong> — для каждой пары: полный контекст шага AS-IS + полный текст пункта банка с иерархией → оценка: match (full/partial/none), coverage %, обоснование</li>
    <li><strong>Раскрытие подпунктов</strong> — для подтверждённых маппингов (≥{min_coverage}%) показываются подпункты банка как детали реализации</li>
  </ol>
</div>

<div class="summary">
  <div class="summary-card"><div class="num">{total_evaluated}</div><div class="label">Оценено пар</div></div>
  <div class="summary-card"><div class="num" style="color:var(--green)">{stats.get('full',0)}</div><div class="label">Full (green)</div></div>
  <div class="summary-card"><div class="num" style="color:var(--yellow)">{stats.get('partial',0)}</div><div class="label">Partial (yellow)</div></div>
  <div class="summary-card"><div class="num" style="color:var(--orange)">{total_none}</div><div class="label">None (отсечено)</div></div>
  <div class="summary-card"><div class="num" style="color:var(--green)">{total_good}</div><div class="label">≥{min_coverage}% в отчёте</div></div>
</div>
"""

    # === ХОРОШИЕ МАППИНГИ ===
    html += f'<div class="section-title">Подтверждённые маппинги (coverage ≥ {min_coverage}%)</div>\n'

    for bp_id in sorted(by_bp.keys()):
        items = by_bp[bp_id]
        bp_name = ""
        for cid, c in asis_all.items():
            if c['metadata'].get('bp_id') == bp_id:
                bp_name = c['metadata'].get('bp_name', ''); break

        html += f'<div class="bp-section">\n<div class="bp-header">{bp_id}: {bp_name}</div>\n'

        by_pair = defaultdict(list)
        for item in items:
            by_pair[item["pair_id"]].append(item)

        for pair_id in sorted(by_pair.keys()):
            pair_items = by_pair[pair_id]
            asis_text = ""
            for cid, c in asis_all.items():
                meta = c['metadata']
                if meta.get('pair_id') == pair_id and meta.get('chunk_type') == 'step_table':
                    asis_text = c['search_text']; break
                elif cid == pair_id:
                    asis_text = c['search_text']; break

            for item in pair_items:
                ev = item.get("evaluation", {})
                coverage = ev.get("coverage_percent", 0)
                match = ev.get("match", "partial")
                color = "green" if match == "full" else "yellow"
                bank_id = item["bank_id"]
                bank_chunk = bank_all.get(bank_id, {})
                bank_plain = bank_chunk.get("chunk_text_plain", "")
                subpoints = expand_subpoints(bank_id, bank_all)

                html += f'<div class="mapping">\n'
                html += f'  <div class="mapping-header"><div class="pair">{pair_id} → {bank_id}</div>'
                html += f'<span class="badge badge-{color}">{match} {coverage}%</span></div>\n'
                html += f'  <div class="mapping-body">\n'
                html += f'    <div class="row"><div class="lbl">AS-IS</div><div>{asis_text}</div></div>\n'
                html += f'    <div class="row"><div class="lbl">Банк</div><div>{bank_plain[:350]}</div></div>\n'
                html += f'    <div class="row"><div class="lbl">Покрыто</div><div>{ev.get("covered","")}</div></div>\n'
                not_cov = ev.get("not_covered", "")
                if not_cov:
                    html += f'    <div class="row"><div class="lbl">Не покрыто</div><div>{not_cov}</div></div>\n'
                html += f'    <div class="row"><div class="lbl">Обоснование</div><div>{ev.get("rationale","")}</div></div>\n'

                if subpoints:
                    html += f'    <div class="subpoints"><h4>Подпункты ({len(subpoints)}) — детали реализации:</h4>\n'
                    for sp in subpoints:
                        sp_lines = sp["plain"].strip().split("\\n")
                        sp_short = sp_lines[-1].strip() if sp_lines else sp["plain"][:200]
                        sp_short = re.sub(r'^→\\s*', '', sp_short).strip()
                        html += f'      <div class="subpoint"><span class="sp-id">{sp["id"]}</span>: {sp_short}</div>\n'
                    html += f'    </div>\n'

                html += f'  </div>\n</div>\n'
        html += f'</div>\n'

    # === ОТСЕЧЁННЫЕ (NONE) ===
    if bad:
        html += f'<div class="section-title">Отсечённые пары (none — функция не совпадает)</div>\n'
        for item in bad[:15]:
            ev = item.get("evaluation", {})
            pair_id = item["pair_id"]
            bank_id = item["bank_id"]
            asis_text = ""
            for cid, c in asis_all.items():
                meta = c['metadata']
                if meta.get('pair_id') == pair_id and meta.get('chunk_type') == 'step_table':
                    asis_text = c['search_text'][:150]; break
                elif cid == pair_id:
                    asis_text = c['search_text'][:150]; break

            html += f'<div class="none-item">\n'
            html += f'  <div class="none-pair">{pair_id} → {bank_id} <span class="badge badge-orange">none</span></div>\n'
            html += f'  <div class="none-reason">AS-IS: {asis_text}</div>\n'
            html += f'  <div class="none-reason">Причина: {ev.get("rationale","")}</div>\n'
            html += f'</div>\n'

    html += """
<footer style="margin-top:40px;padding-top:16px;border-top:1px solid var(--border);color:var(--text2);font-size:12px">
  ProcessScout AI · Пилотный прогон · BGE-M3 + OpenSearch 3.0 RRF + Qwen3-32B
</footer>
</div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nОтчёт: {output_path}")
    print(f"Пар ≥{min_coverage}%: {total_good}, отсечено (none): {total_none}")


def main():
    parser = argparse.ArgumentParser(description="ProcessScout AI — демо")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge-input", default=JUDGE_OUTPUT)
    parser.add_argument("--min-coverage", type=int, default=70)
    args = parser.parse_args()

    min_cov = args.min_coverage

    print("=" * 60)
    print("ProcessScout AI — Демонстрационный пайплайн")
    print("=" * 60)

    print("\nЗагрузка данных...")
    with open(MAPPING_RESULTS) as f:
        search_results = json.load(f)
    with open(ASIS_CHUNKS) as f:
        asis_list = json.load(f)
        asis_all = {c['chunk_id']: c for c in asis_list}
    with open(BANK_CHUNKS) as f:
        bank_list = json.load(f)
        bank_all = {}
        for c in bank_list:
            cid = c.get('metadata', {}).get('chunk_id', c.get('chunk_id', ''))
            bank_all[cid] = c

    pair_to_chunks = defaultdict(list)
    for c in asis_list:
        pid = c['metadata'].get('pair_id', c['chunk_id'])
        pair_to_chunks[pid].append(c)

    mappings = search_results['mappings']
    strong = [m for m in mappings if m['quality'] == 'strong']
    print(f"  Всего пар: {len(mappings)}, strong: {len(strong)}")

    if args.skip_judge:
        print(f"\nЗагрузка: {args.judge_input}")
        with open(args.judge_input) as f:
            judge_data = json.load(f)
        judge_results = judge_data['evaluations']
        stats = judge_data['metadata']['stats']
    else:
        judge_results, stats = run_judge(strong, asis_all, bank_all, pair_to_chunks, limit=args.limit)
        judge_output = {
            "metadata": {"model": QWEN_MODEL, "total": len(judge_results), "stats": stats,
                         "timestamp": datetime.now().isoformat()},
            "evaluations": judge_results
        }
        with open(JUDGE_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(judge_output, f, indent=2, ensure_ascii=False)
        print(f"\nJudge results: {JUDGE_OUTPUT}")

    print(f"\nГенерация отчёта (≥{min_cov}%)...")
    generate_report(judge_results, asis_all, bank_all, pair_to_chunks, stats, REPORT_OUTPUT, min_cov)

    print(f"\n{'=' * 60}")
    print(f"ГОТОВО. Отчёт: {REPORT_OUTPUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
