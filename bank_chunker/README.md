# Chunk Function Bank Processor

## Overview

Script that parses hierarchical Excel tables (point/subpoint structure) into structured JSON chunks optimized for vector search (BM25 + semantic). Each chunk gets two text representations:

- **`chunk_text_plain`** — hierarchical text with up to 3 parent levels as context; all abbreviations expanded inline for BM25 search
- **`chunk_text_context`** — LLM-rephrased single paragraph (GPT-4o) with domain terminology woven in for semantic search. The LLM prompt is dynamically augmented per-chunk with relevant abbreviations and terminology definitions found in the chunk text.

## Project Structure

```
bank_chunker/
├── chunk_function_bank.py            # Main script
├── test_tables.xlsx                  # Input Excel file
├── dictionary/
│   ├── abbreviations.json            # Abbreviation → full name mapping
│   └── terminology_for_llm_full.json # Domain term definitions (see below)
└── docs/                             # Рабочие документы (not in git)
```

Output: `results/chunks_YYYY-MM-DD.json` (auto-created, in `.gitignore`).
Log: `chunk_function_bank.log` (auto-created, in `.gitignore`).
Environment: `.env` in project root (see `../.env.example`).

## Pipeline

1. **Parse Excel → build chunk tree** — nodes (points/subpoints) + parent-child edges
2. **For each chunk:**
   - Build `chunk_text_plain` with up to 3 nearest parents as context
   - Expand abbreviations inline (`МЛ` → `МЛ (маршрутный лист)`)
   - **Search** raw text for terminology matches (3 paths — see below)
   - **Determine expansion level** for each found term (min or full — see below)
   - Build dynamic prompt augmentation block and append it to the system prompt
   - **Check Excel cache**: if the column "Текст для векторизации" already has a value for this row, use it instead of calling LLM. Otherwise, call GPT-4o and write the result back to Excel for future runs.
   - Send to GPT-4o → get `chunk_text_context`
3. **Save** dated JSON to `results/`

## Term Search: Three Paths

When the script processes a chunk, it searches for domain terms using three strategies (`find_terms_in_text`):

| Path | How it works | Example |
|------|-------------|---------|
| **1. full_name** | Exact match of the term's full name in text | `"Конструкторский состав изделия"` found in chunk |
| **2. abbreviation** | Exact match of the abbreviation | `"КСИ"` found in chunk |
| **3. search_roots (AND)** | ALL morphological roots must be present in text | `["конструкторск", "состав"]` — both present → match |

Path 3 is a fallback for inflected forms that don't match the exact `full_name`.

## Two-Level Term Expansion

Not all terms need the same depth of explanation. The script uses a **deterministic** (no LLM) proximity heuristic based on `home_section`:

```
get_expansion_level(chunk_id, home_section):

  chunk_id="2.3.1", home="2.3"  →  startswith("2.3.")  →  min  (own section)
  chunk_id="2.2.4", home="2.3"  →  both start with "2"  →  min  (neighbor section)
  chunk_id="5.4.1", home="2.3"  →  "5" ≠ "2"            →  full (distant section)
```

| Level | What's injected into the LLM prompt | Instruction to LLM |
|-------|-------------------------------------|---------------------|
| **min** | `min_expansion` field (one line) | "используй ТОЛЬКО краткую расшифровку, НЕ добавляй пояснений" |
| **full** | `definition` field (full paragraph) | "вплети суть одним коротким уточняющим предложением" |

**Rationale:** a chunk in section 2.2 (КСИ) doesn't need a full explanation of КСИ — the reader is already in that context. But a chunk in section 5 referencing КСИ needs the full definition to be self-contained after vectorization.

## Terminology JSON Fields

Each entry in `terminology_for_llm_full.json`:

| Field | Used in code | Purpose |
|-------|:---:|---------|
| `full_name` | Yes | Full term name. Used for search (path 1) and display |
| `abbreviation` | Yes | Short form. Used for search (path 2) and the "keep abbreviation in parentheses" rule |
| `search_roots` | Yes | Morphological roots for AND-search (path 3). All must match |
| `home_section` | Yes | Section number where the term is "native". Drives min/full expansion level |
| `min_expansion` | Yes | One-line definition for nearby chunks (min level) |
| `definition` | Yes | Full definition for distant chunks (full level) |
| `aliases` | No | Alternative names, synonyms. Reserved for future use |
| `distinguish_from` | No | Explanations of differences vs similar terms. Reserved for future use |

## Excel Caching

The script uses the input Excel file as a cache for LLM results:

- On first run, it creates a column "Текст для векторизации" and writes each `chunk_text_context` after generation.
- On subsequent runs, if a row already has a value in that column, it skips the LLM call and uses the cached text.
- This saves API costs and ensures reproducibility. To regenerate a specific chunk, clear its cell in Excel.

## Output JSON Structure

```json
{
  "chunk_id": "FB-2.2.4",
  "chunk_text_context": "В рамках управления составами изделий, подсистема конструкторского состава изделия (КСИ) обеспечивает поддержку версий спецификаций. Версионирование КСИ осуществляется по трём критериям применяемости: по сроку действия, по серийному номеру изделия и по конкретному заказу клиента...",
  "chunk_text_plain": "[2] Управление составами изделий\n  →  [2.2] Конструкторский состав изделия (КСИ)\n    →  [2.2.4] Поддержка версий спецификаций КСИ\n    Включает подпункты:\n    - [2.2.4.1] Управление несколькими версиями спецификаций КСИ\n    - [2.2.4.2] Присвоение статусов версиям спецификаций КСИ\n    - [2.2.4.3] Определение применяемости каждой версии спецификации КСИ",
  "metadata": {
    "chunk_id": "FB-2.2.4",
    "level": 3,
    "node_type": "group",
    "parent_id": "FB-2.2",
    "parent_chain": ["2", "2.2"],
    "children": ["2.2.4.1", "2.2.4.2", "2.2.4.3"],
    "children_count": 3,
    "source_type": "function_bank",
    "doc_name": "test_tables.xlsx",
    "has_abbreviations": true,
    "found_terms": ["КСИ"],
    "parents_depth": 2
  }
}
```

## Usage

```bash
python bank_chunker/chunk_function_bank.py --input bank_chunker/test_tables.xlsx
```

Optional overrides:
```bash
python bank_chunker/chunk_function_bank.py \
  --input bank_chunker/test_tables.xlsx \
  --terminology bank_chunker/dictionary/terminology_for_llm_full.json \
  --abbreviations bank_chunker/dictionary/abbreviations.json
```

Environment and dependencies — see root [README.md](../README.md).
