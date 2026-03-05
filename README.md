# Chunk Function Bank Processor

## Overview

Script that parses hierarchical Excel tables (point/subpoint structure) into structured JSON chunks optimized for vector search (BM25 + semantic). Each chunk gets two text representations:

- **`chunk_text_plain`** — hierarchical text with up to 3 parent levels as context; all abbreviations expanded inline for BM25 search
- **`chunk_text_context`** — LLM-rephrased single paragraph (GPT-4o) with domain terminology woven into the text for semantic search

The script dynamically augments the LLM prompt per-chunk with relevant abbreviations and terminology definitions found in the chunk text.

## Project Structure

```
chunker_project/
├── chunk_function_bank.py      # Main script
├── run.sh                      # Launch script
├── .env                        # API keys (not in git)
├── test_tables.xlsx            # Input Excel file
├── dictionary/
│   ├── abbreviations.json      # Abbreviation → full name mapping (68 entries)
│   └── terminology_for_llm_full.json  # Domain term definitions with search_roots
└── results/
    └── chunks_YYYY-MM-DD.json  # Output (auto-dated)
```

## Output JSON Structure

```json
{
  "chunk_id": "FB-2.2.4",
  "chunk_text_context": "Раздел: Управление составами изделий  \nПодраздел: Конструкторский состав изделия (КСИ)  \nФункция: Поддержка версий спецификаций КСИ  \n\nОписание: Система обеспечивает управление версиями спецификаций Конструкторского состава изделия (КСИ), что позволяет отслеживать изменения в составе изделия по серийному номеру и конкретному заказу клиента. Это необходимо для точного планирования, кооперации и управления изменениями, обеспечивая соответствие между конструкторскими обозначениями и кодами нормативно-справочной информации (НСИ).  \n\nКлючевые слова: КСИ, спецификации, версионирование, серийный номер, заказ клиента",
  "chunk_text_plain": "[2] Управление составами изделий\n  →  [2.2] Конструкторский состав изделия (КСИ (Конструкторский состав изделия))\n    →  [2.2.4] Поддержка версий спецификаций КСИ (Конструкторский состав изделия)\n    Включает подпункты:\n    - [2.2.4.1] Управление несколькими версиями спецификаций КСИ (Конструкторский состав изделия)\n    - [2.2.4.2] Присвоение статусов версиям спецификаций КСИ (Конструкторский состав изделия)\n    - [2.2.4.3] Определение применяемости каждой версии спецификации КСИ (Конструкторский состав изделия):\n",
  "metadata": {
    "chunk_id": "FB-2.2.4",
    "level": 3,
    "node_type": "group",
    "parent_id": "FB-2.2",
    "parent_chain": [
      "2",
      "2.2"
    ],
    "children": [
      "2.2.4.1",
      "2.2.4.2",
      "2.2.4.3"
    ],
    "children_count": 3,
    "source_type": "function_bank",
    "doc_name": "test_tables.xlsx",
    "has_abbreviations": true,
    "found_terms": [
      "КСИ"
    ],
    "parents_depth": 2
  }
}
```

## Pipeline

1. Parse Excel → build tree (nodes + parent-child edges)
2. For each chunk:
   - Build `chunk_text_plain` with up to 3 nearest parents
   - Expand abbreviations inline (`МЛ` → `МЛ (маршрутный лист)`)
   - Search raw text for abbreviations and terminology matches (full_name, abbreviation, AND-search by `search_roots`)
   - Build dynamic prompt augmentation with found terms and definitions
   - Send to GPT-4o → get `chunk_text_context`
3. Save dated JSON to `results/`

## Environment

`.env` file in project root:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

## Requirements

- openpyxl
- langchain_openai
- python-dotenv
- tiktoken

## Usage

```bash
./run.sh
```

Or manually:
```bash
python chunk_function_bank.py --input test_tables.xlsx
```

Optional overrides:
```bash
python chunk_function_bank.py \
  --input test_tables.xlsx \
  --terminology dictionary/terminology_for_llm_full.json \
  --abbreviations dictionary/abbreviations.json
```
