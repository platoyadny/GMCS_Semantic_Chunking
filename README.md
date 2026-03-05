# Chunk Function Bank Processor

## Overview

This script reads an `.xlsx` file containing two columns:

- hierarchical points/subpoints  
- description text  

It builds a hierarchical tree structure and exports the data into a structured `.json` file with metadata.

The script also generates a `.log` file containing execution details, warnings, and errors.

---

## Output JSON Structure

Each chunk in the output JSON has the following structure:

```json
{
  "chunk_id": "FB-4.1.2.2",
  "chunk_text": "[4] Планирование производства\n  → [4.1] Планирование под объекты...\n    → [4.1.2] Алгоритм планирования...\n      → [4.1.2.2] Поддержка учета ограничений...",
  "metadata": {
    "chunk_id": "FB-4.1.2.2",
    "level": 4,
    "node_type": "leaf",
    "parent_id": "FB-4.1.2",
    "parent_chain": ["4", "4.1", "4.1.2"],
    "children": [],
    "children_count": 0,
    "source_type": "function_bank",
    "doc_name": "bank_sample.xlsx"
  }
}
```

## Requirements

- openpyxl

## Usage

```bash
python chunk_function_bank.py --input <source.xlsx> --output <result.json>
```
