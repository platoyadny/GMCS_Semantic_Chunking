#!/bin/bash
cd "$(dirname "$0")"
../.venv/bin/python3 chunk_function_bank.py \
  --input test_tables.xlsx
