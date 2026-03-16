# Chunker Project

Подготовка данных для RAG-маппинга: документы обследования AS-IS → банк функциональности PLM/ERP.

Проект состоит из двух модулей чанкинга. Оба генерируют JSON с полями `search_text` + `full_context`, пригодные для гибридного поиска (BM25 + semantic).

## Модули

### [bank_chunker/](bank_chunker/) — чанкинг банка функциональности

Парсит иерархические Excel-таблицы (пункты/подпункты) → JSON-чанки. Каждый чанк получает два текстовых представления:

- **`chunk_text_plain`** — иерархический текст с контекстом до 3 родителей + раскрытые аббревиатуры (для BM25)
- **`chunk_text_context`** — LLM-переформулированный связный абзац с терминологией (для semantic search)

Ключевая особенность — **двухуровневое раскрытие терминов**: близкие термины (свой/соседний раздел) получают краткую расшифровку, далёкие — полное определение. Промпт LLM динамически дополняется для каждого чанка.

Подробнее: [bank_chunker/README.md](bank_chunker/README.md)

### [asis_chunker/](asis_chunker/) — чанкинг документов обследования

Парсит `.docx` документы обследования бизнес-процессов → JSON-чанки. Для каждого шага таблицы создаёт до 3 чанков с разным фокусом:

- **step_table** — название операции + контекст БП (точный BM25-поиск)
- **step_narrative** — абзац нарратива с привязкой по номеру шага (семантический поиск)
- **step_enriched** — операция + исполнитель + вход/выход + система (маппинг по ролям и документам)

Также чанкует рекомендации (по автоматизации, процессу, консалтингу), карточки процесса и особенности AS-IS.

Подробнее: [asis_chunker/README.md](asis_chunker/README.md)

## Следующий шаг

Векторизация BGE-M3 + загрузка в OpenSearch для гибридного поиска и маппинга AS-IS ↔ банк функциональности.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# заполнить .env своими ключами
```

### Переменные окружения

| Переменная | Описание | Обязательность |
|-----------|----------|---------------|
| `OPENAI_API_KEY` | API-ключ OpenAI | Всегда для bank_chunker; для asis_chunker — только с `--use-llm` |
| `OPENAI_MODEL` | Модель (по умолчанию `gpt-4o`) | Опционально |

### Зависимости

| Пакет | bank_chunker | asis_chunker |
|-------|:---:|:---:|
| openpyxl | обязательно | — |
| langchain-openai | обязательно | только `--use-llm` |
| langchain-core | обязательно | — |
| python-dotenv | обязательно | только `--use-llm` |

`asis_chunker` без `--use-llm` работает **без внешних зависимостей** (только stdlib Python 3.10+).

## Project Structure

```
chunker_project/
├── .env.example              # Шаблон переменных окружения
├── .gitignore                # Общий для всего проекта
├── requirements.txt          # Объединённые зависимости обоих модулей
├── output/                   # Общая папка для выходных данных
├── bank_chunker/
│   ├── chunk_function_bank.py
│   ├── test_tables.xlsx
│   ├── dictionary/
│   │   ├── abbreviations.json
│   │   └── terminology_for_llm_full.json
│   └── README.md
└── asis_chunker/
    ├── chunk_asis.py
    ├── output/
    └── README.md
```

## Запуск

```bash
# bank_chunker
python bank_chunker/chunk_function_bank.py --input bank_chunker/test_tables.xlsx

# asis_chunker (без LLM)
python asis_chunker/chunk_asis.py --input БП_8.docx

# asis_chunker (с LLM)
python asis_chunker/chunk_asis.py --input БП_8.docx --use-llm
```
