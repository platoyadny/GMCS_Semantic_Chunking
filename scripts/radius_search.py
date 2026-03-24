"""
radius_search.py — Радиальный поиск по банку функциональности.

Принимает текстовый запрос из терминала, кодирует его через BGE-M3,
выполняет script_score запрос к OpenSearch (честный радиусный поиск без
ограничения по количеству результатов), затем распределяет результаты
по трём группам качества в Python.

Использование:
  python radius_search.py \
    --range_3 0.60 \
    --range_2 0.75 \
    --range_1 0.90

Аргументы:
  --range_3  Минимальный порог косинусного сходства для запроса
                          к OpenSearch. Определяет внешний радиус поиска.
                          Все документы ближе этого порога возвращаются.
  --range_2               Нижняя граница группы "совпадения для уточнения".
                          Должен быть строго больше range_3.
  --range_1               Нижняя граница группы "точные совпадения".
                          Должен быть строго больше range_2.
  --output_dir            Директория для выходных файлов (по умолчанию: output)

Выходные файлы:
  results_exact.json    — точные совпадения    (cosine >= range_1)
  results_clarify.json  — совпадения для уточнения (range_2 <= cosine < range_1)
  results_thematic.json — общее тематическое совпадение (min_cos <= cosine < range_2)
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime


OPENSEARCH_URL = "https://10.40.10.111:9200"
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "ProcessScout_2026!"
INDEX_BANK = "processscout_bank"


def load_model():
    from sentence_transformers import SentenceTransformer
    print("Загрузка BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"  Модель загружена, размерность: {model.get_sentence_embedding_dimension()}")
    return model


def encode_query(model, text: str) -> list[float]:
    """
    Кодирует текст запроса в вектор через BGE-M3.
    Идентично тому, как это делается в index_chunks.py:
    normalize_embeddings=True обеспечивает косинусное сходство
    через innerproduct на нормализованных векторах.
    """
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.tolist()


def create_client():
    from opensearchpy import OpenSearch
    client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60
    )
    info = client.info()
    print(f"  OpenSearch: {info['version']['number']}, кластер: {info['cluster_name']}")
    return client


def radius_search(client, query_embedding: list[float], min_cosine: float, index_name=INDEX_BANK) -> list[dict]:
    """
    Выполняет поиск по knn_vector (Faiss) и возвращает все документы с косинусным сходством >= min_cosine.
    """
    body = {
        "size": 10000,  # max_result_window вашего индекса
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": 10000        # кол-во ближайших соседей
                }
            }
        }
    }

    response = client.search(index=index_name, body=body)
    raw_hits = response["hits"]["hits"]
    total_available = response["hits"]["total"]["value"]

    if total_available > len(raw_hits):
        print(
            f"  Предупреждение: найдено {total_available} документов, "
            f"но возвращено только {len(raw_hits)} (лимит size=10000).\n"
            f"  Для полного результата увеличьте index.max_result_window в OpenSearch."
        )

    results = []
    for h in raw_hits:
        # inner product = cosine similarity, нормализуем в диапазон [0,1]
        cosine = round(h["_score"] - 1, 6)  # <-- вычитаем 1
        source = h["_source"]
        results.append({
            "id": h["_id"],
            "cosine_score": cosine,
            "chunk_text_plain": source.get("chunk_text_plain", ""),
            "chunk_text_context": source.get("chunk_text_context", ""),
            "metadata": source.get("metadata", {})
        })

    results.sort(key=lambda x: x["cosine_score"], reverse=True)
    # Отсекаем всё ниже внешнего радиуса
    results = [r for r in results if r["cosine_score"] >= min_cosine]

    return results

def split_by_radius(
    hits: list[dict],
    range_1: float,
    range_2: float,
) -> tuple[list, list, list]:
    """
    Распределяет результаты по трём группам в Python.
    Никаких повторных запросов к OpenSearch — все данные уже получены.

    Группы:
      exact:    cosine >= range_1
      clarify:  range_2 <= cosine < range_1
      thematic: min_cosine <= cosine < range_2

    Порядок внутри каждой группы сохраняется (убывание по cosine_score).
    """
    exact = []
    clarify = []
    thematic = []

    for hit in hits:
        c = hit["cosine_score"]
        if c >= range_1:
            exact.append(hit)
        elif c >= range_2:
            clarify.append(hit)
        else:
            thematic.append(hit)

    return exact, clarify, thematic


def save_group(
    hits: list[dict],
    filepath: Path,
    group_name: str,
    group_label: str,
    shared_meta: dict
):
    """
    Сохраняет одну группу результатов в JSON файл.

    Структура файла:
    {
        "metadata": { общие параметры запроса + статистика группы },
        "hits": [ список документов с cosine_score и всеми полями из БД ]
    }
    """
    output = {
        "metadata": {
            **shared_meta,
            "group": group_name,
            "group_label": group_label,
            "count": len(hits),
            "timestamp": datetime.now().isoformat()
        },
        "hits": hits
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def validate_thresholds(min_cos: float, range_2: float, range_1: float):
    """
    Проверяет что пороги корректны: находятся в [-1, 1] и строго возрастают.
    """
    errors = []

    for name, val in [
        ("--range_3", min_cos),
        ("--range_2", range_2),
        ("--range_1", range_1)
    ]:
        if not (-1.0 <= val <= 1.0):
            errors.append(f"{name}={val} должен быть в диапазоне [-1.0, 1.0]")

    if min_cos >= range_2:
        errors.append(
            f"--range_3 ({min_cos}) должен быть строго меньше --range_2 ({range_2})"
        )
    if range_2 >= range_1:
        errors.append(
            f"--range_2 ({range_2}) должен быть строго меньше --range_1 ({range_1})"
        )

    if errors:
        print("\nОшибки в параметрах:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Радиальный поиск по банку функциональности (processscout_bank)"
    )
    parser.add_argument(
        "--range_3", type=float, required=True,
        help="Минимальный порог косинусного сходства для запроса к OpenSearch (внешний радиус)"
    )
    parser.add_argument(
        "--range_2", type=float, required=True,
        help="Нижняя граница группы 'совпадения для уточнения' (должен быть > range_3)"
    )
    parser.add_argument(
        "--range_1", type=float, required=True,
        help="Нижняя граница группы 'точные совпадения' (должен быть > range_2)"
    )
    parser.add_argument(
        "--output_dir", default="output",
        help="Директория для выходных файлов (по умолчанию: output)"
    )
    args = parser.parse_args()

    validate_thresholds(args.range_3, args.range_2, args.range_1)

    # --- Запрос текста ---
    print("\nВведите текст для поиска:")
    query_text = input("> ").strip()
    if not query_text:
        print("Ошибка: текст запроса не может быть пустым.")
        sys.exit(1)

    print(f"\nЗапрос: «{query_text}»")
    print(
        f"Пороги: "
        f"точные >= {args.range_1} | "
        f"уточнение >= {args.range_2} | "
        f"тематические >= {args.range_3}"
    )

    # --- Загрузка модели и подключение ---
    model = load_model()
    print("\nПодключение к OpenSearch...")
    client = create_client()

    # --- Кодирование запроса ---
    print("\nКодирование запроса...")
    query_embedding = encode_query(model, query_text)
    print("  Готово.")

    # --- Поиск ---
    print(f"\nПоиск в {INDEX_BANK} (min_cosine={args.range_3})...")
    hits = radius_search(client, query_embedding, args.range_3)
    print(f"  Получено документов: {len(hits)}")

    if not hits:
        print(
            "\nНет результатов в заданном радиусе. "
            "Попробуйте снизить --range_3."
        )
        sys.exit(0)

    # --- Разбивка по группам ---
    exact, clarify, thematic = split_by_radius(
        hits, args.range_1, args.range_2
    )

    print(f"\nРаспределение по группам:")
    print(f"  Точные совпадения    cosine >= {args.range_1}:                        {len(exact)}")
    print(f"  Для уточнения        {args.range_2} <= cosine < {args.range_1}:  {len(clarify)}")
    print(f"  Тематические         {args.range_3} <= cosine < {args.range_2}:  {len(thematic)}")

    # --- Сохранение ---
    output_dir = Path(args.output_dir)

    shared_meta = {
        "query_text": query_text,
        "index": INDEX_BANK,
        "range_3": args.range_3,
        "range_2": args.range_2,
        "range_1": args.range_1,
        "total_hits": len(hits),
    }

    groups = [
        (
            exact,
            "results_exact.json",
            "exact",
            f"Точные совпадения (cosine >= {args.range_1})"
        ),
        (
            clarify,
            "results_clarify.json",
            "clarify",
            f"Совпадения для уточнения ({args.range_2} <= cosine < {args.range_1})"
        ),
        (
            thematic,
            "results_thematic.json",
            "thematic",
            f"Тематические совпадения ({args.range_3} <= cosine < {args.range_2})"
        ),
    ]

    print(f"\nСохранение в {output_dir}/...")
    for hits_group, filename, group_name, group_label in groups:
        filepath = output_dir / filename
        save_group(hits_group, filepath, group_name, group_label, shared_meta)
        print(f"  {filepath}: {len(hits_group)} документов")

    print("\nГотово.")


if __name__ == "__main__":
    main()