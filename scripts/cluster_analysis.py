"""
cluster_analysis.py — Кластеризация банка функциональности.

1. Вытягивает эмбеддинги из OpenSearch (уже проиндексированы)
2. Кластеризует через HDBSCAN
3. Визуализирует через t-SNE и UMAP
4. Сравнивает кластеры с иерархией банка
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter


def get_embeddings_from_opensearch():
    """Вытягивает эмбеддинги банка из OpenSearch (уже проиндексированы)."""
    from opensearchpy import OpenSearch

    client = OpenSearch(
        hosts=["https://10.40.10.111:9200"],
        http_auth=("admin", "ProcessScout_2026!"),
        use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=60
    )
    print(f"OpenSearch: {client.info()['version']['number']}")

    # Scroll по всем документам банка
    results = []
    resp = client.search(
        index="processscout_bank",
        body={"size": 100, "query": {"match_all": {}},
              "_source": ["embedding", "chunk_text_plain", "chunk_text_context", "metadata"]},
        scroll="2m"
    )
    scroll_id = resp["_scroll_id"]
    results.extend(resp["hits"]["hits"])

    while len(resp["hits"]["hits"]) > 0:
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        results.extend(resp["hits"]["hits"])

    print(f"Получено из OpenSearch: {len(results)} документов")

    chunks = []
    embeddings = []
    for hit in results:
        src = hit["_source"]
        chunks.append({
            "chunk_id": hit["_id"],
            "chunk_text_plain": src.get("chunk_text_plain", ""),
            "chunk_text_context": src.get("chunk_text_context", ""),
            "metadata": src.get("metadata", {})
        })
        embeddings.append(src["embedding"])

    return chunks, np.array(embeddings)


def get_section_id(chunk_id, level=2):
    clean = chunk_id.replace("FB-", "")
    clean = re.sub(r'[а-яё]\)$', '', clean)
    parts = clean.split(".")
    return ".".join(parts[:level])


def cluster_hdbscan(embeddings, min_cluster_size=5, min_samples=3):
    import hdbscan
    print(f"\nHDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Кластеров: {n_clusters}")
    print(f"  Шум: {n_noise} ({n_noise/len(labels)*100:.0f}%)")
    cluster_sizes = Counter(labels)
    if -1 in cluster_sizes: del cluster_sizes[-1]
    sizes = sorted(cluster_sizes.values(), reverse=True)
    print(f"  Размеры: {sizes[:10]}{'...' if len(sizes) > 10 else ''}")
    return labels, clusterer


def reduce_tsne(embeddings):
    from sklearn.manifold import TSNE
    print("\nt-SNE проекция...")
    return TSNE(n_components=2, perplexity=min(30, len(embeddings)-1),
                random_state=42, metric='cosine', n_iter=1000).fit_transform(embeddings)


def reduce_umap(embeddings):
    import umap
    print("UMAP проекция...")
    return umap.UMAP(n_components=2, metric='cosine',
                     n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeddings)


def compare_with_hierarchy(chunks, cluster_labels):
    print("\n" + "="*60)
    print("Сравнение кластеров с иерархией банка")
    print("="*60)

    cluster_sections = defaultdict(list)
    for chunk, label in zip(chunks, cluster_labels):
        cid = chunk['chunk_id']
        cluster_sections[label].append({
            'chunk_id': cid,
            'section_l1': get_section_id(cid, 1),
            'section_l2': get_section_id(cid, 2),
            'text': chunk.get('chunk_text_plain', '')[:100]
        })

    for label in sorted(cluster_sections.keys()):
        items = cluster_sections[label]
        if label == -1:
            print(f"\n--- ШУМ: {len(items)} чанков ---")
            for s, n in Counter(i['section_l2'] for i in items).most_common(5):
                print(f"    {s}: {n}")
            continue

        l1_counts = Counter(i['section_l1'] for i in items)
        l2_counts = Counter(i['section_l2'] for i in items)
        dominant = l2_counts.most_common(1)[0]
        purity = dominant[1] / len(items) * 100

        print(f"\n--- Кластер {label}: {len(items)} чанков ---")
        print(f"  Доминанта: {dominant[0]} ({dominant[1]}/{len(items)}, {purity:.0f}%)")
        if len(l1_counts) > 1:
            print(f"  !!! КРОСС-СЕКЦИОННЫЙ: {dict(l1_counts)}")
        print(f"  Секции: {dict(l2_counts.most_common(5))}")
        for item in items[:3]:
            print(f"    {item['chunk_id']}: {item['text'][:80]}")

    non_noise = [l for l in cluster_labels if l != -1]
    n_clusters = len(set(non_noise))
    purities = []
    cross = 0
    for label in set(non_noise):
        items = cluster_sections[label]
        l2 = Counter(i['section_l2'] for i in items)
        purities.append(l2.most_common(1)[0][1] / len(items))
        if len(set(i['section_l1'] for i in items)) > 1:
            cross += 1

    print(f"\n{'='*60}")
    print(f"ИТОГО:")
    print(f"  Кластеров: {n_clusters}")
    print(f"  Средняя чистота: {np.mean(purities)*100:.0f}%")
    print(f"  Кросс-секционных: {cross} ({cross/n_clusters*100:.0f}%)")
    print(f"  Вывод: {'иерархия ~ кластеры' if np.mean(purities) > 0.7 else 'кластеры != иерархия'}")
    print(f"{'='*60}")
    return cluster_sections


def plot_clusters(coords, cluster_labels, chunks, output_dir, method_name="tsne"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_labels))))
    for label in unique_labels:
        mask = cluster_labels == label
        if label == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], c='lightgray', s=15, alpha=0.4, label='шум')
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[label % 20]], s=25, alpha=0.7, label=f'C{label}')
    ax.set_title(f'HDBSCAN ({method_name.upper()})', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    ax = axes[1]
    section_labels = [get_section_id(c['chunk_id'], 2) for c in chunks]
    unique_sections = sorted(set(section_labels))
    section_colors = [unique_sections.index(s) for s in section_labels]
    ax.scatter(coords[:, 0], coords[:, 1], c=section_colors, cmap='tab20', s=25, alpha=0.7)
    for section in unique_sections:
        mask = np.array([s == section for s in section_labels])
        if mask.sum() > 0:
            cx, cy = coords[mask].mean(axis=0)
            ax.annotate(section, (cx, cy), fontsize=7, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.set_title(f'Иерархия L2 ({method_name.upper()})', fontsize=14)

    plt.tight_layout()
    out = output_dir / f"clusters_{method_name}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {out}")


def save_cluster_data(chunks, cluster_labels, cluster_sections, output_dir):
    result = {"n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
              "n_noise": int((np.array(cluster_labels) == -1).sum()), "clusters": {}}
    for label in sorted(set(cluster_labels)):
        if label == -1: continue
        items = cluster_sections[label]
        l2 = Counter(i['section_l2'] for i in items)
        result["clusters"][str(label)] = {
            "size": len(items), "chunk_ids": [i['chunk_id'] for i in items],
            "sections": dict(l2.most_common()),
            "dominant_section": l2.most_common(1)[0][0],
            "purity": round(l2.most_common(1)[0][1] / len(items), 2)
        }
    out = output_dir / "cluster_assignments.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Сохранено: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/clusters/")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks, embeddings = get_embeddings_from_opensearch()

    cluster_labels, clusterer = cluster_hdbscan(embeddings, args.min_cluster_size, args.min_samples)
    cluster_sections = compare_with_hierarchy(chunks, cluster_labels)

    tsne_coords = reduce_tsne(embeddings)
    plot_clusters(tsne_coords, cluster_labels, chunks, output_dir, "tsne")

    umap_coords = reduce_umap(embeddings)
    plot_clusters(umap_coords, cluster_labels, chunks, output_dir, "umap")

    save_cluster_data(chunks, cluster_labels, cluster_sections, output_dir)
    print(f"\nГотово! Результаты в {output_dir}/")


if __name__ == "__main__":
    main()
