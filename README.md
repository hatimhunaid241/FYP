# HKTVmall Semantic Search Project

**Team:** Sutanto, Winiera | Suhandjaja, Alexander Gaudi | Quettawalla, Hatim

## Overview

A bilingual (English + Chinese) semantic product search system for HKTVmall.

**Core idea:** When a user searches for a keyword like *"apple"* on HKTVmall,
they currently receive ~1 000 mixed, unrelated products in random order —
Apple iPhones, Apple Pencils, Apple Watches, and actual apples (the fruit)
all jumbled together.

This system clusters those results into meaningful groups so users can
immediately navigate to the category they care about:

```
Search: "apple"
──────────────────────────────────────────────────
Cluster 0 — iphone | 蘋果 | apple     (312 products)
Cluster 1 — pencil | apple | 手寫筆    (87 products)
Cluster 2 — watch | 手錶 | apple       (156 products)
Cluster 3 — fruit | 新鮮 | 青蘋果      (45 products)
──────────────────────────────────────────────────
Pick a cluster, then sub-search within it:
  Cluster 0 > "iPhone 15 Pro Max"  → top 10 ranked results
```

Each keyword is processed **independently** through the full pipeline.
Clusters for "apple" are formed only from apple search results, clusters
for "milk" only from milk results, and so on.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# All 20 keywords, all 7 steps
python run_pipeline.py

# Specific keywords only
python run_pipeline.py --keywords apple milk fan

# Skip data collection (if products are already collected)
python run_pipeline.py --skip collect

# Resume from a specific step
python run_pipeline.py --from cluster --keywords apple

# Force re-run (overwrite existing files)
python run_pipeline.py --force --keywords apple
```

### 3. Use the search interface

```bash
# Interactive mode (guided session)
python src/search/semantic_search.py

# Single keyword — show cluster overview
python src/search/semantic_search.py --keyword apple

# Sub-search within a cluster
python src/search/semantic_search.py --keyword apple --cluster 0 --query "iPhone 15 Pro"

# Chinese queries work too
python src/search/semantic_search.py --keyword mask --cluster 1 --query "護膚面膜"
```

### 4. Individual pipeline steps

```bash
# Step 1: Collect ~1 000 products per keyword
python src/data_collection/collect_products.py --keywords apple milk

# Step 2: Generate embeddings (multilingual, per keyword)
python src/embeddings/embedding_generator.py --keywords apple
python src/embeddings/embedding_generator.py --visualize --keywords apple  # 2-D plot

# Step 3: K-Means clustering (auto-K per keyword)
python src/clustering/kmeans_cluster.py --keywords apple
python src/clustering/kmeans_cluster.py --n_clusters 5 --keywords fan  # fixed K

# Step 4: TF-IDF cluster labels
python src/clustering/cluster_labeling.py --keywords apple

# Step 5: Clustering quality metrics
python src/clustering/clustering_metrics.py --keyword apple

# Step 6: Semantic vs keyword comparison
python src/evaluation/search_metrics.py --keywords apple milk

# Step 7: Charts from saved results
python src/evaluation/visualization.py

# Keyword baseline comparison
python src/search/keyword_baseline.py --keyword apple --query "iphone 15"
python src/search/keyword_baseline.py --keyword mask --query "surgical mask" --mode tfidf
```

---

## Architecture — Per-Keyword Pipeline

```
HKTVmall API  (Algolia)
       │
       ▼  collect_products.py
data/processed/
  apple/                        milk/                        mask/
    products.parquet              products.parquet              products.parquet
       │                             │                              │
       ▼  embedding_generator.py     ▼                              ▼
    product_embeddings.npy        product_embeddings.npy        product_embeddings.npy
    products_with_embeddings.parquet
       │
       ▼  kmeans_cluster.py  (auto-K: elbow + silhouette)
    kmeans_clusters.csv           ← cluster IDs for each product
    kmeans_elbow.png
       │
       ▼  cluster_labeling.py  (TF-IDF)
    clusters_labeled.csv          ← + cluster_label column
    cluster_labels.csv            ← compact id → label lookup
       │
       ├──► semantic_search.py     ← cluster overview + cosine sub-search
       └──► keyword_baseline.py    ← TF-IDF baseline for comparison
                    │
                    ▼  search_metrics.py + visualization.py
              results/tables/comparison_summary.csv
              results/figures/*.png
```

**Key point:** Each keyword's products are clustered only among themselves.
A product returned by the "apple" search is never mixed with "milk" products.

---

## Project Structure

```
FYP/
├── run_pipeline.py                  ← End-to-end pipeline runner
├── config/config.yaml               ← All settings (single source of truth)
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── api_client.py            ← HKTVmall / Algolia API wrapper
│   │   └── collect_products.py      ← Per-keyword product collector
│   ├── preprocessing/
│   │   ├── text_cleaner.py          ← Bilingual cleaner (numbers preserved)
│   │   ├── chinese_processor.py     ← jieba + OpenCC Trad→Simplified
│   │   └── english_processor.py     ← Lowercasing, punctuation removal
│   ├── embeddings/
│   │   └── embedding_generator.py   ← Multilingual embeddings, per keyword
│   ├── clustering/
│   │   ├── kmeans_cluster.py        ← K-Means + auto-K per keyword
│   │   ├── cluster_labeling.py      ← TF-IDF label generation per keyword
│   │   └── clustering_metrics.py    ← Silhouette / CH / DB metrics
│   ├── search/
│   │   ├── semantic_search.py       ← Cluster overview + cosine sub-search
│   │   └── keyword_baseline.py      ← TF-IDF baseline + comparison
│   ├── evaluation/
│   │   ├── search_metrics.py        ← Qualitative comparison report
│   │   └── visualization.py         ← All comparison charts
│   └── utils/
│       ├── config_loader.py         ← load_config(), keyword_paths()
│       └── logger.py                ← Shared logging setup
├── tests/
│   ├── test_preprocessing.py        ← 25 preprocessing unit tests
│   ├── test_clustering_metrics.py   ← 11 clustering metric unit tests
│   └── test_config_loader.py        ← 17 config + path resolver tests
└── data/
    └── processed/
        ├── apple/                   ← All apple artefacts
        ├── milk/                    ← All milk artefacts
        └── ...                      ← One dir per keyword
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

53 tests covering preprocessing, clustering metrics, and the per-keyword
path resolver.

---

## Key Settings (`config/config.yaml`)

| Setting | Value | Notes |
|---|---|---|
| `embeddings.model_name` | `paraphrase-multilingual-mpnet-base-v2` | Handles EN + ZH |
| `keywords.seed_keywords` | 20 keywords | Ambiguous, bilingual |
| `clustering.kmeans.auto_k` | `true` | Sweeps K=2..15 per keyword |
| `clustering.kmeans.k_max` | `15` | Upper bound for ~1 000 products |
| `data_collection.products_per_keyword` | `1000` | ~1 000 products per keyword |
| `search.cluster_label_weight` | `0.05` | Tiny tie-breaker (not dominant) |
