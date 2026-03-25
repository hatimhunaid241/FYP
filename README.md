# HKTVmall Semantic Search Project

**Team:** Sutanto, Winiera | Suhandjaja, Alexander Gaudi | Quettawalla, Hatim

## Project Status
- **Current Phase:** Data Collection & EDA
- **Target:** 50,000 products (~5,000 per keyword)
- **Languages:** English (40%) + Chinese (60%)

## Quick Start
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Data collection (run once or schedule)
python src/data_collection/collect_products.py

# 3) Text preprocessing + embeddings
#    Input: data/processed/products.parquet
#    Output: data/processed/products_with_embeddings.parquet + data/processed/product_embeddings.npy
python src/embeddings/embedding_generator.py

# 4) Clustering (K-means on embeddings)
#    Input: data/processed/product_embeddings.npy, data/processed/products.csv
#    Output: data/processed/kmeans_clusters.csv
python src/clustering/kmeans_cluster.py --n_clusters 10

# 5) Cluster labeling (optional but required for semantic filtering/analysis)
python src/clustering/cluster_labeling.py --clusters_path data/processed/kmeans_clusters.csv --output_path data/processed/clusters_labeled.csv

# 5.1) Cluster labeling (English-only labels)
python src/clustering/cluster_labeling.py --clusters_path data/processed/kmeans_clusters.csv --output_path data/processed/clusters_labeled_en.csv --text_columns name_en description_en --language english

# 6) Semantic search
# Interactive mode:
python src/search/semantic_search.py
# Single query mode:
python src/search/semantic_search.py --query "wireless charger" --top_k 10

# 7) Baseline keyword search (term_frequency/contains/exact)
python src/search/keyword_baseline.py --query "wireless charger" --top_k 10 --mode term_frequency
```

## Project Structure
See `docs/PROJECT_STRUCTURE.md` for details.

## Timeline
- ✅ Feb 13: Project Plan Submitted
- ⏳ Mar 6-21: Data Collection + Initial Clustering
- 🎯 Mar 21-Apr 11: Model Evaluation
- 🎯 Apr 11-20: Results Consolidation
- 🎯 Apr 27: Final Presentation
