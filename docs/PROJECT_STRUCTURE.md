# Project Structure

```
FYP/
├── README.md
├── requirements.txt
├── config/
│   ├── config.yaml              # Main configuration
│   └── keywords.yaml            # Search keywords list
│
├── data/
│   ├── raw/                     # Raw API responses
│   │   ├── products/            # Product JSON files
│   │   └── reviews/             # Review JSON files
│   ├── processed/               # Cleaned data
│   │   ├── products.parquet     # All products
│   │   ├── reviews.parquet      # All reviews
│   │   └── metadata.json        # Collection metadata
│   ├── embeddings/              # Vector representations
│   │   ├── product_embeddings.npy
│   │   └── embedding_metadata.json
│   └── database/
│       └── products.db          # SQLite database
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_collection/         # Phase 1: Data Collection
│   │   ├── __init__.py
│   │   ├── api_client.py        # HKTVmall API wrapper
│   │   ├── collect_products.py  # Main collection script
│   │   ├── collect_reviews.py   # Review collection
│   │   └── keyword_expansion.py # Related keyword extraction
│   │
│   ├── preprocessing/           # Phase 2: Text Processing
│   │   ├── __init__.py
│   │   ├── text_cleaner.py      # Cleaning functions
│   │   ├── chinese_processor.py # Chinese text handling
│   │   ├── english_processor.py # English text handling
│   │   └── feature_engineering.py
│   │
│   ├── embeddings/              # Phase 3: Semantic Representation
│   │   ├── __init__.py
│   │   ├── embedding_generator.py
│   │   └── model_manager.py
│   │
│   ├── clustering/              # Phase 4: Semantic Clustering
│   │   ├── __init__.py
│   │   ├── kmeans_cluster.py
│   │   ├── hdbscan_cluster.py
│   │   └── cluster_labeling.py
│   │
│   ├── sentiment/               # Phase 5: Sentiment Analysis
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py
│   │   └── quality_scorer.py
│   │
│   ├── search/                  # Phase 6: Semantic Search
│   │   ├── __init__.py
│   │   ├── semantic_search.py
│   │   └── keyword_baseline.py
│   │
│   ├── evaluation/              # Phase 7: Evaluation
│   │   ├── __init__.py
│   │   ├── clustering_metrics.py
│   │   ├── search_metrics.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── database.py
│       ├── logger.py
│       └── config_loader.py
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_clustering_experiments.ipynb
│   ├── 04_sentiment_analysis.ipynb
│   └── 05_evaluation.ipynb
│
├── experiments/                 # Experiment tracking
│   ├── clustering/
│   │   └── experiment_logs.json
│   └── embeddings/
│       └── model_comparison.json
│
├── results/                     # Final results
│   ├── figures/                 # Plots and visualizations
│   ├── tables/                  # Result tables
│   └── clusters/                # Cluster outputs
│       ├── cluster_assignments.csv
│       └── cluster_labels.json
│
├── tests/                       # Unit tests
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   └── test_clustering.py
│
└── docs/
    ├── PROJECT_STRUCTURE.md     # This file
    ├── API_DOCUMENTATION.md     # HKTVmall API docs
    ├── METHODOLOGY.md           # Detailed methodology
    └── PROGRESS_LOG.md          # Weekly progress tracking
```

## Phase Execution Order

### Phase 1: Data Collection (Mar 6-9)
**Scripts to run:**
1. `src/data_collection/collect_products.py`
2. `src/data_collection/collect_reviews.py`

**Outputs:**
- `data/raw/products/` - JSON files
- `data/processed/products.parquet` - Cleaned dataset

### Phase 2: EDA (Mar 10-12)
**Notebooks:**
1. `notebooks/01_data_exploration.ipynb`

**Key tasks:**
- Language distribution analysis
- Product category distribution
- Missing data analysis
- Text length statistics

### Phase 3: Embeddings (Mar 13-16)
**Scripts:**
1. `src/preprocessing/text_cleaner.py`
2. `src/embeddings/embedding_generator.py`

**Outputs:**
- `data/embeddings/product_embeddings.npy`

### Phase 4: Clustering (Mar 17-21)
**Scripts:**
1. `src/clustering/kmeans_cluster.py`
2. `notebooks/03_clustering_experiments.ipynb`

**Outputs:**
- `results/clusters/cluster_assignments.csv`
- Initial cluster validation results
