"""
Configuration Loader Utility
==============================
Provides:
  load_config()     — load config/config.yaml as a dict
  keyword_paths()   — resolve all per-keyword file paths for one keyword
  keyword_dir()     — return the Path for a keyword's data directory
"""

import re
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file and return as a dictionary."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def keyword_slug(keyword: str) -> str:
    """
    Convert a keyword to a filesystem-safe slug.

    Examples:
        "apple"          → "apple"
        "wireless charger" → "wireless_charger"
        "USB 3.0"        → "usb_3_0"
    """
    slug = keyword.strip().lower()
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", slug)  # keep CJK, alphanum, underscore
    slug = slug.strip("_")
    return slug


def keyword_dir(keyword: str, config: Dict[str, Any]) -> Path:
    """Return the data directory for a given keyword (created if needed)."""
    processed_dir = Path(config["paths"]["processed_dir"])
    kw_dir = processed_dir / keyword_slug(keyword)
    kw_dir.mkdir(parents=True, exist_ok=True)
    return kw_dir


def keyword_paths(keyword: str, config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Resolve all per-keyword file paths for one keyword.

    Returns a dict with the following keys (all as Path objects):
        dir                  — data/processed/<slug>/
        products_parquet     — .../products.parquet
        embeddings_npy       — .../product_embeddings.npy
        embeddings_parquet   — .../products_with_embeddings.parquet
        clusters_csv         — .../kmeans_clusters.csv
        clusters_labeled_csv — .../clusters_labeled.csv
        cluster_labels_csv   — .../cluster_labels.csv
        elbow_plot           — .../kmeans_elbow.png
    """
    kw_dir = keyword_dir(keyword, config)
    p = config["paths"]

    return {
        "dir": kw_dir,
        "products_parquet": kw_dir / p["kw_products_parquet"],
        "embeddings_npy": kw_dir / p["kw_embeddings_npy"],
        "embeddings_parquet": kw_dir / p["kw_embeddings_parquet"],
        "clusters_csv": kw_dir / p["kw_clusters_csv"],
        "clusters_labeled_csv": kw_dir / p["kw_clusters_labeled_csv"],
        "cluster_labels_csv": kw_dir / p["kw_cluster_labels_csv"],
        "elbow_plot": kw_dir / p["kw_elbow_plot"],
    }
