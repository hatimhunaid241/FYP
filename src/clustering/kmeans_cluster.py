"""
Per-Keyword K-Means Clustering
================================
For each keyword, runs K-Means on that keyword's embedding matrix and writes:

    data/processed/<keyword>/kmeans_clusters.csv   — products + cluster column
    data/processed/<keyword>/kmeans_elbow.png       — elbow + silhouette chart

Optimal K is selected per keyword via an elbow + silhouette sweep.
Because each keyword has only ~1 000 products the sweep is fast and
k_max is capped at 15 (sensible upper bound for ~1 000 items).

Run from the project root:
    python src/clustering/kmeans_cluster.py
    python src/clustering/kmeans_cluster.py --keywords apple milk
    python src/clustering/kmeans_cluster.py --n_clusters 5 --keywords fan
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings

warnings.filterwarnings("ignore")

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/kmeans_cluster.log")


def _dominant_category(series: pd.Series) -> str:
    counts = series.value_counts()
    return str(counts.index[0]) if len(counts) > 0 else "unknown"


# ── Optimal-K sweep ───────────────────────────────────────────────────── #


def find_optimal_k(
    embeddings: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    elbow_plot_path: Path,
    keyword: str = "",
) -> tuple[int, dict]:
    """Sweep K-Means over k_min..k_max; return (best_k, results_dict)."""
    n = len(embeddings)
    # With ~1 000 products silhouette on the full set is fast
    sil_sample = min(n, 1000)
    rng = np.random.default_rng(random_state)
    sil_idx = rng.choice(n, size=sil_sample, replace=False)

    results: Dict[int, Dict[str, Any]] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        try:
            sil = float(silhouette_score(embeddings[sil_idx], labels[sil_idx]))
        except Exception:
            sil = None
        try:
            ch = float(calinski_harabasz_score(embeddings, labels))
            db = float(davies_bouldin_score(embeddings, labels))
        except Exception:
            ch = db = None
        results[k] = {
            "inertia": float(km.inertia_),
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
        }

    valid = {k: v for k, v in results.items() if v["silhouette_score"] is not None}
    best_k = max(valid, key=lambda k: valid[k]["silhouette_score"]) if valid else k_min
    logger.info(
        f"  [{keyword}] Best K={best_k} "
        f"(silhouette={results[best_k]['silhouette_score']:.4f})"
    )

    # Elbow plot
    ks = list(results.keys())
    inertias = [results[k]["inertia"] for k in ks]
    sils = [results[k]["silhouette_score"] for k in ks]

    elbow_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inertias, "bo-", ms=5)
    ax1.axvline(best_k, color="red", ls="--", label=f"Best K={best_k}")
    ax1.set(xlabel="K", ylabel="Inertia", title=f"'{keyword}' — Elbow")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    valid_ks = [k for k in ks if results[k]["silhouette_score"] is not None]
    valid_sils = [results[k]["silhouette_score"] for k in valid_ks]
    ax2.plot(valid_ks, valid_sils, "ro-", ms=5)
    ax2.axvline(best_k, color="blue", ls="--", label=f"Best K={best_k}")
    ax2.set(xlabel="K", ylabel="Silhouette", title=f"'{keyword}' — Silhouette")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(elbow_plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"  [{keyword}] Elbow plot → {elbow_plot_path}")

    return best_k, results


# ── Per-keyword clustering ────────────────────────────────────────────── #


def cluster_keyword(
    keyword: str,
    config: dict,
    n_clusters: Optional[int] = None,
    visualize: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run K-Means on one keyword's embeddings and save results.

    Returns (clustered_df, metrics_dict).
    """
    paths = keyword_paths(keyword, config)
    km_cfg = config["clustering"]["kmeans"]
    random_state = km_cfg.get("random_state", 42)
    auto_k = km_cfg.get("auto_k", True)
    k_min = km_cfg.get("k_min", 2)
    k_max = km_cfg.get("k_max", 15)

    if not paths["embeddings_npy"].exists():
        raise FileNotFoundError(
            f"Embeddings not found for '{keyword}' at {paths['embeddings_npy']}. "
            "Run embedding_generator.py first."
        )
    if not paths["embeddings_parquet"].exists():
        raise FileNotFoundError(
            f"Product parquet not found for '{keyword}' at {paths['embeddings_parquet']}."
        )

    embeddings = np.load(paths["embeddings_npy"])
    products = pd.read_parquet(paths["embeddings_parquet"])

    if len(embeddings) != len(products):
        raise ValueError(
            f"[{keyword}] Length mismatch: {len(embeddings)} embeddings "
            f"vs {len(products)} products."
        )

    logger.info(
        f"  [{keyword}] {len(products):,} products, embedding dim={embeddings.shape[1]}"
    )

    # Determine K
    if n_clusters is not None:
        best_k = n_clusters
        sweep_results: dict = {}
    elif auto_k:
        best_k, sweep_results = find_optimal_k(
            embeddings,
            k_min,
            k_max,
            random_state,
            elbow_plot_path=paths["elbow_plot"],
            keyword=keyword,
        )
    else:
        best_k = km_cfg.get("n_clusters", 5)
        sweep_results = {}

    # Fit final K-Means
    km = KMeans(
        n_clusters=best_k,
        random_state=random_state,
        n_init=km_cfg.get("n_init", 10),
        max_iter=km_cfg.get("max_iter", 300),
    )
    cluster_labels_arr = km.fit_predict(embeddings)

    # Metrics
    sil_sample = min(len(embeddings), 1000)
    rng = np.random.default_rng(random_state)
    sil_idx = rng.choice(len(embeddings), sil_sample, replace=False)
    try:
        silhouette = float(
            silhouette_score(embeddings[sil_idx], cluster_labels_arr[sil_idx])
        )
    except Exception:
        silhouette = None
    try:
        ch_score = float(calinski_harabasz_score(embeddings, cluster_labels_arr))
        db_score = float(davies_bouldin_score(embeddings, cluster_labels_arr))
    except Exception:
        ch_score = db_score = None

    sil_str = f"{silhouette:.4f}" if silhouette is not None else "N/A"
    logger.info(
        f"  [{keyword}] K={best_k}  silhouette={sil_str}  inertia={km.inertia_:.1f}"
    )

    # Build results DataFrame
    results_df = products.copy()
    results_df["cluster"] = cluster_labels_arr.astype(int)

    # Cluster stats (safe: no lambda)
    agg: Dict = {"cluster": ["count"]}
    if "price" in results_df.columns:
        agg["price"] = ["mean"]
    if "average_rating" in results_df.columns:
        agg["average_rating"] = ["mean"]

    metrics: Dict[str, Any] = {
        "keyword": keyword,
        "n_clusters": best_k,
        "n_products": len(results_df),
        "silhouette_score": silhouette,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score,
        "inertia": float(km.inertia_),
        "cluster_sizes": results_df["cluster"].value_counts().sort_index().to_dict(),
    }

    # Save
    results_df.to_csv(paths["clusters_csv"], index=False, encoding="utf-8-sig")
    logger.info(f"  [{keyword}] Saved → {paths['clusters_csv']}")

    if visualize:
        _visualize_keyword_clusters(keyword, results_df, best_k, config)

    return results_df, metrics


def _visualize_keyword_clusters(
    keyword: str, df: pd.DataFrame, k: int, config: dict
) -> None:
    fig_dir = Path(config["paths"]["results_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(max(8, k), 4))
    counts = df["cluster"].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
    plt.title(f"'{keyword}' — Cluster Size Distribution (K={k})")
    plt.xlabel("Cluster ID")
    plt.ylabel("Products")
    plt.tight_layout()
    plt.savefig(fig_dir / f"cluster_sizes_{keyword}.png", dpi=130, bbox_inches="tight")
    plt.close()


# ── Run all keywords ──────────────────────────────────────────────────── #


def cluster_all_keywords(
    keywords: list[str] | None = None,
    n_clusters: Optional[int] = None,
    config_path: str = "config/config.yaml",
    force: bool = False,
    visualize: bool = True,
) -> Dict[str, dict]:
    """Cluster every keyword independently. Returns {keyword: metrics} dict."""
    config = load_config(config_path)
    if keywords is None:
        keywords = config["keywords"]["seed_keywords"]

    logger.info("=" * 60)
    logger.info(f"Per-keyword clustering: {len(keywords)} keywords")
    logger.info("=" * 60)

    all_metrics: Dict[str, dict] = {}
    for kw in keywords:
        paths = keyword_paths(kw, config)
        if not force and paths["clusters_csv"].exists():
            logger.info(f"  '{kw}' clusters already exist — skipping")
            continue
        try:
            _, metrics = cluster_keyword(
                kw, config, n_clusters=n_clusters, visualize=visualize
            )
            all_metrics[kw] = metrics
        except Exception as exc:
            logger.error(f"  '{kw}' failed: {exc}")

    logger.info("Clustering complete.")
    return all_metrics


# ── CLI ───────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-keyword K-Means clustering")
    parser.add_argument("--keywords", nargs="+", default=None)
    parser.add_argument(
        "--n_clusters", type=int, default=None, help="Fixed K (skips auto-K sweep)"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    args = parser.parse_args()

    cluster_all_keywords(
        keywords=args.keywords,
        n_clusters=args.n_clusters,
        force=args.force,
        visualize=not args.no_visualize,
    )
