"""
Clustering Metrics
==================
Standalone module for computing and reporting K-Means clustering quality metrics.

Metrics implemented
-------------------
* Silhouette Score        — [-1, 1], higher is better
* Calinski-Harabasz Score — [0, ∞), higher is better
* Davies-Bouldin Score    — [0, ∞), **lower** is better
* Inertia (WCSS)          — within-cluster sum of squares

All metrics are compatible with the outputs of kmeans_cluster.py.

Usage (standalone):
    from src.clustering.clustering_metrics import compute_all_metrics, print_metrics_report
    metrics = compute_all_metrics(embeddings, cluster_labels)
    print_metrics_report(metrics)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/clustering_metrics.log")


# ── Core metric computation ───────────────────────────────────────────── #


def compute_all_metrics(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    sample_size: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute a full suite of clustering quality metrics.

    For large datasets silhouette is estimated on a random sub-sample to
    keep runtime manageable.

    Args:
        embeddings:     (N, D) array of normalised embeddings.
        cluster_labels: (N,) integer array of cluster assignments.
        sample_size:    Max samples used for silhouette calculation.
        random_state:   RNG seed for sub-sampling.

    Returns:
        Dictionary with metric names and values.
    """
    n_samples, n_dims = embeddings.shape
    n_clusters = len(np.unique(cluster_labels))

    metrics: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_dimensions": int(n_dims),
        "n_clusters": int(n_clusters),
    }

    # ── Silhouette ──────────────────────────────────────────────────── #
    sil_n = min(n_samples, sample_size)
    rng = np.random.default_rng(random_state)
    sil_idx = rng.choice(n_samples, size=sil_n, replace=False)

    try:
        sil_global = float(
            silhouette_score(embeddings[sil_idx], cluster_labels[sil_idx])
        )
        sil_per_sample = silhouette_samples(
            embeddings[sil_idx], cluster_labels[sil_idx]
        )

        # Per-cluster mean silhouette
        sil_per_cluster: Dict[int, float] = {}
        for cid in np.unique(cluster_labels[sil_idx]):
            mask = cluster_labels[sil_idx] == cid
            sil_per_cluster[int(cid)] = float(sil_per_sample[mask].mean())

        metrics["silhouette_score"] = sil_global
        metrics["silhouette_per_cluster"] = sil_per_cluster
        metrics["silhouette_sample_size"] = sil_n
        logger.info(f"  Silhouette Score (n={sil_n}): {sil_global:.4f}")
    except Exception as exc:
        logger.warning(f"  Silhouette failed: {exc}")
        metrics["silhouette_score"] = None
        metrics["silhouette_per_cluster"] = {}
        metrics["silhouette_sample_size"] = sil_n

    # ── Calinski-Harabasz ────────────────────────────────────────────── #
    try:
        ch = float(calinski_harabasz_score(embeddings, cluster_labels))
        metrics["calinski_harabasz_score"] = ch
        logger.info(f"  Calinski-Harabasz Score: {ch:.2f}")
    except Exception as exc:
        logger.warning(f"  Calinski-Harabasz failed: {exc}")
        metrics["calinski_harabasz_score"] = None

    # ── Davies-Bouldin ───────────────────────────────────────────────── #
    try:
        db = float(davies_bouldin_score(embeddings, cluster_labels))
        metrics["davies_bouldin_score"] = db
        logger.info(f"  Davies-Bouldin Score: {db:.4f}")
    except Exception as exc:
        logger.warning(f"  Davies-Bouldin failed: {exc}")
        metrics["davies_bouldin_score"] = None

    # ── Cluster size statistics ───────────────────────────────────────── #
    unique, counts = np.unique(cluster_labels, return_counts=True)
    metrics["cluster_sizes"] = {int(c): int(n) for c, n in zip(unique, counts)}
    metrics["cluster_size_stats"] = {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "std": float(counts.std()),
    }

    return metrics


# ── Reporting ─────────────────────────────────────────────────────────── #


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Pretty-print a metrics dictionary to the console."""
    print("\n" + "=" * 55)
    print("  CLUSTERING QUALITY METRICS")
    print("=" * 55)
    print(f"  Samples   : {metrics.get('n_samples', 'N/A'):,}")
    print(f"  Dimensions: {metrics.get('n_dimensions', 'N/A')}")
    print(f"  Clusters  : {metrics.get('n_clusters', 'N/A')}")
    print("-" * 55)

    sil = metrics.get("silhouette_score")
    print(
        f"  Silhouette Score       : {f'{sil:.4f}' if sil is not None else 'N/A':>10}"
        f"  (higher is better, range [-1, 1])"
    )

    ch = metrics.get("calinski_harabasz_score")
    print(
        f"  Calinski-Harabasz      : {f'{ch:.2f}' if ch is not None else 'N/A':>10}"
        f"  (higher is better)"
    )

    db = metrics.get("davies_bouldin_score")
    print(
        f"  Davies-Bouldin         : {f'{db:.4f}' if db is not None else 'N/A':>10}"
        f"  (LOWER is better)"
    )

    inertia = metrics.get("inertia")
    if inertia is not None:
        print(f"  Inertia (WCSS)         : {inertia:>10.1f}  (lower is better)")

    sz = metrics.get("cluster_size_stats", {})
    if sz:
        print("-" * 55)
        print(
            f"  Cluster sizes: min={sz['min']:,}  max={sz['max']:,}  "
            f"mean={sz['mean']:.0f}  std={sz['std']:.0f}"
        )

    print("=" * 55)


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """Persist metrics as a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # numpy types are not JSON-serialisable by default
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=_default)
    logger.info(f"Metrics saved → {output_path}")


# ── Silhouette visualisation ──────────────────────────────────────────── #


def plot_silhouette(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    sample_size: int = 3000,
    output_path: str = "results/figures/silhouette_plot.png",
    random_state: int = 42,
) -> None:
    """
    Draw a silhouette plot showing per-cluster silhouette coefficient
    distributions — useful for identifying poorly-separated clusters.
    """
    n_samples = len(embeddings)
    sil_n = min(n_samples, sample_size)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_samples, size=sil_n, replace=False)

    emb_s = embeddings[idx]
    lab_s = cluster_labels[idx]

    sil_vals = silhouette_samples(emb_s, lab_s)
    global_avg = float(sil_vals.mean())

    unique_clusters = np.unique(lab_s)
    n_clusters = len(unique_clusters)

    fig, ax = plt.subplots(figsize=(10, max(6, n_clusters // 2)))
    y_lower = 10
    cmap = plt.cm.get_cmap("nipy_spectral", n_clusters)
    cluster_color_map = {
        cid: cmap(idx)
        for idx, cid in enumerate(unique_clusters)
    }

    for cid in unique_clusters:
        c_vals = np.sort(sil_vals[lab_s == cid])
        c_size = len(c_vals)
        y_upper = y_lower + c_size

        color = cluster_color_map[cid]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            c_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * c_size, str(cid), fontsize=7)
        y_lower = y_upper + 10

    ax.axvline(
        x=global_avg,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Avg silhouette = {global_avg:.3f}",
    )
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot — K={n_clusters}  (n={sil_n} sample)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Silhouette plot saved → {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse
    from src.utils.config_loader import load_config, keyword_paths

    parser = argparse.ArgumentParser(description="Compute clustering quality metrics")
    parser.add_argument(
        "--keyword", default=None, help="Keyword to compute metrics for (uses config paths)"
    )
    parser.add_argument(
        "--embeddings", default=None, help="Path to product_embeddings.npy"
    )
    parser.add_argument(
        "--clusters_csv", default=None, help="Path to kmeans_clusters.csv"
    )
    parser.add_argument(
        "--output",
        default="results/tables/clustering_metrics.json",
        help="Path to save JSON metrics",
    )
    parser.add_argument(
        "--silhouette_plot",
        default="results/figures/silhouette_plot.png",
        help="Path to save silhouette plot",
    )
    args = parser.parse_args()

    config = load_config()

    if args.keyword:
        # Use keyword-specific paths from config
        paths = keyword_paths(args.keyword, config)
        emb_path = str(paths["embeddings_npy"])
        csv_path = str(paths["clusters_csv"])
    else:
        # Use provided paths or fall back to defaults (if they existed)
        path_cfg = config["paths"]
        emb_path = args.embeddings
        csv_path = args.clusters_csv
        if not emb_path:
            raise ValueError("Must provide --embeddings path or --keyword")
        if not csv_path:
            raise ValueError("Must provide --clusters_csv path or --keyword")

    logger.info(f"Loading embeddings from {emb_path}")
    embs = np.load(emb_path)

    logger.info(f"Loading cluster assignments from {csv_path}")
    df_clusters = pd.read_csv(csv_path)
    labels = df_clusters["cluster"].to_numpy()

    if len(embs) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(embs)} embeddings vs {len(labels)} rows in CSV"
        )

    metrics = compute_all_metrics(embs, labels)
    print_metrics_report(metrics)
    save_metrics(metrics, args.output)
    plot_silhouette(embs, labels, output_path=args.silhouette_plot)
