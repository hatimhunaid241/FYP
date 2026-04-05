"""
Unit tests for src/clustering/clustering_metrics.py

Run from project root:
    python -m pytest tests/test_clustering_metrics.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.clustering.clustering_metrics import compute_all_metrics, print_metrics_report


def _make_data(
    n_clusters: int = 4, n_per_cluster: int = 50, dim: int = 16, seed: int = 0
):
    """Generate well-separated synthetic clusters."""
    rng = np.random.default_rng(seed)
    embeddings = []
    labels = []
    for cid in range(n_clusters):
        center = rng.uniform(-10, 10, size=dim)
        cluster_emb = center + rng.normal(scale=0.3, size=(n_per_cluster, dim))
        embeddings.append(cluster_emb)
        labels.extend([cid] * n_per_cluster)
    return np.vstack(embeddings).astype(np.float32), np.array(labels)


class TestComputeAllMetrics:
    def setup_method(self):
        self.emb, self.labels = _make_data()

    def test_returns_dict(self):
        result = compute_all_metrics(self.emb, self.labels)
        assert isinstance(result, dict)

    def test_n_clusters_correct(self):
        result = compute_all_metrics(self.emb, self.labels)
        assert result["n_clusters"] == 4

    def test_n_samples_correct(self):
        result = compute_all_metrics(self.emb, self.labels)
        assert result["n_samples"] == 200

    def test_silhouette_is_float(self):
        result = compute_all_metrics(self.emb, self.labels)
        sil = result.get("silhouette_score")
        assert sil is not None
        assert isinstance(sil, float)

    def test_silhouette_high_for_well_separated(self):
        """Well-separated clusters should yield silhouette > 0.5."""
        result = compute_all_metrics(self.emb, self.labels)
        assert result["silhouette_score"] > 0.5, (
            f"Expected high silhouette for well-separated clusters, got {result['silhouette_score']}"
        )

    def test_calinski_harabasz_positive(self):
        result = compute_all_metrics(self.emb, self.labels)
        ch = result.get("calinski_harabasz_score")
        assert ch is not None and ch > 0

    def test_davies_bouldin_positive(self):
        result = compute_all_metrics(self.emb, self.labels)
        db = result.get("davies_bouldin_score")
        assert db is not None and db >= 0

    def test_cluster_sizes_sum_to_n(self):
        result = compute_all_metrics(self.emb, self.labels)
        sizes = result["cluster_sizes"]
        assert sum(sizes.values()) == len(self.emb)

    def test_silhouette_per_cluster_keys(self):
        result = compute_all_metrics(self.emb, self.labels)
        per_cluster = result.get("silhouette_per_cluster", {})
        assert len(per_cluster) == 4

    def test_single_cluster_silhouette_none(self):
        """Silhouette is undefined for k=1 — should return None."""
        emb = np.random.rand(50, 8).astype(np.float32)
        labels = np.zeros(50, dtype=int)
        result = compute_all_metrics(emb, labels)
        assert result["silhouette_score"] is None


class TestPrintMetricsReport:
    """Just test that print_metrics_report doesn't crash."""

    def test_does_not_raise(self, capsys):
        emb, labels = _make_data()
        metrics = compute_all_metrics(emb, labels)
        print_metrics_report(metrics)  # should not raise
        captured = capsys.readouterr()
        assert "Silhouette" in captured.out
