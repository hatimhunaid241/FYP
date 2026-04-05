"""
Evaluation Visualizations
==========================
Generates all comparison charts for the final project report:

  1. Cluster overlap bar chart       — per-query comparison of semantic vs keyword
  2. Score distribution box plots    — semantic scores vs keyword scores
  3. Price / rating by cluster heat-maps
  4. Top-K result category breakdown — where do results come from?

Run from the project root:
    python src/evaluation/visualization.py
    python src/evaluation/visualization.py --summary results/tables/comparison_summary.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings

warnings.filterwarnings("ignore")

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/visualization.log")

# Global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150})


# ── 1. Cluster overlap bar chart ──────────────────────────────────────── #


def plot_cluster_overlap(
    summary_df: pd.DataFrame,
    output_path: str = "results/figures/cluster_overlap.png",
) -> None:
    """
    Bar chart showing the cluster-overlap ratio for each test query.
    Higher = semantic and keyword systems agree more on which product
    categories to return.
    """
    if summary_df.empty or "cluster_overlap_ratio" not in summary_df.columns:
        logger.warning("No cluster_overlap_ratio data — skipping chart.")
        return

    df = summary_df[["query", "cluster_overlap_ratio"]].sort_values(
        "cluster_overlap_ratio", ascending=False
    )

    fig, ax = plt.subplots(figsize=(max(10, len(df) // 2), 6))
    colors = [
        "#2ecc71" if v >= 0.6 else "#e67e22" if v >= 0.3 else "#e74c3c"
        for v in df["cluster_overlap_ratio"]
    ]
    bars = ax.bar(
        df["query"],
        df["cluster_overlap_ratio"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axhline(
        df["cluster_overlap_ratio"].mean(),
        color="navy",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {df['cluster_overlap_ratio'].mean():.2f}",
    )
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Query")
    ax.set_ylabel("Cluster Overlap Ratio")
    ax.set_title(
        "Semantic vs Keyword: Cluster Overlap Ratio per Query\n"
        "(higher = both systems agree on relevant product categories)"
    )
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend()

    # Value labels on bars
    for bar, val in zip(bars, df["cluster_overlap_ratio"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cluster overlap chart → {output_path}")


# ── 2. Score distribution box plots ──────────────────────────────────── #


def plot_score_distributions(
    full_df: pd.DataFrame,
    output_path: str = "results/figures/score_distributions.png",
) -> None:
    """
    Box plot comparing score distributions between semantic and keyword search
    across all queries.
    """
    if (
        full_df.empty
        or "score" not in full_df.columns
        or "system" not in full_df.columns
    ):
        logger.warning("Missing data for score distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall distribution
    sns.boxplot(
        data=full_df,
        x="system",
        y="score",
        palette={"semantic": "#3498db", "keyword": "#e67e22"},
        ax=axes[0],
    )
    axes[0].set_title("Score Distribution: Semantic vs Keyword")
    axes[0].set_xlabel("System")
    axes[0].set_ylabel("Score")

    # Per-query heatmap of average score
    pivot = full_df.groupby(["query", "system"])["score"].mean().unstack("system")
    if not pivot.empty:
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.4, ax=axes[1]
        )
        axes[1].set_title("Average Score by Query and System")
        axes[1].tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score distribution chart → {output_path}")


# ── 3. n_clusters returned per query ─────────────────────────────────── #


def plot_cluster_diversity(
    summary_df: pd.DataFrame,
    output_path: str = "results/figures/cluster_diversity.png",
) -> None:
    """
    Grouped bar chart: how many distinct clusters does each system return per query?
    More clusters = more diverse / broader retrieval.
    """
    needed = {"query", "sem_n_clusters", "kw_n_clusters"}
    if not needed.issubset(summary_df.columns):
        logger.warning("Missing cluster diversity columns — skipping chart.")
        return

    df = summary_df[["query", "sem_n_clusters", "kw_n_clusters"]].copy()
    df = df.melt(id_vars="query", var_name="system", value_name="n_clusters")
    df["system"] = df["system"].map(
        {"sem_n_clusters": "Semantic", "kw_n_clusters": "Keyword"}
    )

    fig, ax = plt.subplots(figsize=(max(10, len(summary_df) // 2), 6))
    sns.barplot(
        data=df,
        x="query",
        y="n_clusters",
        hue="system",
        palette={"Semantic": "#3498db", "Keyword": "#e67e22"},
        ax=ax,
    )
    ax.set_title("Cluster Diversity: Distinct Clusters in Top-K Results")
    ax.set_xlabel("Query")
    ax.set_ylabel("Number of Distinct Clusters")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend(title="System")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cluster diversity chart → {output_path}")


# ── 4. Result price & rating scatter ─────────────────────────────────── #


def plot_price_rating_scatter(
    full_df: pd.DataFrame,
    output_path: str = "results/figures/price_rating_scatter.png",
) -> None:
    """
    Price vs rating scatter for top-K results, coloured by system.
    Helps spot whether one system skews towards expensive or high-rated items.
    """
    needed = {"price", "average_rating", "system"}
    if not needed.issubset(full_df.columns):
        logger.warning("Missing price/rating data — skipping scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {"semantic": "#3498db", "keyword": "#e67e22"}

    for system, group in full_df.groupby("system"):
        sub = group.dropna(subset=["price", "average_rating"])
        sub = sub[sub["price"] > 0]  # exclude zero-price products
        ax.scatter(
            sub["price"],
            sub["average_rating"],
            label=system.capitalize(),
            alpha=0.5,
            s=40,
            color=palette.get(system, "gray"),
        )

    ax.set_xlabel("Price (HK$)")
    ax.set_ylabel("Average Rating")
    ax.set_title("Price vs Rating of Retrieved Products (Semantic vs Keyword)")
    ax.legend(title="System")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved price-rating scatter → {output_path}")


# ── 5. Category breakdown for selected queries ────────────────────────── #


def plot_category_breakdown(
    full_df: pd.DataFrame,
    queries: Optional[List[str]] = None,
    output_path: str = "results/figures/category_breakdown.png",
) -> None:
    """
    For a subset of queries, show what keyword_source categories appear in
    the top-K results for semantic vs keyword search.
    """
    needed = {"query", "system", "keyword_source"}
    if not needed.issubset(full_df.columns):
        logger.warning("Missing keyword_source column — skipping category breakdown.")
        return

    if queries is None:
        # Pick the 6 most interesting queries (most overlap difference)
        queries = full_df["query"].unique()[:6].tolist()

    n_q = len(queries)
    fig, axes = plt.subplots(n_q, 2, figsize=(16, n_q * 3.5))
    if n_q == 1:
        axes = [axes]

    for i, q in enumerate(queries):
        sub = full_df[full_df["query"] == q]
        for j, system in enumerate(["semantic", "keyword"]):
            ax = axes[i][j]
            sys_sub = sub[sub["system"] == system]
            if sys_sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No results",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                counts = sys_sub["keyword_source"].value_counts()
                counts.plot.pie(
                    ax=ax,
                    autopct="%1.0f%%",
                    startangle=90,
                    textprops={"fontsize": 8},
                    legend=False,
                )
            ax.set_title(f"'{q}' — {system.capitalize()}", fontsize=10)
            ax.set_ylabel("")

    plt.suptitle("Product Category Breakdown per Query", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved category breakdown chart → {output_path}")


# ── Main entry point ──────────────────────────────────────────────────── #


def generate_all_visualizations(
    summary_csv: str = "results/tables/comparison_summary.csv",
    full_csv: str = "results/tables/search_comparison.csv",
    output_dir: str = "results/figures",
) -> None:
    """Load saved evaluation CSVs and regenerate all charts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame()
    full_df = pd.DataFrame()

    if Path(summary_csv).exists() and Path(summary_csv).stat().st_size > 5:
        try:
            summary_df = pd.read_csv(summary_csv)
            logger.info(f"Loaded summary: {len(summary_df)} rows")
        except Exception as exc:
            logger.warning(f"Could not read '{summary_csv}': {exc}")
    else:
        logger.warning(
            f"Summary CSV not found or empty at '{summary_csv}'. "
            "Run src/evaluation/search_metrics.py first."
        )

    if Path(full_csv).exists() and Path(full_csv).stat().st_size > 5:
        try:
            full_df = pd.read_csv(full_csv)
            logger.info(f"Loaded full results: {len(full_df)} rows")
        except Exception as exc:
            logger.warning(f"Could not read '{full_csv}': {exc}")
    else:
        logger.warning(
            f"Full results CSV not found or empty at '{full_csv}'. "
            "Run src/evaluation/search_metrics.py first."
        )

    if not summary_df.empty:
        plot_cluster_overlap(summary_df, f"{output_dir}/cluster_overlap.png")
        plot_cluster_diversity(summary_df, f"{output_dir}/cluster_diversity.png")

    if not full_df.empty:
        plot_score_distributions(full_df, f"{output_dir}/score_distributions.png")
        plot_price_rating_scatter(full_df, f"{output_dir}/price_rating_scatter.png")
        plot_category_breakdown(
            full_df, output_path=f"{output_dir}/category_breakdown.png"
        )

    logger.info(f"All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument("--summary", default="results/tables/comparison_summary.csv")
    parser.add_argument("--full", default="results/tables/search_comparison.csv")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    generate_all_visualizations(args.summary, args.full, args.output_dir)
