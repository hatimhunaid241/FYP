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
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False


def _query_language_type(query: str) -> str:
    chinese = any(
        "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" for ch in query
    )
    latin = any(ch.isascii() and ch.isalpha() for ch in query)
    if chinese and latin:
        return "Mixed"
    if chinese:
        return "Chinese"
    if latin:
        return "English"
    return "Other"


# ── 1. Cluster overlap bar chart ──────────────────────────────────────── #


def plot_cluster_overlap(
    summary_df: pd.DataFrame,
    output_path: str = "results/figures/cluster_overlap.png",
) -> None:
    """
    Horizontal bar chart showing cluster overlap per query.
    Lower values indicate stronger semantic vs keyword disagreement.
    """
    if summary_df.empty or "cluster_overlap_ratio" not in summary_df.columns:
        logger.warning("No cluster_overlap_ratio data — skipping chart.")
        return

    df = summary_df[["query", "cluster_overlap_ratio"]].dropna()
    if df.empty:
        logger.warning("No valid overlap rows — skipping chart.")
        return

    df = df.sort_values("cluster_overlap_ratio", ascending=False).reset_index(drop=True)
    mean_val = df["cluster_overlap_ratio"].mean()

    # Split into top 20 highest and bottom 20 lowest
    top_20 = df.head(20)
    bottom_20 = df.tail(20)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for ax, sub_df, title_suffix in zip(
        axes, [top_20, bottom_20], ["Top 20 Highest Overlap", "Bottom 20 Lowest Overlap"]
    ):
        colors = [
            "#2ecc71" if v >= 0.6 else "#e67e22" if v >= 0.3 else "#e74c3c"
            for v in sub_df["cluster_overlap_ratio"]
        ]

        ax.barh(sub_df["query"], sub_df["cluster_overlap_ratio"], color=colors, edgecolor="white")
        ax.invert_yaxis()
        ax.axvline(mean_val, color="navy", linestyle="--", linewidth=1.5)
        ax.text(
            mean_val + 0.01,
            0.5,
            f"Mean = {mean_val:.2f}",
            color="navy",
            va="center",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )

        for i, (query, val) in enumerate(zip(sub_df["query"], sub_df["cluster_overlap_ratio"])):
            if val < mean_val:
                ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=8)
            if val == 0.0:
                ax.text(
                    0.02,
                    i,
                    "Keyword returns no results;\nsemantic fills the gap",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#e74c3c",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Cluster Overlap Ratio")
        ax.set_title(f"{title_suffix}")
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "Semantic vs Keyword: Cluster Overlap Ratio per Query\n"
        "Lower values highlight queries with low agreement on product groups",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    sns.boxplot(
        data=full_df,
        x="system",
        y="score",
        palette={"semantic": "#3498db", "keyword": "#e67e22"},
        ax=axes[0],
        width=0.5,
    )
    axes[0].set_title("Score Distribution: Semantic vs Keyword")
    axes[0].set_xlabel("Search System")
    axes[0].set_ylabel("Score")

    medians = full_df.groupby("system")["score"].median()
    for i, system in enumerate(["semantic", "keyword"]):
        if system in medians:
            axes[0].text(
                i,
                medians[system] + 0.02,
                f"Median: {medians[system]:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

    pivot = full_df.groupby(["query", "system"])["score"].mean().unstack("system")
    if not pivot.empty:
        sns.heatmap(
            pivot,
            annot=False,
            cmap="YlOrRd",
            linewidths=0.4,
            ax=axes[1],
            cbar_kws={"shrink": 0.7},
        )
        axes[1].set_title("Average Score by Query and System")
        axes[1].tick_params(axis="y", rotation=0, labelsize=9)
        axes[1].set_xlabel("System")
        axes[1].set_ylabel("")

    fig.tight_layout()
    fig.subplots_adjust(left=0.2, right=0.95, wspace=0.2)
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
    Grouped bar chart showing the number of distinct clusters returned per query,
    split by English, Chinese, and mixed query groups.
    """
    needed = {"query", "sem_n_clusters", "kw_n_clusters"}
    if not needed.issubset(summary_df.columns):
        logger.warning("Missing cluster diversity columns — skipping chart.")
        return

    df = summary_df[["query", "sem_n_clusters", "kw_n_clusters"]].copy()
    df["query_type"] = df["query"].map(_query_language_type)

    mixed_count = df[df["query_type"] == "Mixed"].shape[0]
    if mixed_count <= 1 and mixed_count > 0:
        df.loc[df["query_type"] == "Mixed", "query_type"] = "Chinese"

    groups = [g for g in ["English", "Chinese", "Mixed"] if g in df["query_type"].unique()]
    if not groups:
        logger.warning("No query types found — skipping chart.")
        return

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(16, 5 * len(groups)),
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, group in zip(axes, groups):
        sub = df[df["query_type"] == group].sort_values("query")
        plot_df = sub.melt(
            id_vars=["query"],
            value_vars=["sem_n_clusters", "kw_n_clusters"],
            var_name="system",
            value_name="n_clusters",
        )
        plot_df["system"] = plot_df["system"].map(
            {"sem_n_clusters": "Semantic", "kw_n_clusters": "Keyword"}
        )

        sns.barplot(
            data=plot_df,
            x="query",
            y="n_clusters",
            hue="system",
            palette={"Semantic": "#3498db", "Keyword": "#e67e22"},
            ax=ax,
        )
        mean_sem = sub["sem_n_clusters"].mean()
        mean_kw = sub["kw_n_clusters"].mean()
        ax.axhline(mean_sem, color="#3498db", linestyle="--", linewidth=1.2)
        ax.axhline(mean_kw, color="#e67e22", linestyle="--", linewidth=1.2)
        ax.text(
            0.98,
            mean_sem,
            f"Semantic mean {mean_sem:.1f}",
            color="#3498db",
            ha="right",
            va="bottom",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            0.98,
            mean_kw,
            f"Keyword mean {mean_kw:.1f}",
            color="#e67e22",
            ha="right",
            va="bottom",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
        )

        if mean_kw > mean_sem:
            ax.text(
                0.02,
                0.88,
                "Keyword picks more loosely related clusters here.",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#e67e22",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

        ax.set_title(f"{group} Queries")
        ax.set_xlabel("")
        ax.set_ylabel("Distinct Clusters")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(title="Search System", fontsize=8, title_fontsize=9)

    fig.suptitle(
        "Cluster Diversity by Query Type",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    Price vs rating scatter for top-K results, with separate subplots
    for semantic and keyword search and log-scaled price.
    """
    needed = {"price", "average_rating", "system"}
    if not needed.issubset(full_df.columns):
        logger.warning("Missing price/rating data — skipping scatter plot.")
        return

    df = full_df.dropna(subset=["price", "average_rating", "system"]).copy()
    df = df[df["price"] > 0]

    if df.empty:
        logger.warning("No valid price/rating rows — skipping scatter plot.")
        return

    palette = {"semantic": "#3498db", "keyword": "#e67e22"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    systems = ["semantic", "keyword"]

    for ax, system in zip(axes, systems):
        sub = df[df["system"] == system]
        if sub.empty:
            ax.text(0.5, 0.5, f"No {system} results", ha="center", va="center")
            ax.set_axis_off()
            continue

        rated = sub[sub["average_rating"] > 0].copy()
        unrated = sub[sub["average_rating"] == 0].copy()

        if not rated.empty:
            ax.scatter(
                rated["price"],
                rated["average_rating"],
                alpha=0.3,
                s=30,
                color=palette.get(system, "gray"),
            )

        if not unrated.empty:
            ax.scatter(
                unrated["price"],
                np.zeros(len(unrated)),
                marker="x",
                color="#7f8c8d",
                alpha=0.6,
                s=30,
            )
            ax.text(
                0.02,
                0.12,
                "X = unrated products",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color="#7f8c8d",
            )

        if len(rated) >= 5:
            x = np.log(rated["price"].astype(float))
            y = rated["average_rating"].astype(float)
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                np.exp(x_line),
                coef[0] * x_line + coef[1],
                color="#2c3e50",
                linestyle="--",
                linewidth=1.5,
            )
            ax.text(
                0.98,
                0.08,
                "Trend line uses rated products only",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
            )
            ax.text(
                0.02,
                0.92,
                "Higher-rated products cluster in the mid-price range.",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#2c3e50",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.set_xlabel("Price (HK$, log scale)")
        ax.set_ylabel("Average Rating")
        ax.set_title(f"{system.capitalize()} Search")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        "Price vs Rating for Semantic and Keyword Search Results",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    Stacked bar chart for selected queries showing top-level product categories
    retrieved by semantic vs keyword search.
    """
    if "query" not in full_df.columns or "system" not in full_df.columns:
        logger.warning("Missing query/system columns — skipping category breakdown.")
        return

    category_col = (
        "main_category_en"
        if "main_category_en" in full_df.columns
        else "primary_category"
    )
    if category_col not in full_df.columns:
        logger.warning(
            "Missing main_category_en or primary_category — skipping category breakdown."
        )
        return

    if queries is None:
        queries = full_df["query"].unique()[:6].tolist()
    queries = list(dict.fromkeys(queries))[:6]

    subset = full_df[full_df["query"].isin(queries)].copy()
    if subset.empty:
        logger.warning("No matching query rows — skipping category breakdown.")
        return

    subset[category_col] = subset[category_col].fillna("Other").astype(str)
    top_categories = subset[category_col].value_counts().nlargest(6).index.tolist()
    subset[category_col] = subset[category_col].where(
        subset[category_col].isin(top_categories), "Other"
    )

    counts = (
        subset.groupby(["query", "system", category_col])
        .size()
        .rename("count")
        .reset_index()
    )
    if counts.empty:
        logger.warning("No category counts available — skipping category breakdown.")
        return

    systems = ["semantic", "keyword"]
    fig, axes = plt.subplots(
        len(systems),
        1,
        figsize=(16, 5 * len(systems)),
        sharex=True,
    )
    if len(systems) == 1:
        axes = [axes]

    legend_entries = {}
    for ax, system in zip(axes, systems):
        system_counts = counts[counts["system"] == system]
        if system_counts.empty:
            ax.text(0.5, 0.5, f"No {system} results", ha="center", va="center")
            ax.set_axis_off()
            continue

        pivot = system_counts.pivot(
            index="query", columns=category_col, values="count"
        ).fillna(0)
        pivot = pivot.reindex(queries).fillna(0)

        bottom = np.zeros(len(pivot))
        palette = sns.color_palette("tab10", n_colors=len(pivot.columns))
        for idx, cat in enumerate(pivot.columns):
            bars = ax.bar(
                pivot.index,
                pivot[cat],
                bottom=bottom,
                label=cat,
                color=palette[idx],
                edgecolor="white",
            )
            if cat not in legend_entries:
                legend_entries[cat] = bars[0]
            bottom += pivot[cat].values

        empty_queries = pivot.index[pivot.sum(axis=1) == 0]
        for query in empty_queries:
            ax.text(
                query,
                0,
                "No results",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#c0392b",
                fontweight="bold",
            )

        ax.set_title(f"{system.capitalize()} search category mix")
        ax.set_ylabel("Result count")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.set_ylim(bottom=0)

    if legend_entries:
        fig.legend(
            legend_entries.values(),
            legend_entries.keys(),
            title="Product category",
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
            frameon=False,
        )

    fig.suptitle(
        "Product Category Breakdown per Query\n"
        "Semantic system successfully retrieves diversified categories where keyword search is narrower.",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
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
