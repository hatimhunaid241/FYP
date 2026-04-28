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
    query_filter=None,
    title: str = "Cluster Overlap Ratio for All Queries",
) -> None:
    """
    Horizontal bar chart showing cluster overlap per query.
    Lower values indicate stronger semantic vs keyword disagreement.

    Args:
        summary_df: DataFrame with query cluster overlap ratios.
        output_path: File path to save the chart.
        query_filter: Optional callable that accepts a query string and returns True
            if the query should be included.
        title: Chart title.
    """
    if summary_df.empty or "cluster_overlap_ratio" not in summary_df.columns:
        logger.warning("No cluster_overlap_ratio data — skipping chart.")
        return

    df = summary_df[["query", "cluster_overlap_ratio"]].dropna()
    if df.empty:
        logger.warning("No valid overlap rows — skipping chart.")
        return

    if query_filter is not None:
        df = df[df["query"].apply(query_filter)].copy()
        if df.empty:
            logger.warning("No queries match the filter — skipping chart.")
            return

    df = df.sort_values("cluster_overlap_ratio", ascending=False).reset_index(drop=True)
    mean_val = df["cluster_overlap_ratio"].mean()

    fig_height = max(8, len(df) * 0.25)
    fig, ax = plt.subplots(1, 1, figsize=(12, fig_height))

    colors = [
        "#2ecc71" if v >= 0.6 else "#e67e22" if v >= 0.3 else "#e74c3c"
        for v in df["cluster_overlap_ratio"]
    ]

    ax.barh(df["query"], df["cluster_overlap_ratio"], color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.axvline(mean_val, color="navy", linestyle="--", linewidth=1.5)
    ax.text(
        mean_val + 0.01,
        0.02,
        f"Mean = {mean_val:.2f}",
        color="navy",
        va="bottom",
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )

    for i, (query, val) in enumerate(zip(df["query"], df["cluster_overlap_ratio"])):
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
    ax.set_title(title)
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


def _is_two_word_query(query: str) -> bool:
    q = str(query).strip()
    tokens = [token for token in q.split() if token]
    return len(tokens) == 2


def _is_one_word_query(query: str) -> bool:
    q = str(query).strip()
    tokens = [token for token in q.split() if token]
    return len(tokens) == 1


def _is_one_word_keyword_query(query: str, supported_keywords: set) -> bool:
    return str(query).strip().lower() in supported_keywords


def plot_cluster_overlap_by_keyword(
    summary_df: pd.DataFrame,
    output_path: str = "results/figures/cluster_overlap.png",
) -> None:
    """
    Horizontal bar chart showing average cluster overlap per supported keyword.
    """
    if summary_df.empty or "cluster_overlap_ratio" not in summary_df.columns:
        logger.warning("No cluster_overlap_ratio data — skipping keyword overlap chart.")
        return
    if "keyword" not in summary_df.columns:
        logger.warning("No keyword column found — skipping keyword overlap chart.")
        return

    config = load_config("config/config.yaml")
    supported_keywords = {
        kw.strip().lower()
        for kw in config.get("keywords", {}).get("seed_keywords", [])
        if isinstance(kw, str)
    }

    df = summary_df[
        summary_df["keyword"].astype(str).str.lower().isin(supported_keywords)
    ].copy()
    if df.empty:
        logger.warning(
            "No summary rows match the config seed keywords — skipping keyword overlap chart."
        )
        return

    grouped = (
        df.groupby("keyword")["cluster_overlap_ratio"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, max(10, len(grouped) * 0.35)))
    colors = [
        "#2ecc71" if v >= 0.6 else "#e67e22" if v >= 0.3 else "#e74c3c"
        for v in grouped["cluster_overlap_ratio"]
    ]
    ax.barh(grouped["keyword"], grouped["cluster_overlap_ratio"], color=colors, edgecolor="white")
    ax.invert_yaxis()

    mean_val = grouped["cluster_overlap_ratio"].mean()
    ax.axvline(mean_val, color="navy", linestyle="--", linewidth=1.5)
    ax.text(
        mean_val + 0.01,
        0.02,
        f"Mean = {mean_val:.2f}",
        color="navy",
        va="bottom",
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )

    for i, val in enumerate(grouped["cluster_overlap_ratio"]):
        ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=8)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Average Cluster Overlap Ratio")
    ax.set_title("Average Cluster Overlap Ratio by Seed Keyword")
    ax.tick_params(axis="y", labelsize=10)

    fig.suptitle(
        "Semantic vs Keyword: Cluster Overlap Ratio by Supported Keyword",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved keyword-level cluster overlap chart → {output_path}")


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


def plot_low_overlap_query_scores(
    full_df: pd.DataFrame,
    keywords: List[str],
    output_path: str = "results/figures/low_overlap_query_scores.png",
) -> None:
    """
    Plot relevance score distributions for selected low-overlap keywords.
    """
    if full_df.empty or "score" not in full_df.columns or "system" not in full_df.columns:
        logger.warning("Missing data for low-overlap keyword score plot.")
        return

    subset = full_df[full_df["keyword"].isin(keywords)].copy()
    if subset.empty:
        logger.warning(
            "No low-overlap keyword rows found — skipping low-overlap score plot."
        )
        return

    query_order = sorted(subset["query"].dropna().unique().tolist())
    subset["query"] = pd.Categorical(subset["query"], categories=query_order, ordered=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=subset,
        x="query",
        y="score",
        hue="system",
        palette={"semantic": "#3498db", "keyword": "#e67e22"},
        ax=ax,
        width=0.6,
    )

    ax.set_title("Relevance Score Distributions for Low-Overlap Keywords")
    ax.set_xlabel("Query")
    ax.set_ylabel("Relevance Score")
    ax.legend(title="Search System", loc="upper right")
    ax.tick_params(axis="x", rotation=35, labelsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved low-overlap keyword score chart → {output_path}")


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


# ── 6. Semantic cluster distribution (stacked bars) ──────────────────── #


def plot_cluster_distribution_stacked(
    full_df: pd.DataFrame,
    keywords: list = None,
    output_path: str = "results/figures/cluster_distribution_stacked.png",
) -> None:
    """
    Stacked bar chart showing product distribution across semantic clusters
    for each query, with paired bars (keyword vs semantic) per query.
    
    Args:
        full_df:     Full results DataFrame from search_comparison.csv
        keywords:    List of keywords to visualize (default: apple, milk, camera, xiaomi)
        output_path: Path to save the chart
    """
    if keywords is None:
        keywords = ["apple", "milk", "camera", "xiaomi", "samsung"]
    
    if full_df.empty or "keyword" not in full_df.columns or "cluster" not in full_df.columns:
        logger.warning("Missing required columns for cluster distribution plot.")
        return
    
    # Filter for selected keywords
    df = full_df[full_df["keyword"].isin(keywords)].copy()
    if df.empty:
        logger.warning("No data found for selected keywords.")
        return
    
    # Get unique queries per keyword
    kw_queries = df.groupby("keyword")["query"].unique()
    
    # Prepare data: count products per system, query, cluster
    rows = []
    for kw in keywords:
        kw_df = df[df["keyword"] == kw]
        for query in sorted(kw_df["query"].unique()):
            for system in ["semantic", "keyword"]:
                query_system_df = kw_df[
                    (kw_df["query"] == query) & (kw_df["system"] == system)
                ]
                if query_system_df.empty:
                    continue
                
                # Count products per cluster
                cluster_counts = query_system_df["cluster"].value_counts().sort_index()
                for cluster, count in cluster_counts.items():
                    rows.append({
                        "keyword": kw,
                        "query": query,
                        "system": system,
                        "cluster": f"Cluster {int(cluster)}",
                        "count": count
                    })
    
    if not rows:
        logger.warning("No cluster distribution data available.")
        return
    
    plot_data = pd.DataFrame(rows)
    
    # Create figure with subplots (one per keyword)
    fig, axes = plt.subplots(
        len(keywords), 1, figsize=(14, 4 * len(keywords)), sharex=False
    )
    if len(keywords) == 1:
        axes = [axes]
    
    # Unique clusters across all data for consistent coloring
    all_clusters = sorted(plot_data["cluster"].unique())
    colors = sns.color_palette("husl", len(all_clusters))
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(all_clusters)}
    
    for ax, kw in zip(axes, keywords):
        kw_data = plot_data[plot_data["keyword"] == kw]
        
        if kw_data.empty:
            ax.text(0.5, 0.5, f"No data for '{kw}'", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        
        # Get all unique queries for this keyword
        queries = sorted(kw_data["query"].unique())
        
        # Create bar positions: pair of bars (keyword, semantic) per query
        x_pos = []
        labels = []
        pos = 0
        bar_width = 0.35
        
        for query_idx, query in enumerate(queries):
            # Keyword bar
            query_kw = kw_data[(kw_data["query"] == query) & (kw_data["system"] == "keyword")]
            query_sem = kw_data[(kw_data["query"] == query) & (kw_data["system"] == "semantic")]
            
            x_pos.append((pos, pos + bar_width))  # positions for keyword and semantic
            labels.append(query)
            pos += bar_width * 2.5
        
        # Plot stacked bars
        bottom_kw = np.zeros(len(queries))
        bottom_sem = np.zeros(len(queries))
        
        for cluster in all_clusters:
            counts_kw = []
            counts_sem = []
            
            for query in queries:
                query_kw_cluster = kw_data[
                    (kw_data["query"] == query) 
                    & (kw_data["system"] == "keyword")
                    & (kw_data["cluster"] == cluster)
                ]
                query_sem_cluster = kw_data[
                    (kw_data["query"] == query) 
                    & (kw_data["system"] == "semantic")
                    & (kw_data["cluster"] == cluster)
                ]
                counts_kw.append(query_kw_cluster["count"].sum() if not query_kw_cluster.empty else 0)
                counts_sem.append(query_sem_cluster["count"].sum() if not query_sem_cluster.empty else 0)
            
            # Plot keyword bars (left in each pair)
            x_kw = [pos[0] for pos in x_pos]
            ax.bar(x_kw, counts_kw, bar_width, bottom=bottom_kw, 
                  label=cluster, color=cluster_colors[cluster], edgecolor="white")
            bottom_kw += np.array(counts_kw)
            
            # Plot semantic bars (right in each pair)
            x_sem = [pos[1] for pos in x_pos]
            ax.bar(x_sem, counts_sem, bar_width, bottom=bottom_sem,
                  color=cluster_colors[cluster], alpha=0.8, edgecolor="white")
            bottom_sem += np.array(counts_sem)
        
        # Set x-axis labels and positions
        x_labels_pos = [(pos[0] + pos[1]) / 2 for pos in x_pos]
        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Product Count")
        ax.set_title(f"'{kw}' — Product Distribution by Semantic Cluster\n(Left=Keyword, Right=Semantic)")
        ax.grid(axis="y", alpha=0.3)
    
    # Single legend at bottom
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cluster_colors[c]) for c in all_clusters]
    fig.legend(
        handles, all_clusters,
        title="Semantic Clusters",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(5, len(all_clusters)),
        frameon=False,
        fontsize=9
    )
    
    fig.suptitle(
        "Semantic Cluster Distribution: Keyword vs Semantic Search Results",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved cluster distribution chart → {output_path}")


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
        plot_cluster_overlap_by_keyword(summary_df, f"{output_dir}/cluster_overlap.png")

        config = load_config("config/config.yaml")
        supported_keywords = {
            kw.strip().lower()
            for kw in config.get("keywords", {}).get("seed_keywords", [])
            if isinstance(kw, str)
        }

        one_word_filter = lambda q: _is_one_word_keyword_query(q, supported_keywords)
        one_word_label = "Cluster Overlap Ratio for Broad One-Word Supported Keywords"
        if summary_df[summary_df["query"].apply(one_word_filter)].empty:
            logger.warning(
                "No exact one-word supported keyword queries were found in the summary; "
                "falling back to all one-word queries present in the data."
            )
            one_word_filter = _is_one_word_query
            one_word_label = (
                "Cluster Overlap Ratio for One-Word Queries Present "
                "in the Evaluation Data"
            )

        plot_cluster_overlap(
            summary_df,
            f"{output_dir}/cluster_overlap_one_word.png",
            query_filter=one_word_filter,
            title=one_word_label,
        )
        plot_cluster_overlap(
            summary_df,
            f"{output_dir}/cluster_overlap_two_word.png",
            query_filter=_is_two_word_query,
            title="Cluster Overlap Ratio for Specific Two-Word Queries",
        )

        plot_cluster_diversity(summary_df, f"{output_dir}/cluster_diversity.png")

    if not full_df.empty:
        plot_score_distributions(full_df, f"{output_dir}/score_distributions.png")
        plot_low_overlap_query_scores(
            full_df,
            ["xiaomi", "camera", "vitamin", "apple", "mask"],
            f"{output_dir}/low_overlap_query_scores.png",
        )
        plot_price_rating_scatter(full_df, f"{output_dir}/price_rating_scatter.png")
        plot_category_breakdown(
            full_df, output_path=f"{output_dir}/category_breakdown.png"
        )
        plot_cluster_distribution_stacked(
            full_df, keywords=["apple", "milk", "camera", "xiaomi"],
            output_path=f"{output_dir}/cluster_distribution_stacked.png"
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
