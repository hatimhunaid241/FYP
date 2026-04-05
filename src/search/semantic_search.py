"""
Per-Keyword Semantic Search with Cluster Navigation
=====================================================
User experience:
  1. User types a keyword (e.g. "apple")
  2. System loads the pre-built clusters for that keyword
  3. Results are shown grouped by cluster label:
       Cluster 0 — Apple iPhone | 蘋果 | iphone  (312 products)
       Cluster 1 — Apple Pencil | 手寫筆          (87 products)
       Cluster 2 — Apple Fruit | 蘋果 | 水果      (45 products)
       Cluster 3 — Apple Watch | 手錶             (156 products)
  4. User picks a cluster (e.g. "0") and can optionally type a sub-query
     (e.g. "iPhone 15 Pro Max") to rank products within that cluster
     by cosine similarity.

Run from the project root:
    python src/search/semantic_search.py
    python src/search/semantic_search.py --keyword apple
    python src/search/semantic_search.py --keyword apple --cluster 0 --query "iPhone 15"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/semantic_search.log")


# ── Data loading ──────────────────────────────────────────────────────── #


def load_keyword_data(
    keyword: str,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Load the labeled cluster CSV and embedding matrix for one keyword.

    Returns:
        {
          'df':         DataFrame with all product columns + cluster + cluster_label,
          'embeddings': (N, D) float32 numpy array (L2-normalised),
          'keyword':    the keyword string,
          'config':     full config dict,
        }
    """
    config = load_config(config_path)
    paths = keyword_paths(keyword, config)

    if not paths["clusters_labeled_csv"].exists():
        raise FileNotFoundError(
            f"Labeled clusters not found for '{keyword}' at "
            f"{paths['clusters_labeled_csv']}.\n"
            "Run the full pipeline first:\n"
            "  python run_pipeline.py --keywords " + keyword
        )
    if not paths["embeddings_npy"].exists():
        raise FileNotFoundError(
            f"Embeddings not found for '{keyword}' at {paths['embeddings_npy']}."
        )

    df = pd.read_csv(paths["clusters_labeled_csv"], low_memory=False)
    embeddings = np.load(paths["embeddings_npy"])

    if len(df) != len(embeddings):
        raise ValueError(
            f"[{keyword}] Row mismatch: {len(df)} CSV rows vs "
            f"{len(embeddings)} embeddings. Re-run the pipeline."
        )

    embeddings_normed = normalize(embeddings, norm="l2", axis=1).astype(np.float32)
    logger.info(
        f"Loaded '{keyword}': {len(df):,} products, {df['cluster'].nunique()} clusters"
    )

    return {
        "df": df,
        "embeddings": embeddings_normed,
        "keyword": keyword,
        "config": config,
    }


def build_model(
    model_name: str | None = None,
    config_path: str = "config/config.yaml",
) -> SentenceTransformer:
    config = load_config(config_path)
    model_name = model_name or config["embeddings"]["model_name"]
    logger.info(f"Loading model: {model_name}")
    return SentenceTransformer(model_name)


# ── Cluster overview ──────────────────────────────────────────────────── #


def get_cluster_overview(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Return a summary DataFrame with one row per cluster:
        cluster | cluster_label | n_products | avg_price | avg_rating
    """
    df = data["df"]
    agg: Dict = {"cluster": "count"}
    if "price" in df.columns:
        agg["price"] = "mean"
    if "average_rating" in df.columns:
        agg["average_rating"] = "mean"

    overview = (
        df.groupby("cluster")
        .agg(
            n_products=("cluster", "count"),
            **{
                "avg_price": pd.NamedAgg("price", "mean")
                if "price" in df.columns
                else pd.NamedAgg("cluster", "count"),  # fallback — dropped below
                "avg_rating": pd.NamedAgg("average_rating", "mean")
                if "average_rating" in df.columns
                else pd.NamedAgg("cluster", "count"),
            },
        )
        .reset_index()
    )

    # Attach label
    label_map = df.groupby("cluster")["cluster_label"].first()
    overview["cluster_label"] = overview["cluster"].map(label_map)

    # Drop fallback columns if price/rating not available
    cols = ["cluster", "cluster_label", "n_products"]
    if "price" in df.columns:
        overview["avg_price"] = overview["avg_price"].round(2)
        cols.append("avg_price")
    if "average_rating" in df.columns:
        overview["avg_rating"] = overview["avg_rating"].round(3)
        cols.append("avg_rating")

    return overview[cols].sort_values("cluster")


# ── Sub-search within a cluster ───────────────────────────────────────── #


def search_within_cluster(
    query: str,
    cluster_id: int,
    data: Dict[str, Any],
    model: SentenceTransformer,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Encode *query* and rank all products in *cluster_id* by cosine similarity.

    Returns top_k products with a 'score' column, sorted descending.
    """
    df = data["df"]
    embeddings = data["embeddings"]
    cfg_search = data["config"].get("search", {})

    # Filter to the requested cluster
    mask = df["cluster"] == cluster_id
    if not mask.any():
        raise ValueError(
            f"Cluster {cluster_id} not found for keyword '{data['keyword']}'."
        )

    cluster_df = df[mask].copy().reset_index(drop=True)
    cluster_emb = embeddings[df.index[mask].to_numpy()]

    # Encode query
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores = (cluster_emb @ q_vec.T).squeeze()

    # Optional tiny cluster-label boost (tie-breaker only)
    if (
        cfg_search.get("cluster_label_boost", True)
        and "cluster_label" in cluster_df.columns
    ):
        weight = float(cfg_search.get("cluster_label_weight", 0.05))
        q_lower = query.strip().lower()
        boost = np.array(
            [
                weight if q_lower in str(lbl).lower() else 0.0
                for lbl in cluster_df["cluster_label"].fillna("")
            ],
            dtype=np.float32,
        )
        scores = scores + boost

    cluster_df["score"] = scores
    return (
        cluster_df.sort_values("score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


# ── Interactive session ───────────────────────────────────────────────── #


def interactive_search(model: SentenceTransformer | None = None) -> None:
    """
    Full interactive CLI session:
      1. User types a keyword → sees cluster overview
      2. User picks a cluster number → sees top products
      3. Optionally types a sub-query → re-ranks products in that cluster
    """
    config = load_config()
    if model is None:
        model = build_model()

    known_keywords = config["keywords"]["seed_keywords"]

    display_cols = [
        "cluster",
        "cluster_label",
        "name_en",
        "name_zh",
        "brand",
        "price",
        "average_rating",
        "score",
    ]

    print("\n" + "=" * 65)
    print("  HKTVmall Semantic Search — Cluster Navigation")
    print("=" * 65)
    print("  Supported keywords:", ", ".join(known_keywords))
    print("  Type 'quit' at any prompt to exit.")
    print("=" * 65)

    while True:
        # ── Step 1: keyword ─────────────────────────────────────────── #
        try:
            keyword = input("\nKeyword to search: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if keyword in {"quit", "exit", "q", ""}:
            print("Goodbye.")
            break

        try:
            data = load_keyword_data(keyword)
        except FileNotFoundError as exc:
            print(f"\n  Error: {exc}")
            continue

        # ── Step 2: show cluster overview ─────────────────────────────
        overview = get_cluster_overview(data)
        print(f"\n  Clusters for '{keyword}' ({len(data['df']):,} products total):\n")
        print(overview.to_string(index=False))

        # ── Step 3: choose a cluster ───────────────────────────────── #
        try:
            choice = input("\nPick a cluster ID (or Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "":
            continue
        if choice.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            cluster_id = int(choice)
        except ValueError:
            print("  Please enter a valid integer cluster ID.")
            continue

        if cluster_id not in data["df"]["cluster"].values:
            print(f"  Cluster {cluster_id} does not exist for '{keyword}'.")
            continue

        # ── Step 4: optional sub-query ────────────────────────────── #
        cluster_size = (data["df"]["cluster"] == cluster_id).sum()
        label = data["df"][data["df"]["cluster"] == cluster_id]["cluster_label"].iloc[0]
        print(f"\n  Cluster {cluster_id}: '{label}'  ({cluster_size:,} products)")

        try:
            sub_query = input(
                "  Sub-query to rank within cluster (or Enter to show top products): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if sub_query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        # If no sub-query, use the keyword itself as the query
        effective_query = sub_query if sub_query else keyword

        top_k = data["config"].get("search", {}).get("top_k_results", 10)
        results = search_within_cluster(
            effective_query, cluster_id, data, model, top_k=top_k
        )

        avail = [c for c in display_cols if c in results.columns]
        print(
            f"\n  Top {len(results)} results in cluster {cluster_id} for '{effective_query}':\n"
        )
        print(results[avail].to_string(index=True, max_colwidth=45))


# ── CLI ───────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-keyword semantic search with cluster navigation"
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Keyword to search (runs interactive mode if omitted)",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Cluster ID to inspect / sub-search within",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Sub-query to rank products within the cluster",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    model = build_model(args.model_name)

    if args.keyword is None:
        # Full interactive session
        interactive_search(model)
    else:
        data = load_keyword_data(args.keyword)
        overview = get_cluster_overview(data)
        print(f"\nClusters for '{args.keyword}':\n")
        print(overview.to_string(index=False))

        if args.cluster is not None:
            query = args.query or args.keyword
            results = search_within_cluster(
                query, args.cluster, data, model, top_k=args.top_k
            )
            print(
                f"\nTop {len(results)} results in cluster {args.cluster} for '{query}':\n"
            )
            display_cols = [
                "cluster",
                "cluster_label",
                "name_en",
                "name_zh",
                "brand",
                "price",
                "average_rating",
                "score",
            ]
            avail = [c for c in display_cols if c in results.columns]
            print(results[avail].to_string(index=True, max_colwidth=50))
