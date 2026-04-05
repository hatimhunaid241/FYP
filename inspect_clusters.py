"""
Cluster Inspector
=================
Browse cluster titles and products for any keyword in the dataset.

Usage examples
--------------
# Show cluster summary for one keyword
python inspect_clusters.py --keyword apple

# Show cluster summary for all keywords
python inspect_clusters.py --all

# Show every product in cluster 5 of "apple"
python inspect_clusters.py --keyword apple --cluster 5

# Show every product in cluster 5 with specific columns
python inspect_clusters.py --keyword apple --cluster 5 --cols name_en name_zh price average_rating

# Save the cluster-5 product list to a CSV
python inspect_clusters.py --keyword apple --cluster 5 --save

# Search for a product name within a keyword's clusters
python inspect_clusters.py --keyword apple --search "iphone 15"
"""

import sys
import io
import argparse
from pathlib import Path

# Force UTF-8 output so Chinese characters print correctly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.utils.config_loader import load_config, keyword_paths


# ── Helpers ───────────────────────────────────────────────────────────── #


def load_keyword_clusters(
    keyword: str, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load clusters_labeled.csv and cluster_labels.csv for a keyword.
    Returns (products_df, labels_df).
    """
    paths = keyword_paths(keyword, config)

    if not paths["clusters_labeled_csv"].exists():
        raise FileNotFoundError(
            f"No labeled clusters found for '{keyword}'.\n"
            f"Expected: {paths['clusters_labeled_csv']}\n"
            "Run the pipeline first: python run_pipeline.py --keywords " + keyword
        )

    products_df = pd.read_csv(
        paths["clusters_labeled_csv"], encoding="utf-8-sig", low_memory=False
    )
    labels_df = pd.read_csv(paths["cluster_labels_csv"], encoding="utf-8-sig")
    return products_df, labels_df


def print_cluster_summary(
    keyword: str, products_df: pd.DataFrame, labels_df: pd.DataFrame
) -> None:
    """Print a table: one row per cluster, showing ID, label, size, avg price, avg rating."""
    print(f"\n{'=' * 70}")
    print(
        f"  Keyword: '{keyword}'  —  {len(products_df):,} products  |  {products_df['cluster'].nunique()} clusters"
    )
    print(f"{'=' * 70}")

    rows = []
    for _, label_row in labels_df.sort_values("cluster").iterrows():
        cid = int(label_row["cluster"])
        label = str(label_row["label"])
        subset = products_df[products_df["cluster"] == cid]
        n = len(subset)
        avg_price = (
            f"HK${subset['price'].mean():.0f}" if "price" in subset.columns else "—"
        )
        avg_rating = (
            f"{subset['average_rating'].mean():.2f}"
            if "average_rating" in subset.columns
            else "—"
        )
        rows.append((cid, label, n, avg_price, avg_rating))

    # Column widths
    max_label = max(len(r[1]) for r in rows) if rows else 20
    header = f"  {'ID':>3}  {'Cluster Label':<{max_label}}  {'Products':>8}  {'Avg Price':>10}  {'Avg Rating':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cid, label, n, price, rating in rows:
        print(f"  {cid:>3}  {label:<{max_label}}  {n:>8,}  {price:>10}  {rating:>10}")
    print()


def print_cluster_products(
    keyword: str,
    cluster_id: int,
    products_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    cols: list[str] | None = None,
    max_rows: int = 50,
) -> pd.DataFrame:
    """Print every product in a specific cluster and return the subset DataFrame."""
    subset = (
        products_df[products_df["cluster"] == cluster_id].copy().reset_index(drop=True)
    )

    label_row = labels_df[labels_df["cluster"] == cluster_id]
    label = (
        label_row["label"].iloc[0] if not label_row.empty else f"Cluster {cluster_id}"
    )

    print(f"\n{'=' * 70}")
    print(f"  Keyword: '{keyword}'  |  Cluster {cluster_id}: '{label}'")
    print(f"  {len(subset):,} products total")
    print(f"{'=' * 70}")

    if cols is None:
        # Sensible default columns
        preferred = [
            "cluster",
            "cluster_label",
            "name_en",
            "name_zh",
            "brand",
            "price",
            "average_rating",
            "keyword_source",
        ]
        cols = [c for c in preferred if c in subset.columns]

    if len(subset) > max_rows:
        print(
            f"  (showing first {max_rows} of {len(subset):,} — use --max_rows N to see more)\n"
        )
        display = subset[cols].head(max_rows)
    else:
        display = subset[cols]

    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    print(display.to_string(index=True))
    print()
    return subset


def search_products(
    keyword: str,
    query: str,
    products_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    cols: list[str] | None = None,
) -> None:
    """Search for *query* in name_en / name_zh across all clusters of a keyword."""
    q = query.strip().lower()
    mask = pd.Series(False, index=products_df.index)

    for col in ["name_en", "name_zh", "description_en", "description_zh"]:
        if col in products_df.columns:
            mask |= products_df[col].fillna("").str.lower().str.contains(q, regex=False)

    subset = products_df[mask].copy().reset_index(drop=True)

    print(f"\n{'=' * 70}")
    print(f"  Keyword: '{keyword}'  |  Search: '{query}'  |  {len(subset):,} matches")
    print(f"{'=' * 70}")

    if subset.empty:
        print("  No products matched.")
        return

    if cols is None:
        preferred = [
            "cluster",
            "cluster_label",
            "name_en",
            "name_zh",
            "brand",
            "price",
            "average_rating",
        ]
        cols = [c for c in preferred if c in subset.columns]

    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    print(subset[cols].to_string(index=True))
    print()


def show_all_keywords(config: dict) -> None:
    """Print the cluster summary for every keyword."""
    keywords = config["keywords"]["seed_keywords"]
    for kw in keywords:
        try:
            products_df, labels_df = load_keyword_clusters(kw, config)
            print_cluster_summary(kw, products_df, labels_df)
        except FileNotFoundError as exc:
            print(f"\n  '{kw}': not ready ({exc})")


# ── CLI ───────────────────────────────────────────────────────────────── #


def main():
    parser = argparse.ArgumentParser(
        description="Inspect cluster labels and products for any keyword.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--keyword",
        "-k",
        type=str,
        default=None,
        help="Keyword to inspect (e.g. apple)",
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Show cluster summary for ALL keywords"
    )
    parser.add_argument(
        "--cluster",
        "-c",
        type=int,
        default=None,
        help="Cluster ID to drill into (shows all products)",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        default=None,
        help="Search string to find matching products across all clusters",
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        default=None,
        help="Columns to display (default: name_en name_zh brand price average_rating)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=50,
        help="Max products to display per cluster (default 50)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the displayed cluster subset to a CSV in results/tables/",
    )
    args = parser.parse_args()

    config = load_config("config/config.yaml")

    # ── All keywords summary ─────────────────────────────────────────── #
    if args.all:
        show_all_keywords(config)
        return

    # ── Single keyword required below ────────────────────────────────── #
    if args.keyword is None:
        parser.print_help()
        print("\nError: provide --keyword <keyword> or --all")
        sys.exit(1)

    keyword = args.keyword.strip().lower()
    products_df, labels_df = load_keyword_clusters(keyword, config)

    # ── Search mode ──────────────────────────────────────────────────── #
    if args.search:
        search_products(keyword, args.search, products_df, labels_df, cols=args.cols)
        return

    # ── Cluster drill-down ───────────────────────────────────────────── #
    if args.cluster is not None:
        subset = print_cluster_products(
            keyword,
            args.cluster,
            products_df,
            labels_df,
            cols=args.cols,
            max_rows=args.max_rows,
        )
        if args.save:
            out_dir = Path("results/tables")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{keyword}_cluster{args.cluster}_products.csv"
            subset.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  Saved {len(subset):,} products → {out_path}")
        return

    # ── Default: cluster summary ─────────────────────────────────────── #
    print_cluster_summary(keyword, products_df, labels_df)


if __name__ == "__main__":
    main()
