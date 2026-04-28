"""Generate Top-N product tables for semantic clusters.

Example:
    python src/evaluation/top_products.py --keyword apple --top-n 5

Output:
    results/tables/apple_top5_products_by_cluster.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def build_top_products_by_cluster(
    full_csv: Path,
    keyword: str,
    top_n: int = 5,
) -> pd.DataFrame:
    if not full_csv.exists():
        raise FileNotFoundError(f"Missing search comparison CSV: {full_csv}")

    df = pd.read_csv(full_csv, low_memory=False)
    df = df[df["keyword"].astype(str).str.lower() == keyword.lower()]
    df = df[df["system"] == "semantic"].copy()
    if df.empty:
        raise ValueError(f"No semantic rows found for keyword '{keyword}'")

    # Keep only the highest relevance score per product, cluster, and cluster label.
    agg = (
        df.sort_values("score", ascending=False)
        .groupby(["product_id", "cluster", "cluster_label"], as_index=False)
        .first()
    )

    agg["cluster_label"] = agg["cluster_label"].fillna("Cluster " + agg["cluster"].astype(str))
    agg["name_en"] = agg["name_en"].fillna("")
    agg["name_zh"] = agg["name_zh"].fillna("")

    rows = []
    for cluster_key, group in agg.groupby(["cluster", "cluster_label"], sort=True):
        group_sorted = group.sort_values("score", ascending=False).head(top_n)
        for rank, row in enumerate(group_sorted.itertuples(index=False), start=1):
            rows.append(
                {
                    "cluster": int(row.cluster) if pd.notna(row.cluster) else None,
                    "cluster_label": row.cluster_label,
                    "rank": rank,
                    "name_en": row.name_en,
                    "name_zh": row.name_zh,
                    "average_rating": row.average_rating,
                    "score": row.score,
                    "query_with_max_score": row.query,
                }
            )

    result = pd.DataFrame(rows)
    result = result.sort_values(["cluster", "rank"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a top-N product table for semantic clusters."
    )
    parser.add_argument(
        "--keyword",
        default="apple",
        help="Seed keyword to build the top product table for.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top products to keep per cluster.",
    )
    parser.add_argument(
        "--full-csv",
        default="results/tables/search_comparison.csv",
        help="Path to the full search comparison CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tables/apple_top5_products_by_cluster.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    full_csv = Path(args.full_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    table = build_top_products_by_cluster(full_csv, args.keyword, top_n=args.top_n)
    table.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved top-{args.top_n} table for '{args.keyword}' to {output_csv}")


if __name__ == "__main__":
    main()
