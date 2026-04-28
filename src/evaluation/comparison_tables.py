"""Generate comparison tables: keyword vs semantic search top-5 per cluster.

Example:
    python src/evaluation/comparison_tables.py --keyword apple --top-n 5

Output:
    results/tables/apple_comparison_keyword_vs_semantic.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def build_comparison_table(
    full_csv: Path,
    keyword: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build a side-by-side comparison of keyword vs semantic search top products per cluster."""
    if not full_csv.exists():
        raise FileNotFoundError(f"Missing search comparison CSV: {full_csv}")

    df = pd.read_csv(full_csv, low_memory=False)
    df = df[df["keyword"].astype(str).str.lower() == keyword.lower()]
    
    rows = []
    
    # Get all unique clusters from semantic search results
    semantic_df = df[df["system"] == "semantic"].copy()
    keyword_df = df[df["system"] == "keyword"].copy()
    
    if semantic_df.empty:
        raise ValueError(f"No semantic rows found for keyword '{keyword}'")
    
    clusters = sorted(semantic_df["cluster"].dropna().unique())
    
    for cluster_id in clusters:
        # Get cluster label from semantic data
        cluster_label = semantic_df[semantic_df["cluster"] == cluster_id]["cluster_label"].iloc[0]
        cluster_label = cluster_label if pd.notna(cluster_label) else f"Cluster {int(cluster_id)}"
        
        # Get top N semantic results for this cluster
        sem_cluster = (
            semantic_df[semantic_df["cluster"] == cluster_id]
            .sort_values("score", ascending=False)
            .head(top_n)
        )
        
        # Get top N keyword results for this cluster
        kw_cluster = (
            keyword_df[keyword_df["cluster"] == cluster_id]
            .sort_values("score", ascending=False)
            .head(top_n)
        )
        
        # Determine max rows to show
        max_rows = max(len(sem_cluster), len(kw_cluster))
        
        for i in range(max_rows):
            row_dict = {
                "cluster": int(cluster_id),
                "cluster_label": cluster_label,
                "rank": i + 1,
            }
            
            # Semantic side
            if i < len(sem_cluster):
                sem_row = sem_cluster.iloc[i]
                row_dict["semantic_product"] = sem_row["name_en"]
                row_dict["semantic_product_zh"] = sem_row["name_zh"]
                row_dict["semantic_score"] = sem_row["score"]
            else:
                row_dict["semantic_product"] = ""
                row_dict["semantic_product_zh"] = ""
                row_dict["semantic_score"] = None
            
            # Keyword side
            if i < len(kw_cluster):
                kw_row = kw_cluster.iloc[i]
                row_dict["keyword_product"] = kw_row["name_en"]
                row_dict["keyword_product_zh"] = kw_row["name_zh"]
                row_dict["keyword_score"] = kw_row["score"]
            else:
                row_dict["keyword_product"] = ""
                row_dict["keyword_product_zh"] = ""
                row_dict["keyword_score"] = None
            
            rows.append(row_dict)
    
    result = pd.DataFrame(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a comparison table of keyword vs semantic search top-N products per cluster."
    )
    parser.add_argument(
        "--keyword",
        default="apple",
        help="Seed keyword to build the comparison table for.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top products to keep per cluster per system.",
    )
    parser.add_argument(
        "--full-csv",
        default="results/tables/search_comparison.csv",
        help="Path to the full search comparison CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tables/apple_comparison_keyword_vs_semantic.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    full_csv = Path(args.full_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(full_csv, args.keyword, top_n=args.top_n)
    table.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved comparison table for '{args.keyword}' to {output_csv}")


if __name__ == "__main__":
    main()
