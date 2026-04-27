"""
Render presentation-ready charts for the HKTVmall semantic search project.

This script produces the evaluation plots used for the final presentation,
plus per-keyword embedding visualizations for semantic clustering.

Generated graphs:
  - cluster_overlap.png
  - cluster_diversity.png
  - score_distributions.png
  - price_rating_scatter.png
  - category_breakdown.png
  - embeddings_<keyword>_umap.png

Usage:
    python render_presentation_figures.py
    python render_presentation_figures.py --keywords apple milk
    python render_presentation_figures.py --output-dir results/figures/presentation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.embeddings.embedding_generator import visualize_keyword
from src.evaluation.visualization import generate_all_visualizations
from src.utils.config_loader import load_config


def generate_presentation_assets(
    keywords: list[str] | None = None,
    output_dir: str = "results/figures",
    embedding_method: str = "umap",
    sample_size: int = 500,
    skip_embeddings: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating evaluation charts in: {output_path}")
    generate_all_visualizations(output_dir=str(output_path))

    if skip_embeddings:
        print("Skipping embedding visualizations.")
        return

    cfg = load_config()
    if keywords is None:
        keywords = cfg["keywords"]["seed_keywords"][:2]

    for keyword in keywords:
        print(f"Generating embedding plot for keyword: {keyword}")
        visualize_keyword(
            keyword,
            cfg,
            method=embedding_method,
            sample_size=sample_size,
        )

    print("Presentation assets generated.")
    print("Files to use:")
    print("  - cluster_overlap.png")
    print("  - cluster_diversity.png")
    print("  - score_distributions.png")
    print("  - price_rating_scatter.png")
    print("  - category_breakdown.png")
    print("  - embeddings_<keyword>_umap.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render presentation graphs for semantic search deliverables"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Keyword(s) to visualize with embedding projection",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory to save presentation figures",
    )
    parser.add_argument(
        "--method",
        default="umap",
        choices=["umap", "tsne"],
        help="2-D projection method for embedding visualizations",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of products to sample for embedding projection",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Generate only evaluation charts, not embedding projection plots",
    )
    args = parser.parse_args()

    generate_presentation_assets(
        keywords=args.keywords,
        output_dir=args.output_dir,
        embedding_method=args.method,
        sample_size=args.sample_size,
        skip_embeddings=args.skip_embeddings,
    )
