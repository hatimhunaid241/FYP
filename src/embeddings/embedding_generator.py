"""
Per-Keyword Embedding Generator
=================================
For each keyword, reads its products parquet, cleans bilingual text, generates
multilingual sentence embeddings, and saves:

    data/processed/<keyword>/products_with_embeddings.parquet
    data/processed/<keyword>/product_embeddings.npy

All original product columns are preserved so downstream clustering and search
scripts always have access to price, ratings, keyword_source, etc.

Run from the project root:
    python src/embeddings/embedding_generator.py
    python src/embeddings/embedding_generator.py --keywords apple milk
    python src/embeddings/embedding_generator.py --visualize --keywords apple
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from src.preprocessing.text_cleaner import TextCleaner
from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/embedding_generator.log")


# ── Per-keyword embedding ─────────────────────────────────────────────── #


def embed_keyword(
    keyword: str,
    model,  # SentenceTransformer instance passed in (loaded once)
    config: dict,
) -> pd.DataFrame:
    """
    Generate embeddings for one keyword's product set.

    Reads:  data/processed/<keyword>/products.parquet
    Writes: data/processed/<keyword>/products_with_embeddings.parquet
            data/processed/<keyword>/product_embeddings.npy

    Returns the enriched DataFrame.
    """
    paths = keyword_paths(keyword, config)
    batch_size = config["embeddings"].get("batch_size", 64)

    if not paths["products_parquet"].exists():
        raise FileNotFoundError(
            f"Products not found for keyword '{keyword}' at {paths['products_parquet']}. "
            "Run collect_products.py first."
        )

    products = pd.read_parquet(paths["products_parquet"])
    logger.info(f"  [{keyword}] Loaded {len(products):,} products")

    required = ["name_zh", "name_en", "description_zh", "description_en"]
    missing = [c for c in required if c not in products.columns]
    if missing:
        raise KeyError(f"[{keyword}] Missing columns: {missing}")

    # Clean text
    cleaner = TextCleaner(remove_stopwords=False, convert_simplified=True)
    for col in required:
        products[f"{col}_clean"] = (
            products[col].fillna("").astype(str).map(cleaner.clean_text)
        )

    # Build combined embedding text: ZH first, then EN
    products["embedding_text"] = (
        products["name_zh_clean"].fillna("")
        + " "
        + products["description_zh_clean"].fillna("")
        + " "
        + products["name_en_clean"].fillna("")
        + " "
        + products["description_en_clean"].fillna("")
    ).str.strip()

    # Generate embeddings
    texts = products["embedding_text"].tolist()
    logger.info(f"  [{keyword}] Encoding {len(texts):,} texts…")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info(f"  [{keyword}] Embeddings shape: {embeddings.shape}")

    # Save enriched parquet (all columns preserved)
    products.to_parquet(paths["embeddings_parquet"], index=False)

    # Save raw embedding matrix
    np.save(paths["embeddings_npy"], embeddings)

    logger.info(f"  [{keyword}] Saved → {paths['embeddings_parquet']}")
    logger.info(f"  [{keyword}] Saved → {paths['embeddings_npy']}")
    return products


def embed_all_keywords(
    keywords: list[str] | None = None,
    config_path: str = "config/config.yaml",
    force: bool = False,
) -> None:
    """
    Embed every keyword's product set (skips already-done ones unless force=True).
    The SentenceTransformer model is loaded once and reused across all keywords.
    """
    config = load_config(config_path)
    if keywords is None:
        keywords = config["keywords"]["seed_keywords"]

    from sentence_transformers import SentenceTransformer  # lazy import

    model_name = config["embeddings"]["model_name"]
    logger.info(f"Loading model '{model_name}'…")
    model = SentenceTransformer(model_name)

    logger.info("=" * 60)
    logger.info(f"Per-keyword embedding: {len(keywords)} keywords")
    logger.info("=" * 60)

    for kw in keywords:
        paths = keyword_paths(kw, config)
        if not force and paths["embeddings_npy"].exists():
            logger.info(f"  '{kw}' embeddings already exist — skipping")
            continue
        try:
            embed_keyword(kw, model, config)
        except Exception as exc:
            logger.error(f"  '{kw}' failed: {exc}")

    logger.info("Embedding generation complete.")


# ── 2-D visualisation (optional, per keyword) ────────────────────────── #


def visualize_keyword(
    keyword: str,
    config: dict,
    method: str = "umap",
    sample_size: int = 500,
) -> None:
    """Project one keyword's embeddings to 2-D and save HTML + PNG."""
    try:
        import umap as umap_lib
        import plotly.express as px
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        logger.error(f"Visualisation dependency missing: {e}")
        return

    paths = keyword_paths(keyword, config)
    if not paths["embeddings_npy"].exists():
        logger.warning(f"No embeddings for '{keyword}' — run embed first.")
        return

    embs = np.load(paths["embeddings_npy"])
    df = pd.read_parquet(paths["embeddings_parquet"])

    n = min(sample_size, len(embs))
    rng = np.random.default_rng(config.get("random_seed", 42))
    idx = rng.choice(len(embs), size=n, replace=False)
    embs_s, df_s = embs[idx], df.iloc[idx].reset_index(drop=True)

    if method == "umap":
        reducer = umap_lib.UMAP(
            n_components=2, random_state=config.get("random_seed", 42)
        )
        coords = reducer.fit_transform(embs_s)
        label = "UMAP"
    else:
        from sklearn.manifold import TSNE

        coords = TSNE(
            n_components=2, random_state=config.get("random_seed", 42)
        ).fit_transform(embs_s)
        label = "t-SNE"

    fig_dir = Path(config["paths"]["results_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Build plot dataframe with cluster information if available
    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "name": df_s.get("name_en", df_s.get("name_zh", "")),
        }
    )
    
    # Try to load cluster assignments
    cluster_file = paths["clusters_csv"]
    if cluster_file.exists():
        clusters_df = pd.read_csv(cluster_file)
        # Map clusters by product_id
        product_id_to_cluster = dict(zip(clusters_df["product_id"], clusters_df["cluster"]))
        plot_df["cluster"] = df_s["product_id"].map(product_id_to_cluster).fillna(-1).astype(int)
        
        # Plotly scatter with cluster coloring
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="cluster",
            hover_data=["name", "cluster"],
            title=f"'{keyword}' embeddings — {label}",
            color_continuous_scale="Viridis",
        )
        fig.write_html(fig_dir / f"embeddings_{keyword}_{method}.html")
        
        # Matplotlib scatter with cluster coloring
        plt.figure(figsize=(10, 8))
        clusters = sorted(plot_df["cluster"].unique())
        colors = sns.color_palette("husl", len(clusters))
        cluster_colors = {c: colors[i] for i, c in enumerate(clusters)}
        
        for cluster in clusters:
            mask = plot_df["cluster"] == cluster
            label_text = f"Cluster {cluster}" if cluster >= 0 else "Unassigned"
            plt.scatter(
                plot_df[mask]["x"],
                plot_df[mask]["y"],
                c=[cluster_colors[cluster]],
                label=label_text,
                s=30,
                alpha=0.7,
                edgecolors="black" if cluster >= 0 else "gray",
                linewidth=0.5 if cluster >= 0 else 0.3,
            )
        
        plt.title(f"'{keyword}' embeddings — {label} (Colored by Cluster)")
        plt.xlabel(f"{label} 1")
        plt.ylabel(f"{label} 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / f"embeddings_{keyword}_{method}.png", dpi=130, bbox_inches="tight")
        plt.close()
    else:
        # Fallback: single color if no clusters found
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            hover_data=["name"],
            title=f"'{keyword}' embeddings — {label}",
        )
        fig.write_html(fig_dir / f"embeddings_{keyword}_{method}.html")
        
        plt.figure(figsize=(9, 7))
        plt.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.6, color="#3498db")
        plt.title(f"'{keyword}' embeddings — {label}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"embeddings_{keyword}_{method}.png", dpi=130)
        plt.close()
    
    logger.info(f"  Saved visualisation for '{keyword}' → {fig_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate per-keyword product embeddings"
    )
    parser.add_argument("--keywords", nargs="+", default=None)
    parser.add_argument(
        "--force", action="store_true", help="Re-embed even if files exist"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Also generate 2-D plots"
    )
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"])
    args = parser.parse_args()

    embed_all_keywords(keywords=args.keywords, force=args.force)

    if args.visualize:
        cfg = load_config()
        kws = args.keywords or cfg["keywords"]["seed_keywords"]
        for kw in kws:
            visualize_keyword(kw, cfg, method=args.method)
