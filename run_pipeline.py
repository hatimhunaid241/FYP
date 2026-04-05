"""
HKTVmall Semantic Search — End-to-End Per-Keyword Pipeline
============================================================
Executes the complete pipeline for every keyword independently:

  Step 1 — collect   : Fetch ~1 000 products per keyword from HKTVmall API
  Step 2 — embed     : Clean bilingual text + generate multilingual embeddings
  Step 3 — cluster   : K-Means with auto-K (elbow + silhouette) per keyword
  Step 4 — label     : TF-IDF cluster labels per keyword
  Step 5 — metrics   : Compute silhouette / CH / DB per keyword
  Step 6 — evaluate  : Qualitative semantic vs keyword comparison
  Step 7 — visualize : Generate all comparison charts

Each keyword's artefacts are written to data/processed/<keyword>/ so that
keywords are fully independent and can be re-run individually.

Usage
-----
    python run_pipeline.py                             # all keywords, all steps
    python run_pipeline.py --keywords apple milk fan  # specific keywords
    python run_pipeline.py --skip collect              # skip data collection
    python run_pipeline.py --from cluster              # resume from clustering
    python run_pipeline.py --steps embed cluster label # specific steps only
    python run_pipeline.py --force                     # overwrite existing files
"""

import sys
import argparse
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger("pipeline", log_file="logs/pipeline.log")

STEP_ORDER = [
    "collect",
    "embed",
    "cluster",
    "label",
    "metrics",
    "evaluate",
    "visualize",
]


# ── Pipeline steps ────────────────────────────────────────────────────── #


def step_collect(keywords: list, config: dict, force: bool) -> bool:
    logger.info("[Step 1] Data Collection")
    try:
        from src.data_collection.collect_products import ProductCollector

        collector = ProductCollector()
        if force:
            # Delete existing parquets so they are re-collected
            for kw in keywords:
                p = keyword_paths(kw, config)["products_parquet"]
                if p.exists():
                    p.unlink()
                    logger.info(f"  Deleted {p} (--force)")
        collector.collect_all_keywords(keywords=keywords)
        return True
    except Exception:
        logger.error("  Collection failed:\n" + traceback.format_exc())
        return False


def step_embed(keywords: list, config: dict, force: bool) -> bool:
    logger.info("[Step 2] Embedding Generation")
    try:
        from src.embeddings.embedding_generator import embed_all_keywords

        embed_all_keywords(keywords=keywords, force=force)
        return True
    except Exception:
        logger.error("  Embedding failed:\n" + traceback.format_exc())
        return False


def step_cluster(keywords: list, config: dict, force: bool) -> bool:
    logger.info("[Step 3] K-Means Clustering (per keyword)")
    try:
        from src.clustering.kmeans_cluster import cluster_all_keywords

        cluster_all_keywords(keywords=keywords, force=force)
        return True
    except Exception:
        logger.error("  Clustering failed:\n" + traceback.format_exc())
        return False


def step_label(keywords: list, config: dict, force: bool) -> bool:
    logger.info("[Step 4] Cluster Labeling (per keyword)")
    try:
        from src.clustering.cluster_labeling import label_all_keywords

        label_all_keywords(keywords=keywords, force=force)
        return True
    except Exception:
        logger.error("  Labeling failed:\n" + traceback.format_exc())
        return False


def step_metrics(keywords: list, config: dict, force: bool) -> bool:
    logger.info("[Step 5] Clustering Quality Metrics (per keyword)")
    try:
        import json
        import numpy as np
        import pandas as pd
        from src.clustering.clustering_metrics import (
            compute_all_metrics,
            print_metrics_report,
            save_metrics,
            plot_silhouette,
        )

        results_dir = Path(config["paths"]["results_dir"])
        all_metrics = {}

        for kw in keywords:
            paths = keyword_paths(kw, config)
            if (
                not paths["embeddings_npy"].exists()
                or not paths["clusters_csv"].exists()
            ):
                logger.warning(f"  '{kw}' — missing files, skipping metrics")
                continue
            emb = np.load(paths["embeddings_npy"])
            df = pd.read_csv(paths["clusters_csv"])
            labels = df["cluster"].to_numpy()
            m = compute_all_metrics(
                emb, labels, random_state=config.get("random_seed", 42)
            )
            m["keyword"] = kw
            all_metrics[kw] = m
            sil_val = m.get("silhouette_score")
            sil_str = f"{sil_val:.4f}" if sil_val is not None else "N/A"
            logger.info(f"  [{kw}] silhouette={sil_str}")

            # Per-keyword silhouette plot
            fig_dir = results_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_silhouette(
                    emb, labels, output_path=str(fig_dir / f"silhouette_{kw}.png")
                )
            except Exception:
                pass

        # Save aggregate metrics table
        tables_dir = results_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = tables_dir / "clustering_metrics_all.json"

        def _default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        with open(metrics_path, "w") as fh:
            json.dump(all_metrics, fh, indent=2, default=_default)
        logger.info(f"  Metrics saved → {metrics_path}")
        return True
    except Exception:
        logger.error("  Metrics failed:\n" + traceback.format_exc())
        return False


def step_evaluate(keywords: list, config: dict, force: bool) -> bool:
    logger.info("[Step 6] Search Evaluation (semantic vs keyword)")
    try:
        from src.evaluation.search_metrics import run_evaluation

        run_evaluation(
            keywords=keywords,
            top_k=config.get("evaluation", {}).get("top_k", 10),
            output_dir=str(Path(config["paths"]["results_dir"]) / "tables"),
            config_path="config/config.yaml",
        )
        return True
    except Exception:
        logger.error("  Evaluation failed:\n" + traceback.format_exc())
        return False


def step_visualize(keywords: list, config: dict, force: bool) -> bool:
    logger.info("[Step 7] Generating Visualizations")
    try:
        from src.evaluation.visualization import generate_all_visualizations

        generate_all_visualizations(
            summary_csv="results/tables/comparison_summary.csv",
            full_csv="results/tables/search_comparison.csv",
            output_dir="results/figures",
        )
        return True
    except Exception:
        logger.error("  Visualization failed:\n" + traceback.format_exc())
        return False


STEPS = {
    "collect": step_collect,
    "embed": step_embed,
    "cluster": step_cluster,
    "label": step_label,
    "metrics": step_metrics,
    "evaluate": step_evaluate,
    "visualize": step_visualize,
}


# ── Main ──────────────────────────────────────────────────────────────── #


def main():
    parser = argparse.ArgumentParser(
        description="HKTVmall Semantic Search — Per-Keyword Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Specific keywords (default: all from config)",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=STEP_ORDER, help="Run only these steps"
    )
    parser.add_argument(
        "--skip", nargs="+", choices=STEP_ORDER, help="Skip these steps"
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        choices=STEP_ORDER,
        help="Run from this step onward (inclusive)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing artefacts"
    )
    args = parser.parse_args()

    config = load_config("config/config.yaml")
    keywords: list = args.keywords or config["keywords"]["seed_keywords"]

    # Determine steps
    if args.steps:
        to_run = args.steps
    elif args.from_step:
        to_run = STEP_ORDER[STEP_ORDER.index(args.from_step) :]
    else:
        to_run = STEP_ORDER

    skip = set(args.skip or [])
    to_run = [s for s in to_run if s not in skip]

    # Ensure output dirs exist
    for d in ["logs", "results/tables", "results/figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("HKTVmall Semantic Search — Per-Keyword Pipeline")
    logger.info(f"Keywords : {keywords}")
    logger.info(f"Steps    : {to_run}")
    logger.info("=" * 65)

    results = {}
    total_start = time.time()

    for step_name in to_run:
        logger.info(f"\n{'─' * 65}")
        t0 = time.time()
        ok = STEPS[step_name](keywords, config, args.force)
        results[step_name] = ok
        logger.info(
            f"  {step_name}: {'OK' if ok else 'FAILED'} ({time.time() - t0:.1f}s)"
        )

    # Summary
    logger.info("\n" + "=" * 65)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 65)
    for step, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'}  {step}")
    logger.info(f"\n  Total: {time.time() - total_start:.1f}s")
    n_fail = sum(1 for ok in results.values() if not ok)
    logger.info(
        "  All steps OK."
        if n_fail == 0
        else f"  {n_fail} step(s) failed — check logs/pipeline.log"
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
