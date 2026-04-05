"""
Search Evaluation — Qualitative Comparison Report (Per-Keyword)
================================================================
For a set of test queries, runs both semantic (cluster navigation) and
TF-IDF keyword baseline within each keyword's product set and produces:

  results/tables/comparison_summary.csv  — one row per (keyword, query)
  results/tables/search_comparison.csv   — full ranked results for every query

Primary signals reported:
  1. Cluster overlap ratio  — do both systems agree on which clusters are relevant?
  2. Average cosine score   — semantic model confidence
  3. Average keyword score  — TF-IDF match strength

Run from the project root:
    python src/evaluation/search_metrics.py
    python src/evaluation/search_metrics.py --keywords apple milk
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings

warnings.filterwarnings("ignore")

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.search.semantic_search import build_model, load_keyword_data
from src.search.keyword_baseline import build_keyword_index, keyword_search
from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/search_metrics.log")


# ── Test queries per keyword ──────────────────────────────────────────── #
# Maps keyword → list of test queries (bilingual, ambiguous where relevant)

DEFAULT_QUERIES_PER_KEYWORD: Dict[str, List[str]] = {
    "apple": [
        "apple iphone",
        "apple fruit",
        "apple pencil",
        "apple watch",
        "蘋果手機",
        "青蘋果",
    ],
    "samsung": ["samsung phone", "samsung galaxy", "samsung tv", "三星手機"],
    "xiaomi": ["xiaomi phone", "xiaomi mi", "小米手機", "小米電視"],
    "speaker": [
        "bluetooth speaker",
        "wireless speaker",
        "portable speaker",
        "藍牙喇叭",
    ],
    "battery": ["phone battery", "aa battery", "rechargeable battery", "電池"],
    "charger": ["wireless charger", "usb charger", "fast charger", "無線充電"],
    "fan": ["desk fan", "cooling fan", "handheld fan", "手持風扇", "散熱風扇"],
    "camera": ["digital camera", "camera lens", "security camera", "相機"],
    "headphone": [
        "wireless headphone",
        "bluetooth earphone",
        "noise cancelling",
        "藍牙耳機",
    ],
    "cable": ["usb cable", "hdmi cable", "charging cable", "數據線"],
    "milk": ["whole milk", "baby formula", "almond milk", "牛奶", "奶粉"],
    "oil": ["olive oil", "cooking oil", "essential oil", "motor oil", "橄欖油"],
    "mask": [
        "face mask skincare",
        "surgical mask",
        "sheet mask",
        "護膚面膜",
        "外科口罩",
    ],
    "tea": ["green tea", "milk tea", "herbal tea", "綠茶", "奶茶"],
    "coffee": ["instant coffee", "coffee beans", "espresso machine", "咖啡"],
    "cream": ["moisturizing cream", "whipping cream", "sunscreen cream", "保濕面霜"],
    "vitamin": ["vitamin c", "vitamin d", "multivitamin", "維他命C", "維他命"],
    "protein": ["whey protein", "protein bar", "plant protein", "蛋白質"],
    "brush": ["makeup brush", "toothbrush", "paint brush", "化妝掃", "牙刷"],
    "bag": ["backpack", "handbag", "shopping bag", "手袋", "背包"],
}


# ── Per-query evaluation ──────────────────────────────────────────────── #


def evaluate_query(
    keyword: str,
    query: str,
    data: Dict[str, Any],
    model: Any,
    kw_index: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Any]:
    """Run one query through semantic and keyword search; return comparison stats."""
    # Semantic: rank all products in this keyword by cosine similarity
    q_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    scores = (data["embeddings"] @ q_vec.T).squeeze()
    sem_df = data["df"].copy()
    sem_df["score"] = scores
    sem_results = (
        sem_df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    )

    # Keyword
    kw_results = keyword_search(query, data["df"], index=kw_index, top_k=top_k)

    sem_clusters = (
        set(sem_results["cluster"].dropna().astype(int).tolist())
        if "cluster" in sem_results.columns
        else set()
    )
    kw_clusters = (
        set(kw_results["cluster"].dropna().astype(int).tolist())
        if "cluster" in kw_results.columns
        else set()
    )
    overlap = sorted(sem_clusters & kw_clusters)
    ratio = len(overlap) / max(1, min(len(sem_clusters), len(kw_clusters)))

    def _mean(df: pd.DataFrame, col: str) -> Optional[float]:
        return float(df[col].mean()) if col in df.columns and not df.empty else None

    return {
        "keyword": keyword,
        "query": query,
        "sem_n_results": len(sem_results),
        "sem_avg_score": _mean(sem_results, "score"),
        "sem_n_clusters": len(sem_clusters),
        "sem_clusters": sorted(sem_clusters),
        "kw_n_results": len(kw_results),
        "kw_avg_score": _mean(kw_results, "keyword_score"),
        "kw_n_clusters": len(kw_clusters),
        "kw_clusters": sorted(kw_clusters),
        "overlap_n_clusters": len(overlap),
        "cluster_overlap_ratio": float(ratio),
        "_sem_df": sem_results,
        "_kw_df": kw_results,
    }


# ── Full evaluation run ───────────────────────────────────────────────── #


def run_evaluation(
    keywords: Optional[List[str]] = None,
    queries_per_keyword: Optional[Dict[str, List[str]]] = None,
    top_k: int = 10,
    output_dir: str = "results/tables",
    config_path: str = "config/config.yaml",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all (keyword, query) pairs.

    Args:
        keywords:             Keywords to evaluate (default: all from config).
        queries_per_keyword:  Dict of keyword → [queries] (default: built-in set).
        top_k:                Results per query.
        output_dir:           Where to write CSVs.
        config_path:          Path to config.yaml.
        verbose:              Print per-query summaries.

    Returns:
        Summary DataFrame (one row per keyword×query pair).
    """
    config = load_config(config_path)
    if keywords is None:
        keywords = config["keywords"]["seed_keywords"]
    if queries_per_keyword is None:
        queries_per_keyword = DEFAULT_QUERIES_PER_KEYWORD

    logger.info("=" * 60)
    logger.info(f"Search evaluation: {len(keywords)} keywords")
    logger.info("=" * 60)

    model = build_model(config_path=config_path)

    summary_rows: List[Dict] = []
    all_rows: List[pd.DataFrame] = []

    for kw in keywords:
        paths = keyword_paths(kw, config)
        if not paths["clusters_labeled_csv"].exists():
            logger.warning(f"  '{kw}' not ready — skipping evaluation")
            continue

        try:
            data = load_keyword_data(kw, config_path=config_path)
        except Exception as exc:
            logger.error(f"  '{kw}' load failed: {exc}")
            continue

        kw_index = build_keyword_index(data["df"])
        queries = queries_per_keyword.get(kw, [kw])

        for query in queries:
            try:
                result = evaluate_query(kw, query, data, model, kw_index, top_k=top_k)
            except Exception as exc:
                logger.error(f"  [{kw}] query '{query}' failed: {exc}")
                continue

            summary_rows.append(
                {k: v for k, v in result.items() if not k.startswith("_")}
            )

            sem_df = result["_sem_df"].copy()
            if not sem_df.empty:
                sem_df["query"] = query
                sem_df["keyword"] = kw
                sem_df["system"] = "semantic"
                all_rows.append(sem_df)

            kw_df = result["_kw_df"].copy()
            if not kw_df.empty:
                kw_df["query"] = query
                kw_df["keyword"] = kw
                kw_df["system"] = "keyword"
                kw_df = kw_df.rename(columns={"keyword_score": "score"})
                all_rows.append(kw_df)

            if verbose:
                q_sil = result["sem_avg_score"]
                q_kw = result["kw_avg_score"]
                sil_str = f"{q_sil:.3f}" if q_sil is not None else "N/A"
                kw_str = f"{q_kw:.3f}" if q_kw is not None else "N/A"
                logger.info(
                    f"  [{kw}] '{query}' — "
                    f"sem_score={sil_str}, "
                    f"kw_score={kw_str}, "
                    f"overlap={result['cluster_overlap_ratio']:.0%}"
                )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        f"{output_dir}/comparison_summary.csv", index=False, encoding="utf-8-sig"
    )
    logger.info(f"Summary → {output_dir}/comparison_summary.csv")

    if all_rows:
        full_df = pd.concat(all_rows, ignore_index=True)
        full_df.to_csv(
            f"{output_dir}/search_comparison.csv", index=False, encoding="utf-8-sig"
        )
        logger.info(f"Full results → {output_dir}/search_comparison.csv")

    if verbose and not summary_df.empty:
        print("\n" + "=" * 60)
        print("  EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Total (keyword, query) pairs : {len(summary_df)}")
        if "sem_avg_score" in summary_df:
            print(
                f"  Avg semantic score          : {summary_df['sem_avg_score'].mean():.4f}"
            )
        if "cluster_overlap_ratio" in summary_df:
            print(
                f"  Avg cluster overlap         : {summary_df['cluster_overlap_ratio'].mean():.2%}"
            )
        print("=" * 60)

    return summary_df


# ── CLI ───────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-keyword search evaluation")
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_dir", default="results/tables")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        keywords=args.keywords,
        top_k=args.top_k,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
