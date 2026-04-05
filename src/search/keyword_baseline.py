"""
Per-Keyword TF-IDF Baseline Search
=====================================
Provides a keyword-search baseline that operates on the same per-keyword
product set used by the semantic system — so comparisons are apples-to-apples.

Two functions are exposed:
  keyword_search()          — TF-IDF ranked retrieval within one keyword's data
  compare_search_results()  — runs both semantic and keyword search and reports
                              cluster overlap (how much do the two systems agree?)

Run from the project root:
    python src/search/keyword_baseline.py --keyword apple --query "iphone 15"
    python src/search/keyword_baseline.py --keyword mask --query "surgical mask" --mode tfidf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.search.semantic_search import (
    build_model,
    get_cluster_overview,
    load_keyword_data,
    search_within_cluster,
)
from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/keyword_baseline.log")


# ── TF-IDF index for one keyword ──────────────────────────────────────── #


def build_keyword_index(
    df: pd.DataFrame,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a TF-IDF index over all products in *df*.

    Returns:
        {
          'vectorizer':   fitted TfidfVectorizer,
          'tfidf_matrix': sparse (N, V) matrix (row-normalised),
          'combined_text': Series of combined text used for 'contains'/'exact' modes,
        }
    """
    if fields is None:
        candidates = [
            "name_en_clean",
            "name_zh_clean",
            "description_en_clean",
            "description_zh_clean",
            "name_en",
            "name_zh",
            "description_en",
            "description_zh",
            "cluster_label",
        ]
        fields = [f for f in candidates if f in df.columns]

    if not fields:
        raise ValueError("No usable text fields found.")

    combined = df[fields].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    vec = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        strip_accents="unicode",
    )
    mat = vec.fit_transform(combined)
    mat_normed = normalize(mat, norm="l2")

    logger.info(
        f"TF-IDF index: {mat_normed.shape[0]:,} docs × {mat_normed.shape[1]:,} features"
    )
    return {"vectorizer": vec, "tfidf_matrix": mat_normed, "combined_text": combined}


# ── Keyword search ────────────────────────────────────────────────────── #


def keyword_search(
    query: str,
    df: pd.DataFrame,
    index: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    mode: str = "tfidf",
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Search *df* using one of three modes:

    tfidf    — cosine similarity via TF-IDF vectors (recommended)
    contains — count query token occurrences in product text
    exact    — binary 1/0 depending on whether the full phrase is present
    """
    q = str(query).strip().lower()
    if not q:
        return pd.DataFrame()

    if mode == "tfidf":
        if index is None:
            index = build_keyword_index(df, fields)
        q_vec = index["vectorizer"].transform([q])
        q_norm = normalize(q_vec, norm="l2")
        scores = (index["tfidf_matrix"] @ q_norm.T).toarray().squeeze()
        result_df = df.copy()
        result_df["keyword_score"] = scores.astype(float)

    elif mode in {"contains", "exact"}:
        candidates = [
            "name_en_clean",
            "name_zh_clean",
            "name_en",
            "name_zh",
            "cluster_label",
        ]
        cols = fields or [f for f in candidates if f in df.columns]
        combined = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        tokens = q.split()
        if mode == "exact":
            sc = [1.0 if q in text else 0.0 for text in combined]
        else:
            sc = [sum(text.count(t) for t in tokens) for text in combined]
        result_df = df.copy()
        result_df["keyword_score"] = np.array(sc, dtype=float)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: tfidf, contains, exact")

    result_df = result_df[result_df["keyword_score"] > 0]
    if result_df.empty:
        return result_df

    return (
        result_df.sort_values("keyword_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


# ── Comparison helper ─────────────────────────────────────────────────── #


def compare_search_results(
    keyword: str,
    query: str,
    top_k: int = 10,
    cluster_id: Optional[int] = None,
    keyword_mode: str = "tfidf",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Run semantic and keyword search for the same keyword+query and compare.

    If *cluster_id* is specified, semantic search is restricted to that cluster
    (sub-search mode).  Keyword search always operates on the full keyword set.
    """
    data = load_keyword_data(keyword, config_path=config_path)
    model = build_model(config_path=config_path)

    # Semantic
    if cluster_id is not None:
        sem_results = search_within_cluster(query, cluster_id, data, model, top_k=top_k)
    else:
        # Search across all products in this keyword, rank by cosine
        from sklearn.preprocessing import normalize as sk_norm

        q_vec = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores = (data["embeddings"] @ q_vec.T).squeeze()
        sem_df = data["df"].copy()
        sem_df["score"] = scores
        sem_results = (
            sem_df.sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    # Keyword
    kw_index = build_keyword_index(data["df"])
    kw_results = keyword_search(
        query, data["df"], index=kw_index, top_k=top_k, mode=keyword_mode
    )

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

    return {
        "keyword": keyword,
        "query": query,
        "semantic_results": sem_results,
        "keyword_results": kw_results,
        "overlap_clusters": overlap,
        "cluster_overlap_ratio": float(ratio),
    }


# ── CLI ───────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-keyword keyword baseline + comparison"
    )
    parser.add_argument("--keyword", required=True, help="Keyword (e.g. apple)")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--mode", choices=["tfidf", "contains", "exact"], default="tfidf"
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Restrict semantic search to this cluster ID",
    )
    args = parser.parse_args()

    res = compare_search_results(
        keyword=args.keyword,
        query=args.query,
        top_k=args.top_k,
        cluster_id=args.cluster,
        keyword_mode=args.mode,
    )

    display_sem = [
        "cluster",
        "cluster_label",
        "name_en",
        "name_zh",
        "brand",
        "price",
        "score",
    ]
    display_kw = [
        "cluster",
        "cluster_label",
        "name_en",
        "name_zh",
        "brand",
        "price",
        "keyword_score",
    ]

    print(f"\n{'=' * 65}")
    print(f"  Keyword: '{res['keyword']}'  |  Query: '{res['query']}'")
    print(f"{'=' * 65}")

    print(f"\n  SEMANTIC ({len(res['semantic_results'])} results):\n")
    sem = res["semantic_results"]
    if not sem.empty:
        print(
            sem[[c for c in display_sem if c in sem.columns]].to_string(
                index=False, max_colwidth=38
            )
        )
    else:
        print("  (no results)")

    print(f"\n  KEYWORD TF-IDF ({len(res['keyword_results'])} results):\n")
    kw = res["keyword_results"]
    if not kw.empty:
        print(
            kw[[c for c in display_kw if c in kw.columns]].to_string(
                index=False, max_colwidth=38
            )
        )
    else:
        print("  (no results)")

    print(
        f"\n  Cluster overlap: {res['cluster_overlap_ratio']:.0%}  "
        f"(clusters {res['overlap_clusters']})"
    )
