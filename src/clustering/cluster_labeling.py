"""
Per-Keyword Cluster Labeling (TF-IDF)
=======================================
For each keyword, reads its kmeans_clusters.csv and generates a human-readable
label for every cluster using TF-IDF.  Saves:

    data/processed/<keyword>/clusters_labeled.csv   — full table + cluster_label
    data/processed/<keyword>/cluster_labels.csv     — compact id → label lookup

Run from the project root:
    python src/clustering/cluster_labeling.py
    python src/clustering/cluster_labeling.py --keywords apple milk
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import re
import warnings

warnings.filterwarnings("ignore")

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/cluster_labeling.log")


# ── Tokenisers ────────────────────────────────────────────────────────── #


def _zh_terms(raw: str) -> List[str]:
    only_zh = re.sub(r"[^\u4e00-\u9fff\s]", "", raw)
    return [t for t in jieba.cut(only_zh) if len(t.strip()) > 1]


def _en_terms(raw: str) -> List[str]:
    only_en = re.sub(r"[^a-zA-Z\s]", " ", raw).lower()
    return [t for t in only_en.split() if len(t) > 2]


def _mixed(raw: str) -> List[str]:
    return _zh_terms(raw) + _en_terms(raw)


# ── TF-IDF ────────────────────────────────────────────────────────────── #


def _get_top_tfidf_terms(
    cluster_texts: Dict[int, List[str]],
    top_n: int = 8,
    max_features: int = 1500,
    ngram_range: Tuple[int, int] = (1, 2),
    language: str = "mixed",
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Fit one TF-IDF model over the entire keyword corpus (so IDF is shared),
    then rank terms per cluster by mean TF-IDF within that cluster.
    """
    tokenizer = {"chinese": _zh_terms, "english": _en_terms}.get(language, _mixed)

    stop_words = list(sklearn_text.ENGLISH_STOP_WORDS) + [
        "的",
        "了",
        "和",
        "是",
        "就",
        "都",
        "而",
        "及",
        "与",
        "这",
        "那",
        "什么",
        "如何",
        "没有",
    ]

    all_docs, cluster_ids_for_docs = [], []
    for cid, docs in cluster_texts.items():
        all_docs.extend(docs)
        cluster_ids_for_docs.extend([cid] * len(docs))

    if not all_docs:
        return {cid: [] for cid in cluster_texts}

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        tokenizer=tokenizer,
        token_pattern=None,
        sublinear_tf=True,
        lowercase=(language != "chinese"),
    )
    try:
        mat = vec.fit_transform(all_docs)
    except Exception as exc:
        logger.error(f"TF-IDF failed: {exc}")
        return {cid: [] for cid in cluster_texts}

    features = vec.get_feature_names_out()
    cid_arr = np.array(cluster_ids_for_docs)

    result: Dict[int, List[Tuple[str, float]]] = {}
    for cid in sorted(cluster_texts):
        mask = cid_arr == cid
        if not mask.any():
            result[cid] = []
            continue
        mean_scores = np.asarray(mat[mask].mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[-top_n:][::-1]
        result[cid] = [(features[i], float(mean_scores[i])) for i in top_idx]
    return result


def _make_labels(
    top_terms: Dict[int, List[Tuple[str, float]]],
    max_terms: int = 4,
    threshold: float = 0.05,
) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for cid, terms in top_terms.items():
        filtered = [(t, s) for t, s in terms if s >= threshold] or terms[:max_terms]
        selected = filtered[:max_terms]
        labels[cid] = (
            " | ".join(t for t, _ in selected) if selected else f"Cluster {cid}"
        )
    return labels


# ── Per-keyword labeling ──────────────────────────────────────────────── #


def label_keyword(
    keyword: str,
    config: dict,
    text_columns: Optional[List[str]] = None,
    top_n_terms: int = 8,
    label_max_terms: int = 4,
    language: str = "mixed",
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Label clusters for one keyword and save results."""
    paths = keyword_paths(keyword, config)

    if not paths["clusters_csv"].exists():
        raise FileNotFoundError(
            f"Clusters not found for '{keyword}' at {paths['clusters_csv']}. "
            "Run kmeans_cluster.py first."
        )

    df = pd.read_csv(paths["clusters_csv"])
    logger.info(
        f"  [{keyword}] {len(df):,} products, {df['cluster'].nunique()} clusters"
    )

    # Choose text columns
    if text_columns is None:
        candidates = [
            "embedding_text",
            "name_zh_clean",
            "name_en_clean",
            "name_zh",
            "name_en",
            "description_zh_clean",
            "description_en_clean",
        ]
        text_columns = [c for c in candidates if c in df.columns]
    if not text_columns:
        raise ValueError(f"[{keyword}] No text columns found.")

    # Build per-cluster document list
    cluster_texts: Dict[int, List[str]] = {}
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        docs = []
        for _, row in sub.iterrows():
            parts = [str(row[c]) for c in text_columns if c in row and pd.notna(row[c])]
            if parts:
                docs.append(" ".join(parts))
        cluster_texts[int(cid)] = docs

    top_terms = _get_top_tfidf_terms(
        cluster_texts, top_n=top_n_terms, language=language
    )
    labels = _make_labels(top_terms, max_terms=label_max_terms)

    df_labeled = df.copy()
    df_labeled["cluster_label"] = df_labeled["cluster"].map(labels)

    # Save
    df_labeled.to_csv(paths["clusters_labeled_csv"], index=False, encoding="utf-8-sig")
    logger.info(f"  [{keyword}] Saved → {paths['clusters_labeled_csv']}")

    labels_df = pd.DataFrame(
        [
            {"cluster": cid, "label": lbl, "top_terms": str(top_terms.get(cid, [])[:4])}
            for cid, lbl in sorted(labels.items())
        ]
    )
    labels_df.to_csv(paths["cluster_labels_csv"], index=False, encoding="utf-8-sig")

    # Print
    for cid in sorted(labels):
        size = (df_labeled["cluster"] == cid).sum()
        logger.info(f"    Cluster {cid:>2d} ({size:>4,} products): {labels[cid]}")

    return df_labeled, labels


def label_all_keywords(
    keywords: list[str] | None = None,
    config_path: str = "config/config.yaml",
    force: bool = False,
) -> None:
    config = load_config(config_path)
    if keywords is None:
        keywords = config["keywords"]["seed_keywords"]

    logger.info("=" * 60)
    logger.info(f"Per-keyword labeling: {len(keywords)} keywords")
    logger.info("=" * 60)

    for kw in keywords:
        paths = keyword_paths(kw, config)
        if not force and paths["clusters_labeled_csv"].exists():
            logger.info(f"  '{kw}' already labeled — skipping")
            continue
        try:
            label_keyword(kw, config)
        except Exception as exc:
            logger.error(f"  '{kw}' failed: {exc}")

    logger.info("Labeling complete.")


# ── CLI ───────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-keyword cluster labeling")
    parser.add_argument("--keywords", nargs="+", default=None)
    parser.add_argument(
        "--language", choices=["mixed", "english", "chinese"], default="mixed"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    label_all_keywords(keywords=args.keywords, force=args.force)
