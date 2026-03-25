import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.search.semantic_search import load_search_data, build_search_model, query_semantic_search


def load_keyword_data(clustered_csv: str = 'data/processed/clusters_labeled.csv') -> pd.DataFrame:
    """Load labeled product data for keyword baseline search."""
    if not Path(clustered_csv).exists():
        raise FileNotFoundError(f"Clustered data not found at {clustered_csv}. Run cluster_labeling.py first.")

    df = pd.read_csv(clustered_csv)
    return df


def prepare_keyword_field(df: pd.DataFrame, fields: Optional[List[str]] = None) -> pd.DataFrame:
    """Combine selected fields into a single keyword-search text column."""
    if fields is None:
        fields = ['name_en', 'name_zh', 'description_en', 'description_zh', 'cluster_label']

    fields = [f for f in fields if f in df.columns]
    if not fields:
        raise ValueError('No text fields found for keyword search.')

    combined = df[fields].fillna('').astype(str).agg(' '.join, axis=1)
    df = df.copy()
    df['keyword_text'] = combined.str.lower()

    # Tokenized terms for exact matching or scoring
    df['keyword_terms'] = df['keyword_text'].str.split('\s+')
    return df


def keyword_search(
    query: str,
    df: pd.DataFrame,
    top_k: int = 10,
    fields: Optional[List[str]] = None,
    mode: str = 'term_frequency'
) -> pd.DataFrame:
    """Perform keyword-based retrieval as baseline.

    mode options:
      - contains: simple substring match, score = count of matched substrings
      - term_frequency: count query term matches in keyword_tokens
      - exact: exact phrase contained
    """
    query = str(query).strip().lower()
    if not query:
        return pd.DataFrame()

    df = prepare_keyword_field(df, fields)

    query_tokens = [t for t in query.split() if t]
    if not query_tokens:
        return pd.DataFrame()

    scores = []

    for _, row in df.iterrows():
        text = row['keyword_text']
        if mode == 'exact':
            score = 1.0 if query in text else 0.0
        elif mode == 'contains':
            score = sum(text.count(token) for token in query_tokens)
        else:  # term_frequency
            tokens = row['keyword_terms'] if isinstance(row['keyword_terms'], list) else []
            score = sum(tokens.count(token) for token in query_tokens)

        scores.append(score)

    df = df.copy()
    df['keyword_score'] = np.array(scores, dtype=float)
    df = df[df['keyword_score'] > 0]
    if df.empty:
        return df

    df = df.sort_values('keyword_score', ascending=False).head(top_k)
    return df


def compare_search_results(
    query: str,
    k: int = 10,
    semantic_args: Optional[Dict[str, Any]] = None,
    keyword_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run both semantic and keyword search and return comparable result sets."""
    semantic_args = semantic_args or {}
    keyword_args = keyword_args or {}

    data = load_search_data()
    model = build_search_model()

    semantic_df = query_semantic_search(query, data, model, top_k=k, **semantic_args)
    keyword_df = keyword_search(query, load_keyword_data(), top_k=k, **keyword_args)

    # If keyword cluster labels exist, we can compute overlap with semantic top clusters
    semantic_clusters = set(semantic_df['cluster'].tolist()) if 'cluster' in semantic_df.columns else set()
    keyword_clusters = set(keyword_df['cluster'].tolist()) if 'cluster' in keyword_df.columns else set()

    overlap_clusters = sorted(list(semantic_clusters.intersection(keyword_clusters)))
    overlap_score = len(overlap_clusters) / max(1, min(len(semantic_clusters), len(keyword_clusters)))

    return {
        'query': query,
        'semantic_results': semantic_df,
        'keyword_results': keyword_df,
        'semantic_clusters': semantic_clusters,
        'keyword_clusters': keyword_clusters,
        'overlap_clusters': overlap_clusters,
        'cluster_overlap_ratio': overlap_score,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Baseline keyword search vs semantic search comparison')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to show')
    parser.add_argument('--mode', type=str, choices=['term_frequency', 'contains', 'exact'], default='term_frequency', help='Keyword match mode')
    parser.add_argument('--fields', nargs='*', default=None, help='Text fields for keyword search')
    parser.add_argument('--cluster', type=int, default=None, help='Optional filter on cluster for both searches')

    args = parser.parse_args()

    keyword_args = {'mode': args.mode, 'fields': args.fields}
    semantic_filters = {'cluster': args.cluster} if args.cluster is not None else None

    results = compare_search_results(
        args.query,
        k=args.top_k,
        semantic_args={'filters': semantic_filters},
        keyword_args=keyword_args
    )

    print(f"\nQuery: {args.query}")
    print(f"Top {args.top_k} Semantic Search Results:\n")
    if results['semantic_results'].empty:
        print('No semantic results')
    else:
        print(results['semantic_results'][['cluster', 'cluster_label', 'name_en', 'score']].to_string(index=False, max_cols=4))

    print(f"\nTop {args.top_k} Keyword Search Results (mode={args.mode}):\n")
    if results['keyword_results'].empty:
        print('No keyword results')
    else:
        print(results['keyword_results'][['cluster', 'cluster_label', 'name_en', 'keyword_score']].to_string(index=False, max_cols=4))

    print(f"\nCluster overlap (semantic vs keyword): {results['cluster_overlap_ratio']:.2f}")
    print(f"Overlapping clusters: {results['overlap_clusters']}")
