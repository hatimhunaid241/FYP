import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Add project root to path so 'src' module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_search_data(
    clustered_csv: str = 'data/processed/clusters_labeled.csv',
    embeddings_npy: str = 'data/processed/product_embeddings.npy'
) -> Dict[str, Any]:
    """Load data required for semantic search: clustered products and embeddings."""
    if not Path(clustered_csv).exists():
        raise FileNotFoundError(f"Clustered CSV not found at {clustered_csv}. Generate with cluster_labeling.py first.")
    if not Path(embeddings_npy).exists():
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_npy}. Generate with embedding_generator.py first.")

    df = pd.read_csv(clustered_csv)
    embeddings = np.load(embeddings_npy)

    if len(df) != len(embeddings):
        raise ValueError(f"Length mismatch: {len(df)} rows in CSV vs {len(embeddings)} embeddings")

    # Ensure normed embeddings for cosine search
    embeddings_normed = normalize(embeddings, norm='l2', axis=1)

    return {
        'df': df,
        'embeddings': embeddings_normed,
    }


def build_search_model(model_name: str = 'sentence-transformers/all-mpnet-base-v2') -> SentenceTransformer:
    """Load SentenceTransformer embedding model for queries."""
    return SentenceTransformer(model_name)


def query_semantic_search(
    query: str,
    data: Dict[str, Any],
    model: SentenceTransformer,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    cluster_label_boost: bool = True,
    cluster_label_weight: float = 0.2,
    query_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Perform semantic search on product embeddings with optional cluster filters."""
    df = data['df'].copy()
    embeddings = data['embeddings']

    # Apply filters if requested
    if filters:
        for col, value in filters.items():
            if col not in df.columns:
                raise ValueError(f"Filter column '{col}' not found in DataFrame")
            if isinstance(value, list):
                df = df[df[col].isin(value)]
            else:
                df = df[df[col] == value]

        if df.empty:
            return pd.DataFrame(columns=['query', 'score'])

        idx_map = np.array(df.index)
        embeddings = embeddings[idx_map]

    # encode query
    query_embed = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # cosine similarity by dot product (normalized vectors)
    scores = (embeddings @ query_embed.T).squeeze()

    # optional cluster label boosting
    if cluster_label_boost and 'cluster_label' in df.columns:
        cluster_label_scores = []
        for label in df['cluster_label'].fillna(''):
            cluster_label_scores.append(1.0 if query.lower() in str(label).lower() else 0.0)

        cluster_label_scores = np.array(cluster_label_scores)
        scores = scores + cluster_label_weight * cluster_label_scores

    df = df.copy()
    df['score'] = scores

    # restrict to matching text columns if needed
    if query_columns:
        df['combined_text'] = df[query_columns].astype(str).apply(' '.join, axis=1)

    # return top K results
    result = df.sort_values('score', ascending=False).head(top_k)
    result = result.reset_index(drop=True)

    return result


def interactive_search(
    clustered_csv: str = 'data/processed/clusters_labeled.csv',
    embeddings_npy: str = 'data/processed/product_embeddings.npy',
    model_name: str = 'sentence-transformers/all-mpnet-base-v2',
    top_k: int = 10
):
    data = load_search_data(clustered_csv, embeddings_npy)
    model = build_search_model(model_name)

    print('Entering interactive semantic search. Type q or quit to exit.')

    while True:
        query = input('\nEnter search query: ').strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break

        results = query_semantic_search(query, data, model, top_k=top_k, filters=None)

        if results.empty:
            print('No matches. Try another query.')
            continue

        print(f"Top {len(results)} results for '{query}':")
        display_cols = ['cluster', 'cluster_label', 'name_en', 'name_zh', 'description_en', 'description_zh', 'score']
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=False, max_cols=len(available_cols)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Semantic search over clustered product embeddings')
    parser.add_argument('--query', type=str, default=None, help='Optional single query to run')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to return')
    parser.add_argument('--clustered_csv', type=str, default='data/processed/clusters_labeled.csv', help='Path to labeled cluster CSV')
    parser.add_argument('--embeddings_npy', type=str, default='data/processed/product_embeddings.npy', help='Path to embeddings numpy file')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2', help='SentenceTransformer model name')
    parser.add_argument('--cluster', type=int, default=None, help='Filter results to a specific cluster')

    args = parser.parse_args()

    data = load_search_data(args.clustered_csv, args.embeddings_npy)
    model = build_search_model(args.model_name)

    if args.query:
        filters = {'cluster': args.cluster} if args.cluster is not None else None
        results = query_semantic_search(args.query, data, model, top_k=args.top_k, filters=filters)
        print(f"Top {len(results)} results for '{args.query}':")
        display_cols = ['cluster', 'cluster_label', 'name_en', 'name_zh', 'score']
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=False, max_cols=len(available_cols)))
    else:
        interactive_search(args.clustered_csv, args.embeddings_npy, args.model_name, top_k=args.top_k)
