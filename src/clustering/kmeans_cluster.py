import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def perform_kmeans_clustering(
    embeddings_path: str = 'data/processed/product_embeddings.npy',
    products_path: str = 'data/processed/products.csv',
    n_clusters: int = 10,
    random_state: int = 42,
    standardize: bool = False,
    output_path: str = 'data/processed/kmeans_clusters.csv',
    visualize: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform K-means clustering on product embeddings.

    Args:
        embeddings_path: Path to the .npy file containing embeddings
        products_path: Path to the CSV file containing product data
        n_clusters: Number of clusters to create
        random_state: Random state for reproducibility
        standardize: Whether to standardize embeddings before clustering
        output_path: Path to save cluster assignments
        visualize: Whether to create visualizations
        **kwargs: Additional parameters for KMeans

    Returns:
        Tuple of (clustered_data, metrics_dict)
    """
    # Load data
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)

    print(f"Loading product data from {products_path}")
    products = pd.read_csv(products_path)

    print(f"Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    print(f"Loaded {len(products)} products")

    # Standardize if requested
    if standardize:
        print("Standardizing embeddings...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings

    # Perform K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        **kwargs
    )

    cluster_labels = kmeans.fit_predict(embeddings_scaled)

    # Calculate evaluation metrics
    print("Calculating clustering metrics...")
    n_samples = len(embeddings_scaled)

    # Skip silhouette score for large datasets (> 5000 samples) as it's computationally expensive
    if n_samples > 5000:
        print(f"Skipping silhouette score calculation for large dataset ({n_samples} samples)")
        silhouette = None
    else:
        try:
            silhouette = silhouette_score(embeddings_scaled, cluster_labels)
        except Exception as e:
            print(f"Warning: Could not calculate silhouette score: {e}")
            silhouette = None

    try:
        ch_score = calinski_harabasz_score(embeddings_scaled, cluster_labels)
        db_score = davies_bouldin_score(embeddings_scaled, cluster_labels)
    except Exception as e:
        print(f"Warning: Could not calculate all metrics: {e}")
        ch_score = db_score = None

    # Create results DataFrame
    results_df = products.copy()
    results_df['cluster'] = cluster_labels
    results_df['cluster'] = results_df['cluster'].astype('category')

    # Add cluster centers info
    cluster_centers = kmeans.cluster_centers_

    # Calculate cluster sizes and statistics
    cluster_stats = results_df.groupby('cluster').agg({
        'keyword_source': lambda x: x.value_counts().index[0],  # Most common category
        'price': ['count', 'mean', 'std'],
        'average_rating': ['mean', 'std']
    }).round(2)

    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.rename(columns={
        'keyword_source_<lambda_0>': 'dominant_category',
        'price_count': 'size',
        'price_mean': 'avg_price',
        'price_std': 'price_std',
        'average_rating_mean': 'avg_rating',
        'average_rating_std': 'rating_std'
    })

    # Metrics dictionary
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'calinski_harabasz_score': ch_score,
        'davies_bouldin_score': db_score,
        'inertia': kmeans.inertia_,
        'cluster_sizes': results_df['cluster'].value_counts().sort_index().tolist(),
        'cluster_stats': cluster_stats.to_dict()
    }

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Saved clustered data to {output_path}")

    # Print summary
    print(f"\nClustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.3f}" if silhouette else "Silhouette Score: N/A")
    print(f"Calinski-Harabasz Score: {ch_score:.2f}" if ch_score else "Calinski-Harabasz Score: N/A")
    print(f"Davies-Bouldin Score: {db_score:.3f}" if db_score else "Davies-Bouldin Score: N/A")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"\nCluster sizes: {metrics['cluster_sizes']}")

    # Create visualizations if requested
    if visualize:
        create_clustering_visualizations(results_df, embeddings_scaled, cluster_labels, n_clusters)

    return results_df, metrics


def create_clustering_visualizations(
    data: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    output_dir: str = 'data/processed'
):
    """Create visualizations for clustering results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Cluster distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = data['cluster'].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    plt.title(f'Cluster Size Distribution (K={n_clusters})')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Products')
    plt.savefig(f'{output_dir}/cluster_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Price distribution by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='cluster', y='price')
    plt.title(f'Price Distribution by Cluster (K={n_clusters})')
    plt.xlabel('Cluster')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/price_by_cluster.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Rating distribution by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='cluster', y='average_rating')
    plt.title(f'Rating Distribution by Cluster (K={n_clusters})')
    plt.xlabel('Cluster')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/rating_by_cluster.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Category distribution by cluster (heatmap)
    if 'keyword_source' in data.columns:
        plt.figure(figsize=(12, 8))
        cluster_category = pd.crosstab(data['cluster'], data['keyword_source'], normalize='index')
        sns.heatmap(cluster_category, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Category Distribution by Cluster (K={n_clusters})')
        plt.xlabel('Category')
        plt.ylabel('Cluster')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/category_by_cluster.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {output_dir}/")


def find_optimal_k(
    embeddings_path: str = 'data/processed/product_embeddings.npy',
    k_range: range = range(2, 21),
    random_state: int = 42,
    standardize: bool = False,
    output_path: str = 'data/processed/kmeans_elbow.png'
) -> Dict[int, Dict[str, float]]:
    """
    Find optimal number of clusters using elbow method and silhouette analysis.

    Args:
        embeddings_path: Path to embeddings file
        k_range: Range of k values to test
        random_state: Random state for reproducibility
        standardize: Whether to standardize embeddings
        output_path: Path to save elbow plot

    Returns:
        Dictionary with metrics for each k
    """
    # Load embeddings
    embeddings = np.load(embeddings_path)
    if standardize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    results = {}

    print("Finding optimal k...")
    for k in k_range:
        print(f"Testing k={k}")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Calculate metrics
        try:
            if len(embeddings) <= 5000:
                silhouette = silhouette_score(embeddings, labels)
            else:
                silhouette = None
            ch_score = calinski_harabasz_score(embeddings, labels)
            db_score = davies_bouldin_score(embeddings, labels)
        except:
            silhouette = ch_score = db_score = None

        results[k] = {
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score
        }

    # Create elbow plot
    k_values = list(results.keys())
    inertias = [results[k]['inertia'] for k in k_values]
    silhouettes = [results[k]['silhouette_score'] for k in k_values if results[k]['silhouette_score'] is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow plot
    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)

    # Silhouette plot
    if silhouettes:
        ax2.plot(k_values[:len(silhouettes)], silhouettes, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Elbow plot saved to {output_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='K-means clustering on product embeddings')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters')
    parser.add_argument('--find_optimal', action='store_true', help='Find optimal k')
    parser.add_argument('--k_min', type=int, default=2, help='Minimum k for optimal search')
    parser.add_argument('--k_max', type=int, default=20, help='Maximum k for optimal search')
    parser.add_argument('--standardize', action='store_true', help='Standardize embeddings')
    parser.add_argument('--no_visualize', action='store_true', help='Skip visualizations')

    args = parser.parse_args()

    if args.find_optimal:
        print(f"Finding optimal k from {args.k_min} to {args.k_max}")
        results = find_optimal_k(
            k_range=range(args.k_min, args.k_max + 1),
            standardize=args.standardize
        )

        # Print results
        print("\nOptimal k analysis results:")
        print("k\tInertia\t\tSilhouette\tCH Score\tDB Score")
        print("-" * 60)
        for k, metrics in results.items():
            sil = f"{metrics['silhouette_score']:.3f}" if metrics['silhouette_score'] else "N/A"
            print(f"{k}\t{metrics['inertia']:.0f}\t\t{sil}\t\t{metrics['calinski_harabasz_score']:.0f}\t\t{metrics['davies_bouldin_score']:.3f}")

    else:
        print(f"Performing K-means clustering with {args.n_clusters} clusters")
        clustered_data, metrics = perform_kmeans_clustering(
            n_clusters=args.n_clusters,
            standardize=args.standardize,
            visualize=not args.no_visualize
        )

        print(f"\nClustering completed with {args.n_clusters} clusters!")

        # Print cluster statistics
        print("\nCluster Statistics:")
        cluster_stats = metrics['cluster_stats']
        for cluster_id, stats in cluster_stats.items():
            print(f"Cluster {cluster_id}: {stats['size']} products, dominant category: {stats['dominant_category']}, avg price: ${stats['avg_price']:.2f}")