import sys
from pathlib import Path

# Add project root to path so 'src' module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.preprocessing.text_cleaner import TextCleaner
import numpy as np
import plotly.express as px
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def build_embedding_df(
    parquet_path: str = 'data/processed/products.parquet',
    model_name: str = 'sentence-transformers/all-mpnet-base-v2',
    out_data_path: str = 'data/processed/products_with_embeddings.parquet',
    out_embed_path: str = 'data/processed/product_embeddings.npy',
    keep_original_columns: bool = True,
) -> pd.DataFrame:
    """Load product data, clean text, create embeddings, and save outputs."""
    # Load raw products
    products = pd.read_parquet(parquet_path)

    # Keep target fields
    fields = ["name_zh", "name_en", "description_zh", "description_en"]
    for f in fields:
        if f not in products.columns:
            raise KeyError(f"Required column '{f}' is missing in {parquet_path}")

    products = products[fields].copy()

    # Initialize cleaner and embedding model
    cleaner = TextCleaner(remove_stopwords=True, convert_simplified=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    # Clean text fields
    products['name_zh_clean'] = products['name_zh'].fillna('').astype(str).map(lambda x: cleaner.clean_text(x))
    products['name_en_clean'] = products['name_en'].fillna('').astype(str).map(lambda x: cleaner.clean_text(x))
    products['description_zh_clean'] = products['description_zh'].fillna('').astype(str).map(lambda x: cleaner.clean_text(x))
    products['description_en_clean'] = products['description_en'].fillna('').astype(str).map(lambda x: cleaner.clean_text(x))

    # Combine for embedding; adjust mix strategy as needed
    products['embedding_text'] = (
        products['name_zh_clean'].fillna('') + ' ' +
        products['description_zh_clean'].fillna('') + ' ' +
        products['name_en_clean'].fillna('') + ' ' +
        products['description_en_clean'].fillna('')
    ).str.strip()

    # Generate embeddings
    texts = products['embedding_text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Save outputs (optionally with cleaned columns)
    if keep_original_columns:
        save_df = products.copy()
    else:
        save_df = products[['embedding_text']].copy()

    Path(out_data_path).parent.mkdir(parents=True, exist_ok=True)
    save_df.to_parquet(out_data_path, index=False)

    # Save embedding matrix separately
    Path(out_embed_path).parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(out_embed_path, embeddings)

    print(f"Saved cleaned product data to {out_data_path}")
    print(f"Saved embeddings matrix to {out_embed_path}")

    return products


def visualize_embeddings_2d(
    embeddings_path: str = 'data/processed/product_embeddings.npy',
    products_path: str = 'data/processed/products.csv',
    method: str = 'umap',  # 'umap' or 'tsne'
    n_components: int = 2,
    random_state: int = 42,
    sample_size: int = None,
    plot_library: str = 'plotly',  # 'plotly' or 'matplotlib'
    output_path: str = None,
    **kwargs
) -> None:
    """
    Visualize embeddings in 2D using UMAP or t-SNE.

    Args:
        embeddings_path: Path to the .npy file containing embeddings
        products_path: Path to the CSV file containing product data
        method: Dimensionality reduction method ('umap' or 'tsne')
        n_components: Number of components (should be 2 for 2D visualization)
        random_state: Random state for reproducibility
        sample_size: If provided, randomly sample this many points for faster visualization
        plot_library: Plotting library to use ('plotly' for interactive, 'matplotlib' for static)
        output_path: If provided, save the plot to this path (HTML for plotly, PNG for matplotlib)
        **kwargs: Additional parameters for the reduction method
    """
    # Load embeddings and product data
    embeddings = np.load(embeddings_path)
    products = pd.read_csv(products_path)

    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    print(f"Loaded {len(products)} products")

    # Sample if requested
    if sample_size and sample_size < len(embeddings):
        np.random.seed(random_state)
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[indices]
        products = products.iloc[indices].reset_index(drop=True)
        print(f"Sampled {sample_size} points for visualization")

    # Apply dimensionality reduction
    if method.lower() == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        method_name = 'UMAP'
    elif method.lower() == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        method_name = 't-SNE'
    else:
        raise ValueError("Method must be 'umap' or 'tsne'")

    print(f"Applied {method_name} reduction to 2D")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'category': products['keyword_source'] if 'keyword_source' in products.columns else 'unknown',
        'name': products['name_en'].fillna(products['name_zh']).fillna('Unknown')
    })

    if plot_library.lower() == 'plotly':
        # Create interactive scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='category',
            hover_data=['name'],
            title=f'Product Embeddings 2D Visualization ({method_name})',
            labels={'x': f'{method_name} Component 1', 'y': f'{method_name} Component 2'}
        )

        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(
            width=1000,
            height=800,
            hovermode='closest'
        )

        # Show the plot
        fig.show()

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path)
            print(f"Saved interactive visualization to {output_path}")

    elif plot_library.lower() == 'matplotlib':
        # Create static scatter plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=plot_df,
            x='x',
            y='y',
            hue='category',
            palette='tab10',
            alpha=0.7,
            s=50
        )
        plt.title(f'Product Embeddings 2D Visualization ({method_name})')
        plt.xlabel(f'{method_name} Component 1')
        plt.ylabel(f'{method_name} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved static visualization to {output_path}")
    else:
        raise ValueError("plot_library must be 'plotly' or 'matplotlib'")


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'visualize':
        # Visualize embeddings
        print("Creating 2D visualizations of embeddings...")
        visualize_embeddings_2d(
            method='umap',
            sample_size=1000,
            plot_library='plotly',
            output_path='data/processed/embeddings_umap.html'
        )
        visualize_embeddings_2d(
            method='tsne',
            sample_size=1000,
            plot_library='plotly',
            output_path='data/processed/embeddings_tsne.html'
        )
        print("Visualizations saved!")
    else:
        # Generate embeddings
        df = build_embedding_df()
        print('Sample cleaned + embedded rows:')
        print(df[['name_zh_clean', 'name_en_clean', 'description_zh_clean', 'description_en_clean', 'embedding_text']].head())