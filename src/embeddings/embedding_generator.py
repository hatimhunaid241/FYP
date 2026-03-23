import sys
from pathlib import Path

# Add project root to path so 'src' module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.preprocessing.text_cleaner import TextCleaner
from sentence_transformers import SentenceTransformer


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
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

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


if __name__ == '__main__':
    df = build_embedding_df()
    print('Sample cleaned + embedded rows:')
    print(df[['name_zh_clean', 'name_en_clean', 'description_zh_clean', 'description_en_clean', 'embedding_text']].head())