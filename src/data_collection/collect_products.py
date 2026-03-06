"""
Product Data Collection Script
Collects products from HKTVmall API for specified keywords
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

from api_client import HKTVmallAPIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductCollector:
    """Collects and stores product data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize collector with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_client = HKTVmallAPIClient(
            delay=self.config['data_collection']['api_delay_seconds']
        )
        
        self.raw_dir = Path(self.config['paths']['raw_dir'])
        self.processed_dir = Path(self.config['paths']['processed_dir'])
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.products = []
        self.product_ids_seen = set()
    
    def collect_for_keyword(self, keyword: str, target_count: int = 5000) -> List[Dict]:
        """
        Collect products for a single keyword
        
        Args:
            keyword: Search keyword
            target_count: Target number of products to collect
            
        Returns:
            List of product dictionaries
        """
        logger.info(f"Collecting products for keyword: '{keyword}'")
        
        # Get total available products
        total_available = self.api_client.get_total_results(keyword)
        logger.info(f"Total products available for '{keyword}': {total_available:,}")
        
        # Adjust target if necessary
        actual_target = min(target_count, total_available)
        
        products = []
        page = 0
        hits_per_page = 100  # Max results per page
        
        with tqdm(total=actual_target, desc=f"Collecting '{keyword}'") as pbar:
            while len(products) < actual_target:
                # Fetch page of results
                results = self.api_client.search_products(
                    keyword=keyword,
                    hits_per_page=hits_per_page,
                    page=page
                )
                
                if not results or 'hits' not in results:
                    logger.warning(f"No more results for '{keyword}' at page {page}")
                    break
                
                page_hits = results['hits']
                
                if not page_hits:
                    logger.info(f"Reached end of results for '{keyword}'")
                    break
                
                # Process each product
                for hit in page_hits:
                    # Extract and normalize product data
                    product = self.api_client.extract_product_data(hit)
                    
                    if not product or not product.get('product_id'):
                        continue
                    
                    product_id = product['product_id']
                    
                    # Skip duplicates
                    if product_id in self.product_ids_seen:
                        continue
                    
                    # Add metadata
                    product['keyword_source'] = keyword
                    product['collected_at'] = datetime.now().isoformat()
                    product['collection_page'] = page
                    
                    products.append(product)
                    self.product_ids_seen.add(product_id)
                    
                    # Update progress
                    pbar.update(1)
                    
                    if len(products) >= actual_target:
                        break
                
                # Save raw response (if configured)
                if self.config['data_collection']['save_raw_responses']:
                    raw_file = self.raw_dir / 'products' / f"{keyword}_page{page}.json"
                    raw_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(raw_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                
                # Move to next page
                page += 1
                
                # Safety check (Algolia max is 1000 pages)
                if page >= 1000:
                    logger.warning(f"Reached maximum page limit for '{keyword}'")
                    break
                
                # Check if we've collected enough
                if len(products) >= actual_target:
                    break
        
        logger.info(f"Collected {len(products)} products for '{keyword}'")
        return products
    
    def collect_all_keywords(self) -> pd.DataFrame:
        """
        Collect products for all configured keywords
        
        Returns:
            DataFrame of all collected products
        """
        keywords = self.config['keywords']['seed_keywords']
        products_per_keyword = self.config['data_collection']['products_per_keyword']
        
        logger.info(f"Collecting {products_per_keyword} products for {len(keywords)} keywords")
        logger.info(f"Target total: {products_per_keyword * len(keywords)} products")
        
        all_products = []
        
        for keyword in keywords:
            keyword_products = self.collect_for_keyword(keyword, products_per_keyword)
            all_products.extend(keyword_products)
            
            # Save checkpoint after each keyword
            self._save_checkpoint(all_products, f"checkpoint_{keyword}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_products)
        
        logger.info(f"Total products collected: {len(df)}")
        logger.info(f"Unique products: {df['id'].nunique() if 'id' in df.columns else len(df)}")
        
        return df
    
    def _save_checkpoint(self, products: List[Dict], name: str):
        """Save intermediate checkpoint"""
        checkpoint_file = self.processed_dir / f"{name}.json"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def save_products(self, df: pd.DataFrame):
        """
        Save collected products to disk
        
        Args:
            df: DataFrame of products
        """
        # Save as Parquet (efficient, preserves types)
        parquet_file = self.processed_dir / "products.parquet"
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved {len(df)} products to {parquet_file}")
        
        # Save as CSV (human-readable backup)
        csv_file = self.processed_dir / "products.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"Saved CSV backup to {csv_file}")
        
        # Save metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'total_products': len(df),
            'unique_products': df['product_id'].nunique() if 'product_id' in df.columns else len(df),
            'keywords': self.config['keywords']['seed_keywords'],
            'columns': list(df.columns),
            'product_counts_by_keyword': df['keyword_source'].value_counts().to_dict() if 'keyword_source' in df.columns else {},
            'language_distribution': {
                'has_english': int((df['name_en'].notna() & (df['name_en'] != '')).sum()) if 'name_en' in df.columns else 0,
                'has_chinese': int((df['name_zh'].notna() & (df['name_zh'] != '')).sum()) if 'name_zh' in df.columns else 0,
            },
            'price_stats': {
                'min': float(df['price'].min()) if 'price' in df.columns else 0,
                'max': float(df['price'].max()) if 'price' in df.columns else 0,
                'mean': float(df['price'].mean()) if 'price' in df.columns else 0,
            },
            'rating_stats': {
                'mean_rating': float(df['average_rating'].mean()) if 'average_rating' in df.columns else 0,
                'products_with_reviews': int((df['num_reviews'] > 0).sum()) if 'num_reviews' in df.columns else 0,
            }
        }
        
        metadata_file = self.processed_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("COLLECTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total products: {metadata['total_products']:,}")
        logger.info(f"Unique products: {metadata['unique_products']:,}")
        logger.info(f"Products with English names: {metadata['language_distribution']['has_english']:,}")
        logger.info(f"Products with Chinese names: {metadata['language_distribution']['has_chinese']:,}")
        logger.info(f"Price range: ${metadata['price_stats']['min']:.2f} - ${metadata['price_stats']['max']:.2f}")
        logger.info(f"Average rating: {metadata['rating_stats']['mean_rating']:.2f}")
        logger.info("="*60)


def main():
    """Main collection pipeline"""
    logger.info("="*50)
    logger.info("Starting HKTVmall Product Collection")
    logger.info("="*50)
    
    # Initialize collector
    collector = ProductCollector()
    
    # Collect products
    df = collector.collect_all_keywords()
    
    # Save results
    collector.save_products(df)
    
    logger.info("="*50)
    logger.info("Collection complete!")
    logger.info(f"Total products: {len(df)}")
    logger.info("="*50)
    
    # Display sample
    print("\nSample products:")
    print(df.head())
    
    print("\nData info:")
    print(df.info())


if __name__ == "__main__":
    main()
