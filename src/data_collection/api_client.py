"""
HKTVmall API Client
Wrapper for interacting with HKTVmall search API (Algolia)
"""

import requests
import time
import json
import urllib.parse
from typing import Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HKTVmallAPIClient:
    """Client for HKTVmall search API (powered by Algolia)"""
    
    def __init__(self, delay: float = 0.5):
        """
        Initialize API client
        
        Args:
            delay: Delay between requests in seconds (respect rate limits)
        """
        self.delay = delay
        
        # Algolia API configuration (from HKTVmall)
        self.base_url = "https://8rn1y79f02-dsn.algolia.net/1/indexes/*/queries"
        self.app_id = "8RN1Y79F02"
        self.api_key = "a4a336abc62ab842842a81de642b484a"
        self.index_name = "hktvProduct"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json'
        })
    
    def search_products(self, keyword: str, hits_per_page: int = 1000, page: int = 0) -> Dict:
        """
        Search for products by keyword using Algolia API
        
        Args:
            keyword: Search keyword
            hits_per_page: Number of results per page (max 1000)
            page: Page number (0-indexed)
            
        Returns:
            Dictionary containing product search results
        """
        # Build query parameters for URL
        url_params = {
            'x-algolia-agent': 'Algolia for JavaScript (3.33.0); Browser',
            'x-algolia-application-id': self.app_id,
            'x-algolia-api-key': self.api_key,
            'x-algolia-usertoken': 'anonymous-search'
        }
        
        # Build search parameters (these go in the request body)
        search_params = f"query={keyword}&filters=&facets=%5B%22%22%5D&attributesToRetrieve=%5B%22*%22%5D&hitsPerPage={hits_per_page}&maxValuesPerFacet=1000&page={page}"
        
        # Request body
        body = {
            "requests": [
                {
                    "indexName": self.index_name,
                    "params": search_params
                }
            ]
        }
        
        try:
            response = self.session.post(
                self.base_url,
                params=url_params,
                json=body,
                timeout=30
            )
            response.raise_for_status()
            
            # Respect rate limiting
            time.sleep(self.delay)
            
            data = response.json()
            
            # Extract results from Algolia response format
            if data and 'results' in data and len(data['results']) > 0:
                return data['results'][0]
            else:
                logger.warning(f"No results for keyword '{keyword}'")
                return {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching for '{keyword}': {e}")
            return {}
    
    def get_total_results(self, keyword: str) -> int:
        """
        Get total number of results for a keyword
        
        Args:
            keyword: Search keyword
            
        Returns:
            Total number of hits
        """
        result = self.search_products(keyword, hits_per_page=1, page=0)
        return result.get('nbHits', 0)
    
    def extract_product_data(self, hit: Dict) -> Dict:
        """
        Extract and normalize product data from Algolia hit
        
        Args:
            hit: Product hit from Algolia response
            
        Returns:
            Normalized product dictionary
        """
        try:
            # Extract price information
            selling_price = hit.get('sellingPrice', 0)
            price_list = hit.get('priceList', [])
            buy_price = next((p['value'] for p in price_list if p['priceType'] == 'BUY'), selling_price)
            
            # Extract basic product info
            product = {
                'product_id': hit.get('code', ''),
                'base_product': hit.get('baseProduct', ''),
                'product_search_code': hit.get('productSearchCode', ''),
                
                # Multilingual names
                'name_zh': hit.get('nameZh', ''),
                'name_en': hit.get('nameEn', ''),
                'name_zhcn': hit.get('nameZhCN', ''),
                
                # Descriptions
                'description_zh': hit.get('summaryZh', ''),
                'description_en': hit.get('summaryEn', ''),
                'description_zhcn': hit.get('summaryZhCN', ''),
                
                # Brand
                'brand': hit.get('brand', ''),
                'brand_zh': hit.get('brandZh', ''),
                'brand_en': hit.get('brandEn', ''),
                
                # Category
                'primary_category': hit.get('primaryCatCode', ''),
                'main_category_zh': hit.get('mainCatNameZh', [''])[0] if hit.get('mainCatNameZh') else '',
                'main_category_en': hit.get('mainCatNameEn', [''])[0] if hit.get('mainCatNameEn') else '',
                
                # Pricing
                'price': selling_price,
                'original_price': buy_price,
                'price_range': hit.get('sellingPriceRange', ''),
                
                # Ratings & Reviews
                'average_rating': hit.get('averageRating', 0),
                'num_reviews': hit.get('numberOfReviews', 0),
                'rating_count': hit.get('rating_count', 0),
                
                # Store info
                'store_code': hit.get('storeCode', ''),
                'store_name_zh': hit.get('storeNameZh', ''),
                'store_name_en': hit.get('storeNameEn', ''),
                'store_rating': hit.get('storeRating', 0),
                'store_type': hit.get('storeType', ''),
                
                # Stock & availability
                'has_stock': hit.get('hasStock', False),
                'stock_status': hit.get('stock', {}).get('stockLevelStatus', {}).get('code', ''),
                
                # Other
                'country_of_origin': hit.get('countryOfOrigin', ''),
                'sales_volume': hit.get('salesVolume', 0),
                'loyalty_point': hit.get('loyaltyPoint', 0),
                
                # URLs
                'url_zh': hit.get('urlZh', ''),
                'url_en': hit.get('urlEn', ''),
                
                # Images
                'images': hit.get('images', []),
                'primary_image': hit.get('images', [{}])[0].get('url', '') if hit.get('images') else ''
            }
            
            return product
            
        except Exception as e:
            logger.error(f"Error extracting product data: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing HKTVmall API Client")
    logger.info("="*50)
    
    # Initialize client
    client = HKTVmallAPIClient(delay=0.5)
    
    # Test with a simple search
    keyword = "apple"
    logger.info(f"Searching for '{keyword}'...")
    
    results = client.search_products(keyword, hits_per_page=10, page=0)
    
    if results and 'hits' in results:
        hits = results['hits']
        total_hits = results.get('nbHits', 0)
        
        logger.info(f"✓ Found {total_hits} total products for '{keyword}'")
        logger.info(f"✓ Retrieved {len(hits)} products in this page")
        
        # Extract and display first product
        if hits:
            first_product = client.extract_product_data(hits[0])
            
            logger.info("\nFirst product details:")
            logger.info(f"  ID: {first_product['product_id']}")
            logger.info(f"  Name (EN): {first_product['name_en']}")
            logger.info(f"  Name (ZH): {first_product['name_zh']}")
            logger.info(f"  Brand: {first_product['brand']}")
            logger.info(f"  Price: ${first_product['price']}")
            logger.info(f"  Rating: {first_product['average_rating']} ({first_product['num_reviews']} reviews)")
            logger.info(f"  Category: {first_product['main_category_en']}")
        
        # Save full sample response for inspection
        sample_file = Path("data/raw/sample_response.json")
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✓ Full response saved to {sample_file}")
        
        # Save extracted products
        extracted_file = Path("data/raw/sample_extracted.json")
        extracted_products = [client.extract_product_data(hit) for hit in hits]
        
        with open(extracted_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_products, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Extracted products saved to {extracted_file}")
        
    else:
        logger.error("✗ No results returned - API might be down or configuration incorrect")
    
    logger.info("="*50)
    logger.info("Test complete!")

