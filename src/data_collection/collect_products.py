"""
Product Data Collection — Per-Keyword
=======================================
Collects products from the HKTVmall Algolia API for every keyword listed in
config/config.yaml and saves each keyword's results into its own directory:

    data/processed/<keyword>/products.parquet
    data/processed/<keyword>/products.csv

This per-keyword layout is the foundation of the project:
  Each keyword (e.g. "apple") returns ~1 000 mixed products from HKTVmall.
  The subsequent pipeline clusters THOSE results independently so that users
  see organised groups (Apple iPhone / Apple Pencil / Apple Fruit) rather
  than a random unsorted list.

Global deduplication is applied *within* each keyword's result set only —
the same physical product can appear under multiple keywords if HKTVmall
returns it for both.

Run from the project root:
    python src/data_collection/collect_products.py
    python src/data_collection/collect_products.py --keywords apple milk fan
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import datetime
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from src.data_collection.api_client import HKTVmallAPIClient
from src.utils.config_loader import load_config, keyword_paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/collect_products.log")


class ProductCollector:
    """Collect and save per-keyword product data from HKTVmall."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.api_client = HKTVmallAPIClient(
            delay=self.config["data_collection"]["api_delay_seconds"]
        )

    # ------------------------------------------------------------------ #

    def collect_for_keyword(self, keyword: str) -> pd.DataFrame:
        """
        Collect up to products_per_keyword unique products for *keyword*
        and save them to data/processed/<keyword>/products.parquet.

        Returns the DataFrame of collected products.
        """
        per_kw = self.config["data_collection"]["products_per_keyword"]
        paths = keyword_paths(keyword, self.config)

        logger.info(f"  Keyword: '{keyword}'  (target: {per_kw} products)")

        total_available = self.api_client.get_total_results(keyword)
        logger.info(f"  API reports {total_available:,} products available")

        actual_target = min(per_kw, total_available)
        products: List[Dict] = []
        seen_ids: set = set()  # dedup within this keyword
        page = 0
        hits_per_page = 1000

        with tqdm(total=actual_target, desc=f"  '{keyword}'", unit="prod") as pbar:
            while len(products) < actual_target:
                results = self.api_client.search_products(
                    keyword=keyword, hits_per_page=hits_per_page, page=page
                )
                if not results or not results.get("hits"):
                    break

                for hit in results["hits"]:
                    product = self.api_client.extract_product_data(hit)
                    if not product or not product.get("product_id"):
                        continue
                    pid = product["product_id"]
                    if pid in seen_ids:
                        continue
                    product["keyword_source"] = keyword
                    product["collected_at"] = datetime.now().isoformat()
                    product["collection_page"] = page
                    products.append(product)
                    seen_ids.add(pid)
                    pbar.update(1)
                    if len(products) >= actual_target:
                        break

                if self.config["data_collection"]["save_raw_responses"]:
                    raw_dir = (
                        Path(self.config["paths"]["raw_dir"]) / "products" / keyword
                    )
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    with open(
                        raw_dir / f"page{page}.json", "w", encoding="utf-8"
                    ) as fh:
                        json.dump(results, fh, ensure_ascii=False, indent=2)

                page += 1
                if page >= 100:
                    logger.warning(f"  Reached page limit for '{keyword}'")
                    break

        df = pd.DataFrame(products)
        logger.info(f"  Collected {len(df):,} products for '{keyword}'")

        # Save per-keyword parquet + CSV
        df.to_parquet(paths["products_parquet"], index=False)
        df.to_csv(
            paths["products_parquet"].with_suffix(".csv"),
            index=False,
            encoding="utf-8-sig",
        )
        logger.info(f"  Saved → {paths['products_parquet']}")

        # Save compact metadata
        meta = {
            "keyword": keyword,
            "collection_date": datetime.now().isoformat(),
            "total_collected": len(df),
            "unique_product_ids": int(df["product_id"].nunique())
            if "product_id" in df.columns
            else len(df),
            "api_total_available": total_available,
            "has_english_name": int(
                (df["name_en"].notna() & (df["name_en"] != "")).sum()
            )
            if "name_en" in df.columns
            else 0,
            "has_chinese_name": int(
                (df["name_zh"].notna() & (df["name_zh"] != "")).sum()
            )
            if "name_zh" in df.columns
            else 0,
        }
        with open(paths["dir"] / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        return df

    def collect_all_keywords(
        self, keywords: List[str] | None = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect products for every keyword and return a dict of DataFrames.
        Already-collected keywords are skipped (resume-friendly).
        """
        if keywords is None:
            keywords = self.config["keywords"]["seed_keywords"]

        logger.info("=" * 60)
        logger.info(f"Per-keyword collection: {len(keywords)} keywords")
        logger.info("=" * 60)

        results: Dict[str, pd.DataFrame] = {}
        for kw in keywords:
            paths = keyword_paths(kw, self.config)
            if paths["products_parquet"].exists():
                logger.info(
                    f"  '{kw}' already collected — skipping (delete {paths['products_parquet']} to re-collect)"
                )
                results[kw] = pd.read_parquet(paths["products_parquet"])
                continue
            try:
                results[kw] = self.collect_for_keyword(kw)
            except Exception as exc:
                logger.error(f"  '{kw}' failed: {exc}")

        logger.info("=" * 60)
        logger.info("Collection complete.")
        for kw, df in results.items():
            logger.info(f"  {kw:20s}: {len(df):,} products")
        logger.info("=" * 60)
        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect HKTVmall products per keyword"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Specific keywords to collect (default: all from config)",
    )
    args = parser.parse_args()

    collector = ProductCollector()
    collector.collect_all_keywords(keywords=args.keywords)


if __name__ == "__main__":
    main()
