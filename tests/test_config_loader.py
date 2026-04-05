"""
Unit tests for src/utils/config_loader.py

Run from the project root:
    python -m pytest tests/test_config_loader.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.utils.config_loader import load_config, keyword_slug, keyword_paths


class TestKeywordSlug:
    def test_simple(self):
        assert keyword_slug("apple") == "apple"

    def test_space_to_underscore(self):
        assert keyword_slug("wireless charger") == "wireless_charger"

    def test_uppercase(self):
        assert keyword_slug("Apple") == "apple"

    def test_special_chars(self):
        slug = keyword_slug("USB 3.0")
        assert " " not in slug
        assert slug.startswith("usb")

    def test_chinese_preserved(self):
        slug = keyword_slug("蘋果")
        assert "蘋果" in slug

    def test_no_leading_trailing_underscore(self):
        slug = keyword_slug("  apple  ")
        assert not slug.startswith("_")
        assert not slug.endswith("_")


class TestLoadConfig:
    def test_loads_successfully(self):
        config = load_config("config/config.yaml")
        assert isinstance(config, dict)

    def test_has_keywords(self):
        config = load_config("config/config.yaml")
        assert "keywords" in config
        assert "seed_keywords" in config["keywords"]
        assert len(config["keywords"]["seed_keywords"]) >= 20

    def test_has_embeddings_model(self):
        config = load_config("config/config.yaml")
        model = config["embeddings"]["model_name"]
        assert "multilingual" in model.lower(), (
            f"Expected a multilingual model, got: {model}"
        )

    def test_has_per_keyword_path_keys(self):
        config = load_config("config/config.yaml")
        required = [
            "kw_products_parquet",
            "kw_embeddings_npy",
            "kw_clusters_csv",
            "kw_clusters_labeled_csv",
        ]
        for key in required:
            assert key in config["paths"], f"Missing path key: {key}"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("config/does_not_exist.yaml")


class TestKeywordPaths:
    def setup_method(self):
        self.config = load_config("config/config.yaml")

    def test_returns_dict(self):
        paths = keyword_paths("apple", self.config)
        assert isinstance(paths, dict)

    def test_all_keys_present(self):
        paths = keyword_paths("apple", self.config)
        expected = [
            "dir",
            "products_parquet",
            "embeddings_npy",
            "embeddings_parquet",
            "clusters_csv",
            "clusters_labeled_csv",
            "cluster_labels_csv",
        ]
        for key in expected:
            assert key in paths, f"Missing key: {key}"

    def test_paths_under_keyword_subdir(self):
        paths = keyword_paths("apple", self.config)
        assert "apple" in str(paths["products_parquet"])

    def test_different_keywords_have_different_dirs(self):
        p1 = keyword_paths("apple", self.config)
        p2 = keyword_paths("milk", self.config)
        assert p1["dir"] != p2["dir"]

    def test_creates_directory(self):
        paths = keyword_paths("apple", self.config)
        assert paths["dir"].exists()

    def test_chinese_keyword_slug(self):
        paths = keyword_paths("蘋果", self.config)
        assert paths["dir"].exists()
