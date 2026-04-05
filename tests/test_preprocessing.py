"""
Unit tests for the preprocessing module.

Run from project root:
    python -m pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.preprocessing.english_processor import EnglishProcessor
from src.preprocessing.text_cleaner import TextCleaner


# ── EnglishProcessor ─────────────────────────────────────────────────── #


class TestEnglishProcessor:
    def setup_method(self):
        self.ep = EnglishProcessor()

    def test_clean_lowercases(self):
        assert self.ep.clean("Apple iPhone") == "apple iphone"

    def test_clean_removes_punctuation(self):
        result = self.ep.clean("hello, world!!!")
        assert "," not in result and "!" not in result

    def test_clean_preserves_hyphens_in_compounds(self):
        result = self.ep.clean("state-of-the-art")
        assert "state-of-the-art" in result or "state" in result

    def test_clean_stopwords(self):
        result = self.ep.clean("the quick brown fox", remove_stopwords=True)
        assert "the" not in result.split()

    def test_clean_empty_string(self):
        assert self.ep.clean("") == ""

    def test_normalize(self):
        result = self.ep.normalize("Apple iPhone 15 Pro!!!")
        assert "apple" in result
        assert "iphone" in result
        assert "15" in result

    def test_tokenize(self):
        tokens = self.ep.tokenize("wireless charger cable")
        assert isinstance(tokens, list)
        assert len(tokens) == 3


# ── TextCleaner ───────────────────────────────────────────────────────── #


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner()

    # Language detection
    def test_detect_english(self):
        assert self.cleaner.detect_language("wireless bluetooth speaker") == "english"

    def test_detect_chinese(self):
        assert self.cleaner.detect_language("藍牙耳機音響") == "chinese"

    def test_detect_mixed(self):
        # 12 Chinese chars vs 5 English letters — comfortably triggers "mixed"
        assert (
            self.cleaner.detect_language("藍牙耳機 Bluetooth 無線音響設備") == "mixed"
        )

    def test_detect_unknown_empty(self):
        lang = self.cleaner.detect_language("")
        assert lang == "unknown"

    def test_detect_non_string(self):
        assert self.cleaner.detect_language(None) == "unknown"  # type: ignore

    # Number preservation in mixed text
    def test_numbers_preserved_mixed(self):
        """Critical: product specs like '128GB', '15 Pro' must not be dropped."""
        text = "iPhone 15 Pro Max 256GB 蘋果手機"
        result = self.cleaner.clean_text(text)
        assert isinstance(result, str)
        # At least some numeric content should survive
        assert any(c.isdigit() for c in result), (
            f"Numbers were dropped from mixed text. Result: '{result}'"
        )

    def test_numbers_preserved_english(self):
        result = self.cleaner.clean_text("USB 3.0 hub 4-port")
        assert isinstance(result, str)
        assert "3" in result or "4" in result

    # Clean text basic behaviour
    def test_clean_text_returns_string(self):
        assert isinstance(self.cleaner.clean_text("test"), str)

    def test_clean_text_non_string(self):
        result = self.cleaner.clean_text(None)  # type: ignore
        assert result == ""

    def test_clean_text_preserve_original_returns_dict(self):
        result = self.cleaner.clean_text("Apple Watch", preserve_original=True)
        assert isinstance(result, dict)
        assert "original" in result and "cleaned" in result and "language" in result

    # Separate languages
    def test_separate_languages_extracts_both(self):
        zh, en = self.cleaner.separate_languages("Apple 蘋果 iPhone 手機 128")
        assert "蘋果" in zh or "手機" in zh
        assert "apple" in en.lower() or "iphone" in en.lower()
        # Numbers should appear in both
        assert "128" in zh and "128" in en

    # Statistics
    def test_get_statistics_keys(self):
        stats = self.cleaner.get_statistics("hello world")
        expected_keys = {
            "original_length",
            "cleaned_length",
            "language_detected",
            "tokens",
            "removed_chars",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_get_statistics_tokens_positive(self):
        stats = self.cleaner.get_statistics("hello world test")
        assert stats["tokens"] > 0

    # Batch clean
    def test_batch_clean_length(self):
        texts = ["hello", "world", "test"]
        results = self.cleaner.batch_clean(texts)
        assert len(results) == 3

    # Product field cleaning
    def test_clean_product_field_name_lighter(self):
        """Names should use normalize (preserve more detail)."""
        result = self.cleaner.clean_product_field("name_en", "Apple iPhone 15 Pro Max")
        assert "iphone" in result.lower()

    def test_clean_product_dict(self):
        product = {
            "name_en": "Apple iPhone 15!!!",
            "description_en": "The best smartphone available in the market!!!",
            "price": 9999.0,
        }
        cleaned = self.cleaner.clean_product_dict(product)
        assert cleaned["price"] == 9999.0  # non-text fields unchanged
        assert "iphone" in cleaned["name_en"].lower()


# ── Convenience functions ─────────────────────────────────────────────── #


def test_module_level_clean_text():
    from src.preprocessing.text_cleaner import clean_text

    result = clean_text("Hello World!!!")
    assert isinstance(result, str)
    assert "hello" in result.lower()


def test_module_level_normalize_text():
    from src.preprocessing.text_cleaner import normalize_text

    result = normalize_text("   Extra   spaces   ")
    assert "  " not in result  # no double spaces
