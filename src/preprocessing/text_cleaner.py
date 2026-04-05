"""
Main text cleaner for bilingual (English and Chinese) product text.

Key design decisions
--------------------
* Numbers are *always* preserved — product specs like "128GB", "15 Pro", "5G"
  are semantically critical and must not be dropped.
* Mixed-language text is split by character class but numbers are kept in
  *both* the Chinese and English parts so that the downstream combiner
  always retains them.
* `clean_text` always returns `str`; the `preserve_original` path returns
  a `dict` and the caller must annotate accordingly if type-checking matters.
"""

import re
import logging
from typing import Dict, List, Tuple, Union

from src.preprocessing.english_processor import EnglishProcessor
from src.preprocessing.chinese_processor import ChineseProcessor

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and preprocess bilingual product text (English and Chinese)."""

    def __init__(
        self,
        remove_stopwords: bool = False,
        convert_simplified: bool = True,
    ):
        """
        Args:
            remove_stopwords: Strip common stop-words from both languages.
            convert_simplified: Convert Traditional Chinese → Simplified Chinese.
        """
        self.english_processor = EnglishProcessor()
        self.chinese_processor = ChineseProcessor(
            convert_to_simplified=convert_simplified
        )
        self.remove_stopwords = remove_stopwords

    # ------------------------------------------------------------------ #
    # Language detection                                                   #
    # ------------------------------------------------------------------ #

    def detect_language(self, text: str) -> str:
        """Return 'english', 'chinese', or 'mixed' ('unknown' if no letters)."""
        if not isinstance(text, str):
            return "unknown"

        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_count = len(re.findall(r"[a-zA-Z]", text))

        if chinese_count == 0 and english_count == 0:
            return "unknown"
        if chinese_count > english_count * 2:
            return "chinese"
        if english_count > chinese_count * 2:
            return "english"
        return "mixed"

    # ------------------------------------------------------------------ #
    # Language separation                                                  #
    # ------------------------------------------------------------------ #

    def separate_languages(self, text: str) -> Tuple[str, str]:
        """
        Split mixed text into (chinese_part, english_part).

        Numbers (digits, e.g. "128", "5G", "15 Pro") are appended to
        *both* parts so that product specs are never silently discarded.
        """
        # Chinese characters (CJK Unified Ideographs)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        chinese_text = " ".join(chinese_chars)

        # ASCII letters only
        english_chars = re.findall(r"[a-zA-Z]+", text)
        english_text = " ".join(english_chars)

        # Standalone numeric tokens — could be specs like "128GB", "5G" etc.
        # We extract digit-containing tokens that are NOT pure ASCII words.
        numeric_tokens = re.findall(r"\b\d[\w]*\b", text)
        numeric_str = " ".join(numeric_tokens)

        # Append numeric tokens to both parts so they are always preserved
        if numeric_str:
            chinese_text = (chinese_text + " " + numeric_str).strip()
            english_text = (english_text + " " + numeric_str).strip()

        return chinese_text, english_text

    # ------------------------------------------------------------------ #
    # Core cleaning                                                        #
    # ------------------------------------------------------------------ #

    def clean_text(
        self, text: str, preserve_original: bool = False
    ) -> Union[str, Dict]:
        """
        Clean product text, applying language-appropriate processors.

        Args:
            text: Raw input text (may be English, Chinese, or mixed).
            preserve_original: If True, return a dict with keys
                ``'original'``, ``'cleaned'``, and ``'language'``.

        Returns:
            Cleaned string (default) or dict when *preserve_original* is True.
        """
        if not isinstance(text, str):
            empty: Union[str, Dict] = (
                {"original": "", "cleaned": "", "language": "unknown"}
                if preserve_original
                else ""
            )
            return empty

        original_text = text
        language_type = self.detect_language(text)

        if language_type == "english":
            cleaned = self.english_processor.clean(text, self.remove_stopwords)

        elif language_type == "chinese":
            cleaned = self.chinese_processor.clean(text, self.remove_stopwords)

        else:
            # Mixed or unknown — split and process each language separately,
            # then re-join preserving all content.
            chinese_text, english_text = self.separate_languages(text)

            cleaned_chinese = ""
            cleaned_english = ""

            if chinese_text.strip():
                cleaned_chinese = self.chinese_processor.clean(
                    chinese_text, self.remove_stopwords
                )
            if english_text.strip():
                cleaned_english = self.english_processor.clean(
                    english_text, self.remove_stopwords
                )

            parts = [p for p in [cleaned_chinese, cleaned_english] if p.strip()]
            cleaned = " ".join(parts)

        if preserve_original:
            return {
                "original": original_text,
                "cleaned": cleaned,
                "language": language_type,
            }
        return cleaned

    # ------------------------------------------------------------------ #
    # Normalisation (lighter than full cleaning)                          #
    # ------------------------------------------------------------------ #

    def normalize_text(self, text: str) -> str:
        """
        Lighter cleaning that preserves more of the original formatting.
        Suitable for product *names* where exact terms matter.
        """
        if not isinstance(text, str):
            return ""

        language_type = self.detect_language(text)

        if language_type == "english":
            return self.english_processor.normalize(text)
        if language_type == "chinese":
            return self.chinese_processor.normalize(text)

        # Mixed
        chinese_text, english_text = self.separate_languages(text)
        parts: List[str] = []
        if chinese_text.strip():
            parts.append(self.chinese_processor.normalize(chinese_text))
        if english_text.strip():
            parts.append(self.english_processor.normalize(english_text))
        return " ".join(p for p in parts if p.strip())

    # ------------------------------------------------------------------ #
    # Field-level helpers                                                  #
    # ------------------------------------------------------------------ #

    def clean_product_field(self, field_name: str, field_value: str) -> str:
        """
        Apply the appropriate cleaning strategy for a given product field.

        Product *names* get lighter normalisation; descriptions get full cleaning.
        """
        if not isinstance(field_value, str):
            return ""
        if field_name.lower() in {
            "name",
            "title",
            "product_name",
            "name_en",
            "name_zh",
        }:
            return self.normalize_text(field_value)
        return self.clean_text(field_value)  # type: ignore[return-value]

    def clean_product_dict(self, product: Dict) -> Dict:
        """Clean all recognised text fields in a product dictionary."""
        cleaned = product.copy()
        text_field_keywords = {
            "name",
            "title",
            "description",
            "desc",
            "details",
            "specifications",
            "review",
            "comment",
            "feedback",
        }
        for key, value in product.items():
            if isinstance(value, str) and any(
                kw in key.lower() for kw in text_field_keywords
            ):
                cleaned[key] = self.clean_product_field(key, value)
        return cleaned

    def batch_clean(
        self, texts: List[str], preserve_original: bool = False
    ) -> List[Union[str, Dict]]:
        """Clean a list of texts."""
        return [self.clean_text(t, preserve_original) for t in texts]

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def get_statistics(self, text: str) -> Dict:
        """Return a summary dict describing cleaning impact on *text*."""
        cleaned = self.clean_text(text)
        language = self.detect_language(text)

        assert isinstance(cleaned, str)  # for type-checker

        return {
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "language_detected": language,
            "original_chinese_chars": len(re.findall(r"[\u4e00-\u9fff]", text)),
            "original_english_chars": len(re.findall(r"[a-zA-Z]", text)),
            "cleaned_chinese_chars": len(re.findall(r"[\u4e00-\u9fff]", cleaned)),
            "cleaned_english_chars": len(re.findall(r"[a-zA-Z]", cleaned)),
            "tokens": len(cleaned.split()),
            "removed_chars": len(text) - len(cleaned),
        }


# ── Convenience functions ─────────────────────────────────────────────── #


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Module-level shortcut for one-off text cleaning."""
    return TextCleaner(remove_stopwords=remove_stopwords).clean_text(text)  # type: ignore[return-value]


def normalize_text(text: str) -> str:
    """Module-level shortcut for one-off text normalisation."""
    return TextCleaner().normalize_text(text)
