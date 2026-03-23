"""Main text cleaner for bilingual (English and Chinese) product text."""

import re
import logging
from typing import Tuple, Dict, List, Union
from .english_processor import EnglishProcessor
from .chinese_processor import ChineseProcessor

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and preprocess bilingual product text (English and Chinese)."""
    
    def __init__(self, remove_stopwords: bool = False, convert_simplified: bool = True):
        """
        Initialize TextCleaner.
        
        Args:
            remove_stopwords: Whether to remove common stop words
            convert_simplified: Convert Traditional Chinese to Simplified
        """
        self.english_processor = EnglishProcessor()
        self.chinese_processor = ChineseProcessor(convert_to_simplified=convert_simplified)
        self.remove_stopwords = remove_stopwords
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily English, Chinese, or mixed.
        
        Args:
            text: Input text
            
        Returns:
            'english', 'chinese', or 'mixed'
        """
        if not isinstance(text, str):
            return 'unknown'
        
        # Count Chinese and English characters
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_count = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_count == 0 and english_count == 0:
            return 'unknown'
        elif chinese_count > english_count * 2:
            return 'chinese'
        elif english_count > chinese_count * 2:
            return 'english'
        else:
            return 'mixed'
    
    def separate_languages(self, text: str) -> Tuple[str, str]:
        """
        Separate Chinese and English text.
        
        Args:
            text: Mixed text input
            
        Returns:
            Tuple of (chinese_text, english_text)
        """
        # Extract Chinese text (continuous Chinese characters)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        chinese_text = ' '.join(chinese_chars)
        
        # Extract English text (continuous English words)
        english_chars = re.findall(r'[a-zA-Z]+', text)
        english_text = ' '.join(english_chars)
        
        return chinese_text, english_text
    
    def clean_text(self, text: str, preserve_original: bool = False) -> str:
        """
        Clean text by removing non-English and non-Chinese characters.
        Applies language-specific cleaning to each part.
        
        Args:
            text: Input text to clean
            preserve_original: If True, return a dict with original and cleaned text
            
        Returns:
            Cleaned text (mixed English and Chinese), or dict if preserve_original=True
        """
        if not isinstance(text, str):
            return "" if not preserve_original else {"original": "", "cleaned": ""}
        
        original_text = text
        
        # Detect language
        language_type = self.detect_language(text)
        
        if language_type == 'english':
            cleaned = self.english_processor.clean(text, self.remove_stopwords)
        elif language_type == 'chinese':
            cleaned = self.chinese_processor.clean(text, self.remove_stopwords)
        else:
            # Mixed or unknown
            # Separate and clean both parts
            chinese_text, english_text = self.separate_languages(text)
            
            cleaned_chinese = ""
            cleaned_english = ""
            
            if chinese_text.strip():
                cleaned_chinese = self.chinese_processor.clean(chinese_text, self.remove_stopwords)
            
            if english_text.strip():
                cleaned_english = self.english_processor.clean(english_text, self.remove_stopwords)
            
            # Combine cleaned parts
            parts = [p for p in [cleaned_chinese, cleaned_english] if p.strip()]
            cleaned = ' '.join(parts)
        
        if preserve_original:
            return {
                "original": original_text,
                "cleaned": cleaned,
                "language": language_type
            }
        
        return cleaned
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text with lighter cleaning (preserves more original formatting).
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        language_type = self.detect_language(text)
        
        if language_type == 'english':
            return self.english_processor.normalize(text)
        elif language_type == 'chinese':
            return self.chinese_processor.normalize(text)
        else:
            # Mixed
            chinese_text, english_text = self.separate_languages(text)
            
            normalized_parts = []
            if chinese_text.strip():
                normalized_parts.append(self.chinese_processor.normalize(chinese_text))
            if english_text.strip():
                normalized_parts.append(self.english_processor.normalize(english_text))
            
            return ' '.join([p for p in normalized_parts if p.strip()])
    
    def clean_product_field(self, field_name: str, field_value: str) -> str:
        """
        Clean a specific product field.
        
        Args:
            field_name: Name of the field (e.g., 'name', 'description')
            field_value: Value of the field
            
        Returns:
            Cleaned field value
        """
        if not isinstance(field_value, str):
            return ""
        
        # For product names, use normalize (lighter cleaning)
        if field_name.lower() in ['name', 'title', 'product_name']:
            return self.normalize_text(field_value)
        
        # For descriptions and reviews, use full cleaning
        return self.clean_text(field_value)
    
    def clean_product_dict(self, product: Dict) -> Dict:
        """
        Clean all text fields in a product dictionary.
        
        Args:
            product: Dictionary containing product data
            
        Returns:
            Product dictionary with cleaned text fields
        """
        cleaned_product = product.copy()
        
        # List of common text field names
        text_fields = [
            'name', 'title', 'product_name', 'product_title',
            'description', 'desc', 'details', 'specifications',
            'review', 'comment', 'feedback'
        ]
        
        for key, value in product.items():
            if isinstance(value, str) and any(field.lower() in key.lower() for field in text_fields):
                cleaned_product[key] = self.clean_product_field(key, value)
        
        return cleaned_product
    
    def batch_clean(self, texts: List[str], preserve_original: bool = False) -> List[Union[str, Dict]]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of texts to clean
            preserve_original: If True, return dicts with original and cleaned text
            
        Returns:
            List of cleaned texts or dicts
        """
        return [self.clean_text(text, preserve_original) for text in texts]
    
    def get_statistics(self, text: str) -> Dict:
        """
        Get cleaning statistics for a text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with cleaning statistics
        """
        cleaned = self.clean_text(text)
        language = self.detect_language(text)
        
        # Count characters
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_count = len(re.findall(r'[a-zA-Z]', text))
        
        cleaned_chinese_count = len(re.findall(r'[\u4e00-\u9fff]', cleaned))
        cleaned_english_count = len(re.findall(r'[a-zA-Z]', cleaned))
        
        return {
            'original_length': len(text),
            'cleaned_length': len(cleaned),
            'language_detected': language,
            'original_chinese_chars': chinese_count,
            'original_english_chars': english_count,
            'cleaned_chinese_chars': cleaned_chinese_count,
            'cleaned_english_chars': cleaned_english_count,
            'tokens': len(cleaned.split()),
            'removed_chars': len(text) - len(cleaned)
        }


# Convenience functions for simple use cases
def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Quick function to clean text."""
    cleaner = TextCleaner(remove_stopwords=remove_stopwords)
    return cleaner.clean_text(text)


def normalize_text(text: str) -> str:
    """Quick function to normalize text."""
    cleaner = TextCleaner()
    return cleaner.normalize_text(text)
