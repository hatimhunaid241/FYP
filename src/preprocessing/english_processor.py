"""English text processing and cleaning."""

import re
import string
from typing import List


class EnglishProcessor:
    """Process and clean English text."""
    
    def __init__(self):
        """Initialize English processor."""
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
            'the', 'to', 'was', 'will', 'with', 'this', 'but', 'have', 'not',
            'what', 'when', 'where', 'which', 'who', 'why', 'how', 'all',
            'each', 'every', 'both', 'such', 'so', 'than', 'then', 'they'
        }
    
    def clean(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Clean English text.
        
        Args:
            text: Input text to clean
            remove_stopwords: Whether to remove common English stop words
            
        Returns:
            Cleaned English text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except hyphens in compound words
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Clean up hyphens
        text = re.sub(r'\s+\-\s+|\s*\-\s*', '-', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = text.split()
            tokens = [w for w in tokens if w not in self.stop_words]
            text = ' '.join(tokens)
        
        return text
    
    def normalize(self, text: str) -> str:
        """
        Normalize English text (lighter cleaning, keeps more detail).
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove only special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        
        # Clean up spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize English text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        cleaned = self.clean(text)
        return cleaned.split()
    
    def extract_features(self, text: str) -> dict:
        """
        Extract features from English text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted features
        """
        cleaned = self.clean(text)
        tokens = cleaned.split()
        
        return {
            'original_length': len(text),
            'cleaned_length': len(cleaned),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'cleaned_text': cleaned,
            'tokens': tokens
        }
