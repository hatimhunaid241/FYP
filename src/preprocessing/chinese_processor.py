"""Chinese text processing and cleaning."""

import re
import unicodedata
from typing import List
from opencc import OpenCC
import jieba


class ChineseProcessor:
    """Process and clean Chinese text."""
    
    def __init__(self, convert_to_simplified: bool = True):
        """
        Initialize Chinese processor.
        
        Args:
            convert_to_simplified: Convert Traditional Chinese to Simplified
        """
        # Initialize OpenCC for Traditional to Simplified conversion
        if convert_to_simplified:
            self.converter = OpenCC('t2s')  # Traditional to Simplified
        else:
            self.converter = None
        
        # Common Chinese stop words
        self.stop_words = {
            '的', '一', '是', '在', '不', '了', '有', '和', '人', '这',
            '中', '大', '为', '上', '个', '国', '我', '以', '要', '他',
            '时', '来', '用', '们', '生', '到', '作', '地', '于', '出',
            '就', '分', '对', '成', '会', '可', '主', '发', '年', '动',
            '同', '工', '也', '能', '下', '过', '子', '说', '产', '样',
            '配', '件', '等', '所', '则', '叶', '结', '更', '多', '然'
        }
    
    def clean(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Clean Chinese text.
        
        Args:
            text: Input text to clean
            remove_stopwords: Whether to remove common Chinese stop words
            
        Returns:
            Cleaned Chinese text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert Traditional Chinese to Simplified if converter is available
        if self.converter:
            text = self.converter.convert(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML entities and tags
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        # Remove special characters but keep Chinese characters and spaces
        # Keep Chinese characters (CJK) and alphanumeric
        text = re.sub(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and optionally remove stopwords
        tokens = jieba.cut(text, cut_all=False)
        tokens = list(tokens)
        
        if remove_stopwords:
            tokens = [w.strip() for w in tokens if w.strip() and w.strip() not in self.stop_words]
        else:
            tokens = [w.strip() for w in tokens if w.strip()]
        
        text = ' '.join(tokens)
        
        return text
    
    def normalize(self, text: str) -> str:
        """
        Normalize Chinese text (lighter cleaning, keeps more detail).
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert Traditional Chinese to Simplified if converter is available
        if self.converter:
            text = self.converter.convert(text)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Keep Chinese characters and spaces, remove extra punctuation
        text = re.sub(r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffa-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        cleaned = self.clean(text)
        # jieba.cut already called in clean(), so we split by spaces
        return cleaned.split()
    
    def extract_features(self, text: str) -> dict:
        """
        Extract features from Chinese text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted features
        """
        cleaned = self.clean(text)
        tokens = cleaned.split()
        
        # Count Chinese characters
        chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        
        return {
            'original_length': len(text),
            'cleaned_length': len(cleaned),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'chinese_char_count': chinese_char_count,
            'cleaned_text': cleaned,
            'tokens': tokens
        }
    
    def is_chinese(self, text: str) -> bool:
        """
        Check if text contains Chinese characters.
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Chinese characters
        """
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def extract_chinese(self, text: str) -> str:
        """
        Extract only Chinese characters from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with only Chinese characters and spaces
        """
        # Extract only Chinese characters
        extracted = re.findall(r'[\u4e00-\u9fff]+', text)
        return ' '.join(extracted)
