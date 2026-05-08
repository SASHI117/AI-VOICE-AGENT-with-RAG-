"""
Telugu text preprocessing module.
Handles diacritics normalization and common Telugu script variants.
"""

import re
from typing import Dict, Set


class TeluguPreprocessor:
    """Normalize Telugu text variants and diacritics for better retrieval."""
    
    # Common Telugu diacritics and their normalized forms
    DIACRITICS_MAP = {
        # Chandrabindu variants
        'ఁ': 'ం',  # Candrabindu to Anusvara
        
        # Virama handling (helps with consonant cluster normalization)
        '్': '్',  # Keep as-is, used for consonant clusters
        
        # Vowel sign variants (same sound, different representation)
        # These are typically typing variations
    }
    
    # Common Telugu word variants (phonetic/spelling variations)
    VARIANT_REPLACEMENTS = {
        # Pest/Disease names
        'ఫిరమోన్': ['ఫెరమోన్', 'ఫెర్మోన్'],
        'పిచికారీ': ['పిచికారి', 'పిచికారీ'],
        'కీటకాలు': ['కీటకాల', 'కీటక'],
        
        # Product names
        'స్టిక్కీ': ['స్టిక్కి', 'స్టికీ'],
        'ట్రాప్': ['ట్రాప్‌', 'ట్రాప్'],
        'నీలం': ['నీల', 'నీలా'],
        
        # Action words
        'పిచికారీ': ['పిచికారు', 'పిచికార్చు', 'పిచికారీ చేయడం'],
        'చిటాపు': ['చిటాపు', 'చిటాప్'],
    }
    
    # Stop words that don't help with retrieval (Telugu)
    TELUGU_STOP_WORDS = {
        'ఆ', 'అ', 'ఈ', 'ఉ', 'ఎ', 'ఏ', 'ఒ', 'ఓ', 'ఔ',  # Vowels
        'కు', 'కి', 'కూ', 'కే', 'కో',  # Postpositions
        'కి', 'కికూ', 'కీ', 'కీకూ', 'కీకీ',
        'ఇ', 'ఈ', 'ఎ', 'ఏ',  # More vowels
        'చేసిన', 'చేయడం', 'ఉన్నది', 'ఉందా', 'ఉంది',  # Auxiliary verbs
        'ఇందులో', 'దీనిలో', 'దీనిపై', 'దీని',  # Demonstratives
        'మరియు', 'మరి', 'కాని', 'కానీ',  # Conjunctions
        'ఎందుకు', 'ఎందుకంటే', 'ఎందుకనీ',  # Question words
    }
    
    @staticmethod
    def normalize_diacritics(text: str) -> str:
        """
        Normalize Telugu diacritics and script variants.
        
        Args:
            text: Telugu text to normalize
            
        Returns:
            Normalized text with consistent diacritics
        """
        # Apply diacritics replacements
        for old, new in TeluguPreprocessor.DIACRITICS_MAP.items():
            text = text.replace(old, new)
        
        # Remove zero-width characters that sometimes appear
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        text = text.replace('\ufeff', '')  # Zero-width no-break space
        
        return text.strip()
    
    @staticmethod
    def expand_variants(text: str) -> Set[str]:
        """
        Generate variant forms of the text for better matching.
        
        Args:
            text: Original text
            
        Returns:
            Set of original text and common variants
        """
        variants = {text}
        
        # Check if text contains any known variant words
        for canonical, variant_list in TeluguPreprocessor.VARIANT_REPLACEMENTS.items():
            if canonical in text:
                for variant in variant_list:
                    variants.add(text.replace(canonical, variant))
        
        return variants
    
    @staticmethod
    def remove_stop_words(text: str) -> str:
        """
        Remove Telugu stop words that don't contribute to meaning.
        
        Args:
            text: Telugu text
            
        Returns:
            Text with stop words removed
        """
        words = text.split()
        filtered = [w for w in words if w not in TeluguPreprocessor.TELUGU_STOP_WORDS]
        return ' '.join(filtered)
    
    @staticmethod
    def preprocess(text: str, remove_stops: bool = False) -> str:
        """
        Complete preprocessing pipeline for Telugu text.
        
        Args:
            text: Telugu text to preprocess
            remove_stops: Whether to remove stop words (default: False for retrieval)
            
        Returns:
            Preprocessed text
        """
        # Normalize diacritics first
        text = TeluguPreprocessor.normalize_diacritics(text)
        
        # Optionally remove stop words (usually not for retrieval)
        if remove_stops:
            text = TeluguPreprocessor.remove_stop_words(text)
        
        return text
    
    @staticmethod
    def preprocess_query(query: str) -> tuple[str, Set[str]]:
        """
        Preprocess a query and generate variants for multi-attempt retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Tuple of (preprocessed_query, variant_set)
        """
        preprocessed = TeluguPreprocessor.preprocess(query)
        variants = TeluguPreprocessor.expand_variants(preprocessed)
        return preprocessed, variants
