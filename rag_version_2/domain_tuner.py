"""
Per-domain alpha tuning for optimizing the hybrid dense/sparse blend.
Different query types benefit from different alpha values.
"""

from typing import Dict, Optional, Tuple
import re
import logging

from .rag_config import RAG_V2_DOMAIN_ALPHA_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class DomainAlphaTuner:
    """
    Dynamically determine the best alpha (dense vs sparse weight) based on query domain.
    """
    
    # Domain keywords and their optimal alpha values
    DOMAIN_KEYWORDS = {
        'pest_control': {
            'keywords': [
                'కీటకాలు', 'పిచికారీ', 'కీటక', 'చిటాపు', 'పుగ',
                'ఎగ్', 'వర్మ', 'నాశనం', 'పీడక', 'గడ్డ',
                'రక్షణ', 'నియంత్రణ', 'నాశకుడు', 'పీడకలు',
                'pest', 'insect', 'spray', 'disease', 'control'
            ],
            'alpha': 0.4,  # More sparse for specific pest names
            'reason': 'Pest names are often exact terms, benefit from lexical matching'
        },
        'product_pricing': {
            'keywords': [
                'ధర', 'ఖరీదు', 'ఖర్చు', 'రూపాయలు', 'వేల్యు',
                'మూల్య', 'తక్క', 'ఎక్కువ', 'ఖరీద్', 'ఖరీదు',
                'price', 'cost', 'rupees', 'rate', 'charge'
            ],
            'alpha': 0.5,  # Balanced for product specs and prices
            'reason': 'Pricing queries need both semantic and exact matching'
        },
        'application_method': {
            'keywords': [
                'పిచికారీ', 'చేయడం', 'పద్ధతి', 'విధానం', 'ప్రక్రియ',
                'నిర్దేశాలు', 'సూచనలు', 'ఎలా', 'చేయాలి', 'వర్తించు',
                'spray', 'apply', 'method', 'process', 'how', 'use'
            ],
            'alpha': 0.5,  # Balanced so procedural queries keep more lexical signal
            'reason': 'Application methods still need semantics, but lexical cues matter in STT queries'
        },
        'quantity_dosage': {
            'keywords': [
                'మిల్లీ', 'లీటర్', 'గ్రాం', 'కిలో', 'పరిమాణం',
                'దోషణ', 'ఎంత', 'ఎన్ని', 'సంఖ్య', 'సంఖ్యల',
                'ml', 'litre', 'gram', 'kg', 'quantity', 'how much', 'how many'
            ],
            'alpha': 0.45,  # Slightly sparse for numeric precision
            'reason': 'Quantity queries benefit from exact number matching'
        },
        'timing_season': {
            'keywords': [
                'సీజన్', 'కాలం', 'నెల', 'సమయం', 'ఆర్తువు',
                'వసంత', 'ఉష్ణ', 'శీతల', 'వర్ష', 'నవంబర్',
                'season', 'month', 'time', 'when', 'weather', 'january'
            ],
            'alpha': 0.55,  # Balanced for temporal reasoning
            'reason': 'Seasonal queries need both semantic and temporal context'
        }
    }
    
    @staticmethod
    def identify_domain(query: str) -> Tuple[str, float, str]:
        """
        Identify the domain of a query and return optimal alpha.
        
        Args:
            query: User query (can be in Telugu or English)
            
        Returns:
            Tuple of (domain_name, alpha_value, reason)
        """
        query_lower = query.lower()
        domain_scores: Dict[str, float] = {}
        
        # Score each domain based on keyword matches
        for domain, config in DomainAlphaTuner.DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in config['keywords']:
                if keyword.lower() in query_lower:
                    score += 1
            domain_scores[domain] = score
        
        # Find domain with highest score
        if domain_scores and max(domain_scores.values()) > 0:
            best_domain = max(domain_scores, key=domain_scores.get)
            config = DomainAlphaTuner.DOMAIN_KEYWORDS[best_domain]
            return best_domain, config['alpha'], config['reason']
        
        # Default to balanced
        return 'general', 0.5, 'No specific domain identified, using balanced alpha'
    
    @staticmethod
    def get_alpha_with_confidence(query: str) -> Dict[str, any]:
        """
        Get alpha value with confidence score.
        
        Args:
            query: User query
            
        Returns:
            Dict with keys: domain, alpha, confidence, reason
        """
        domain_scores = {}
        
        query_lower = query.lower()
        total_keywords_found = 0
        
        # Score each domain
        for domain, config in DomainAlphaTuner.DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in config['keywords']:
                if keyword.lower() in query_lower:
                    score += 1
            domain_scores[domain] = score
            total_keywords_found += score
        
        if total_keywords_found == 0:
            return {
                'domain': 'general',
                'alpha': 0.5,
                'confidence': 0.0,
                'reason': 'No specific domain identified',
                'keyword_matches': 0
            }
        
        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_domain]
        
        # Confidence is based on keyword match ratio
        confidence = best_score / (total_keywords_found if total_keywords_found > 0 else 1)
        
        config = DomainAlphaTuner.DOMAIN_KEYWORDS[best_domain]
        
        return {
            'domain': best_domain,
            'alpha': config['alpha'],
            'confidence': min(confidence, 1.0),
            'reason': config['reason'],
            'keyword_matches': best_score
        }
    
    @staticmethod
    def adjust_alpha(base_alpha: float, query: str) -> float:
        """
        Adjust base alpha based on query characteristics.
        
        Args:
            base_alpha: Base alpha from per-domain tuning
            query: User query
            
        Returns:
            Adjusted alpha value
        """
        domain_info = DomainAlphaTuner.get_alpha_with_confidence(query)
        
        # If confidence is low, move towards center (0.5)
        confidence = domain_info['confidence']
        if confidence < 0.3:
            return 0.5 + (base_alpha - 0.5) * 0.5  # Reduce confidence effect
        elif confidence < 0.5:
            return 0.5 + (base_alpha - 0.5) * 0.75  # Moderate reduction
        
        return base_alpha
    
    @staticmethod
    def get_domain_alpha_override(query: str) -> Optional[float]:
        """
        Get domain-specific alpha override if query is clearly in a specific domain.
        
        Args:
            query: User query
            
        Returns:
            Alpha value if domain is identified with high confidence, None otherwise
        """
        info = DomainAlphaTuner.get_alpha_with_confidence(query)
        
        # Only return override if confidence is high
        if info['confidence'] >= RAG_V2_DOMAIN_ALPHA_CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Domain detected: {info['domain']} (confidence: {info['confidence']:.2f}), "
                f"using alpha: {info['alpha']}"
            )
            return info['alpha']
        
        return None
