import spacy
import re
from typing import Dict, List, Any, Optional
import os
from collections import Counter
import math

class NLPService:
    def __init__(self):
        self.nlp = None
        self.load_model()
    
    def load_model(self):
        """Load spaCy model, download if necessary"""
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model is not installed, we'll use basic text processing
            self.nlp = None
            print("Warning: spaCy English model not found. Using basic text processing.")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis"""
        if not text or not isinstance(text, str):
            return self._empty_analysis()
        
        # Basic text statistics
        analysis = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'paragraph_count': len([p for p in text.split('\n') if p.strip()]),
            'avg_word_length': self._average_word_length(text),
            'readability_score': self._calculate_readability(text),
            'formality_score': self._calculate_formality(text),
            'caps_ratio': self._calculate_caps_ratio(text),
            'punctuation_density': self._calculate_punctuation_density(text)
        }
        
        # spaCy analysis if available
        if self.nlp:
            doc = self.nlp(text)
            analysis.update({
                'entities': [{'text': ent.text, 'label': ent.label_} for ent in doc.ents],
                'pos_tags': dict(Counter([token.pos_ for token in doc if not token.is_space])),
                'sentiment_indicators': self._analyze_sentiment_indicators(doc),
                'complexity_score': self._calculate_complexity(doc)
            })
        else:
            # Basic analysis without spaCy
            analysis.update({
                'entities': [],
                'pos_tags': {},
                'sentiment_indicators': self._basic_sentiment_analysis(text),
                'complexity_score': self._basic_complexity_score(text)
            })
        
        # Detect potential anomalies in text
        analysis['anomaly_flags'] = self._detect_text_anomalies(text, analysis)
        
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'avg_word_length': 0,
            'readability_score': 0,
            'formality_score': 0,
            'caps_ratio': 0,
            'punctuation_density': 0,
            'entities': [],
            'pos_tags': {},
            'sentiment_indicators': {},
            'complexity_score': 0,
            'anomaly_flags': []
        }
    
    def _average_word_length(self, text: str) -> float:
        """Calculate average word length"""
        words = text.split()
        if not words:
            return 0
        return sum(len(word.strip('.,!?;:')) for word in words) / len(words)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate a simple readability score (Flesch-like)"""
        words = text.split()
        sentences = re.findall(r'[.!?]+', text)
        
        if not words or not sentences:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower().strip('.,!?;:')
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_formality(self, text: str) -> float:
        """Calculate text formality score"""
        formal_indicators = [
            'therefore', 'however', 'furthermore', 'consequently', 'nevertheless',
            'accordingly', 'subsequently', 'regarding', 'concerning', 'pursuant'
        ]
        
        informal_indicators = [
            'gonna', 'wanna', 'yeah', 'ok', 'okay', 'hey', 'hi', 'wow',
            'awesome', 'cool', 'super', 'really', 'totally'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.5
        
        formal_ratio = formal_count / total_words
        informal_ratio = informal_count / total_words
        
        # Scale to 0-1 where 1 is most formal
        return min(1.0, max(0.0, 0.5 + (formal_ratio - informal_ratio) * 10))
    
    def _calculate_caps_ratio(self, text: str) -> float:
        """Calculate ratio of capital letters"""
        if not text:
            return 0
        caps_count = sum(1 for c in text if c.isupper())
        return caps_count / len(text)
    
    def _calculate_punctuation_density(self, text: str) -> float:
        """Calculate punctuation density"""
        if not text:
            return 0
        punct_count = sum(1 for c in text if c in '.,!?;:()[]{}"-')
        return punct_count / len(text)
    
    def _analyze_sentiment_indicators(self, doc) -> Dict[str, int]:
        """Analyze sentiment indicators using spaCy"""
        positive_words = ['good', 'excellent', 'outstanding', 'superior', 'best', 'high-quality', 'efficient', 'reliable']
        negative_words = ['bad', 'poor', 'inadequate', 'insufficient', 'low-quality', 'unreliable', 'problematic']
        
        text_lower = doc.text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return {
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_balance': positive_count - negative_count
        }
    
    def _basic_sentiment_analysis(self, text: str) -> Dict[str, int]:
        """Basic sentiment analysis without spaCy"""
        positive_words = ['good', 'excellent', 'outstanding', 'superior', 'best', 'high-quality', 'efficient', 'reliable']
        negative_words = ['bad', 'poor', 'inadequate', 'insufficient', 'low-quality', 'unreliable', 'problematic']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return {
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_balance': positive_count - negative_count
        }
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate text complexity using spaCy"""
        if not doc:
            return 0
        
        # Count complex structures
        complex_pos = ['NOUN', 'ADJ', 'ADV', 'VERB']
        complex_count = sum(1 for token in doc if token.pos_ in complex_pos)
        
        # Count dependencies
        dep_complexity = len([token for token in doc if token.dep_ in ['amod', 'nmod', 'acl']])
        
        total_tokens = len([token for token in doc if not token.is_space])
        if total_tokens == 0:
            return 0
        
        return (complex_count + dep_complexity) / total_tokens
    
    def _basic_complexity_score(self, text: str) -> float:
        """Basic complexity score without spaCy"""
        words = text.split()
        if not words:
            return 0
        
        # Simple heuristics for complexity
        long_words = sum(1 for word in words if len(word) > 6)
        complex_punctuation = text.count(';') + text.count(':') + text.count('(') + text.count(')')
        
        return (long_words + complex_punctuation) / len(words)
    
    def _detect_text_anomalies(self, text: str, analysis: Dict[str, Any]) -> List[str]:
        """Detect potential anomalies in text"""
        anomalies = []
        
        # Check for extremely short or long text
        if analysis['word_count'] < 10:
            anomalies.append('extremely_short_text')
        elif analysis['word_count'] > 2000:
            anomalies.append('extremely_long_text')
        
        # Check for unusual formatting
        if analysis['caps_ratio'] > 0.3:
            anomalies.append('excessive_capitals')
        
        if analysis['punctuation_density'] > 0.2:
            anomalies.append('excessive_punctuation')
        
        # Check for copy-paste indicators
        if self._detect_repetition(text):
            anomalies.append('repetitive_content')
        
        # Check for template-like structure
        if self._detect_template_structure(text):
            anomalies.append('template_like_structure')
        
        # Check readability
        if analysis['readability_score'] < 20:
            anomalies.append('very_difficult_to_read')
        elif analysis['readability_score'] > 90:
            anomalies.append('unusually_simple_text')
        
        return anomalies
    
    def _detect_repetition(self, text: str) -> bool:
        """Detect repetitive content"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            return False
        
        # Check for repeated sentences
        sentence_counts = Counter(sentences)
        repeated_sentences = sum(1 for count in sentence_counts.values() if count > 1)
        
        return repeated_sentences > len(sentences) * 0.3
    
    def _detect_template_structure(self, text: str) -> bool:
        """Detect template-like structure"""
        template_indicators = [
            '[', ']', '{{', '}}', '__', 'PLACEHOLDER', 'TODO', 'FILL_IN',
            'INSERT_', 'COMPANY_NAME', 'DATE_HERE', 'AMOUNT_HERE'
        ]
        
        text_upper = text.upper()
        template_count = sum(1 for indicator in template_indicators if indicator in text_upper)
        
        return template_count > 0
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two texts for similarity"""
        if not text1 or not text2:
            return {'similarity_score': 0, 'common_phrases': []}
        
        # Basic similarity metrics
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return {'similarity_score': 0, 'common_phrases': []}
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard_score = len(intersection) / len(union) if union else 0
        
        # Find common phrases (simple n-gram approach)
        common_phrases = self._find_common_phrases(text1, text2)
        
        return {
            'similarity_score': round(jaccard_score, 3),
            'common_words': list(intersection),
            'common_phrases': common_phrases,
            'word_overlap_ratio': round(len(intersection) / min(len(words1), len(words2)), 3)
        }
    
    def _find_common_phrases(self, text1: str, text2: str, min_length: int = 3) -> List[str]:
        """Find common phrases between two texts"""
        def get_ngrams(text: str, n: int) -> List[str]:
            words = text.lower().split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        common_phrases = []
        
        # Check for common 3-grams and 4-grams
        for n in [3, 4]:
            ngrams1 = set(get_ngrams(text1, n))
            ngrams2 = set(get_ngrams(text2, n))
            common = ngrams1.intersection(ngrams2)
            common_phrases.extend(list(common))
        
        # Remove duplicates and sort by length
        common_phrases = list(set(common_phrases))
        common_phrases.sort(key=len, reverse=True)
        
        return common_phrases[:10]  # Return top 10
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from tender/bid text"""
        info = {
            'monetary_amounts': self._extract_monetary_amounts(text),
            'dates': self._extract_dates(text),
            'companies': self._extract_companies(text),
            'technical_terms': self._extract_technical_terms(text),
            'requirements': self._extract_requirements(text)
        }
        
        return info
    
    def _extract_monetary_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts from text"""
        patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # $1,000.00
            r'USD\s*[\d,]+(?:\.\d{2})?',  # USD 1000
            r'[\d,]+(?:\.\d{2})?\s*(?:dollars|USD|\$)',  # 1000 dollars
            r'€[\d,]+(?:\.\d{2})?',  # €1,000.00
            r'£[\d,]+(?:\.\d{2})?'   # £1,000.00
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        return list(set(amounts))
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract company names (basic pattern matching)"""
        patterns = [
            r'\b[A-Z][a-zA-Z\s&]+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co)\b',
            r'\b[A-Z][a-zA-Z\s&]+\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co)\.?\b'
        ]
        
        companies = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)
        
        return list(set(companies))
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and jargon"""
        # Common technical terms in tender/bid contexts
        technical_patterns = [
            r'\b(?:API|SDK|SLA|KPI|ROI|ERP|CRM|AWS|Azure|DevOps|CI/CD|SSL|TLS|HTTPS|REST|JSON|XML|SQL|NoSQL)\b',
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]
        
        terms = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))
    
    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirement statements"""
        requirement_patterns = [
            r'(?:must|shall|should|required?|mandatory|essential|necessary)\s+[^.!?]*[.!?]',
            r'(?:The contractor|The vendor|The supplier)\s+[^.!?]*[.!?]'
        ]
        
        requirements = []
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            requirements.extend([match.strip() for match in matches])
        
        return requirements[:10]  # Return top 10