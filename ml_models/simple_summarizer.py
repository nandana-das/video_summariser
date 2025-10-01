"""
Simple fallback summarizer that doesn't require transformers.
Uses basic NLP techniques for text summarization.
"""

import re
import nltk
from typing import List, Dict, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class SimpleSummarizer:
    """Simple text summarizer using basic NLP techniques."""
    
    def __init__(self):
        """Initialize the simple summarizer."""
        self.stop_words = set()
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK fails
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with'
            }
    
    def summarize_text(self, text: str, max_sentences: int = 5) -> str:
        """
        Summarize text using extractive summarization.
        
        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summarized text
        """
        try:
            # Clean and split text into sentences
            sentences = self._split_sentences(text)
            
            if len(sentences) <= max_sentences:
                return text
            
            # Calculate sentence scores
            sentence_scores = self._calculate_sentence_scores(sentences)
            
            # Select top sentences
            top_sentences = sorted(
                sentence_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:max_sentences]
            
            # Sort by original order
            selected_sentences = sorted(
                [sent for sent, score in top_sentences],
                key=lambda x: sentences.index(x)
            )
            
            return '. '.join(selected_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error in simple summarization: {e}")
            # Fallback: return first few sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:max_sentences]) + '.'
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _calculate_sentence_scores(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate importance scores for sentences."""
        scores = {}
        
        # Get word frequencies
        all_words = []
        for sentence in sentences:
            words = self._extract_words(sentence)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Normalize frequencies
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score each sentence
        for sentence in sentences:
            words = self._extract_words(sentence)
            if not words:
                scores[sentence] = 0
                continue
            
            # Calculate score based on word frequency and sentence length
            word_scores = [word_freq.get(word, 0) for word in words]
            avg_word_score = sum(word_scores) / len(words)
            
            # Bonus for longer sentences (but not too long)
            length_bonus = min(len(words) / 20, 1.0)  # Cap at 1.0
            
            scores[sentence] = avg_word_score + (length_bonus * 0.1)
        
        return scores
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract and clean words from text."""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Remove stop words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return words
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        try:
            words = self._extract_words(text)
            word_freq = Counter(words)
            return [word for word, freq in word_freq.most_common(num_keywords)]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def extract_action_items(self, text: str) -> List[str]:
        """Extract potential action items from text."""
        try:
            sentences = self._split_sentences(text)
            action_items = []
            
            # Look for action words and patterns
            action_patterns = [
                r'\b(?:need to|should|must|have to|will|going to)\b',
                r'\b(?:action|task|step|next|follow up|implement)\b',
                r'\b(?:please|make sure|ensure|remember)\b'
            ]
            
            for sentence in sentences:
                for pattern in action_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        action_items.append(sentence.strip())
                        break
            
            return action_items[:10]  # Limit to 10 items
            
        except Exception as e:
            logger.error(f"Error extracting action items: {e}")
            return []
