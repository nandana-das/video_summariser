"""
Advanced transcript summarization module for the Video Summarizer project.
"""
import nltk
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter
import spacy
from utils.logger import setup_logger
from config import SUMMARIZATION_SETTINGS, OUTPUT_DIR

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

logger = setup_logger(__name__)

class AdvancedSummarizer:
    """Advanced text summarization using multiple NLP techniques."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the advanced summarizer.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.output_dir = OUTPUT_DIR / "summaries"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            self.nlp = None
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for summarization.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of preprocessed sentences
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences
        min_length = SUMMARIZATION_SETTINGS["min_sentence_length"]
        sentences = [s for s in sentences if len(s.split()) >= min_length]
        
        return sentences
    
    def extract_keywords(self, text: str, top_k: int = 20) -> List[str]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        try:
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_indices = np.argsort(scores)[-top_k:][::-1]
            keywords = [feature_names[i] for i in keyword_indices if scores[i] > 0]
            
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity, label) tuples
        """
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except Exception as e:
            logger.error(f"Error extracting named entities: {str(e)}")
            return []
    
    def extract_action_items(self, text: str) -> List[str]:
        """
        Extract action items and tasks from text.
        
        Args:
            text: Input text
            
        Returns:
            List of action items
        """
        action_patterns = [
            r'(?:need to|should|must|have to|will|going to)\s+[^.]*',
            r'(?:action item|task|todo|follow up|next step)[^.]*',
            r'(?:assign|delegate|responsible for)[^.]*',
            r'(?:deadline|due date|by when)[^.]*'
        ]
        
        action_items = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            action_items.extend(matches)
        
        return action_items
    
    def calculate_sentence_scores(self, sentences: List[str], keywords: List[str]) -> Dict[int, float]:
        """
        Calculate scores for sentences based on various features.
        
        Args:
            sentences: List of sentences
            keywords: List of keywords
            
        Returns:
            Dictionary mapping sentence index to score
        """
        scores = {}
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Keyword frequency score
            words = word_tokenize(sentence.lower())
            keyword_count = sum(1 for word in words if word in keywords)
            score += keyword_count * 0.1
            
            # Position score (sentences at beginning and end are more important)
            if i < len(sentences) * 0.1 or i > len(sentences) * 0.9:
                score += 0.1
            
            # Length score (prefer medium-length sentences)
            word_count = len(words)
            if 10 <= word_count <= 30:
                score += 0.05
            
            # Named entity score
            if self.nlp:
                doc = self.nlp(sentence)
                entity_count = len(doc.ents)
                score += entity_count * 0.05
            
            scores[i] = score
        
        return scores
    
    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build similarity matrix between sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix
        """
        try:
            # Vectorize sentences
            sentence_vectors = self.tfidf_vectorizer.fit_transform(sentences)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(sentence_vectors)
            
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error building similarity matrix: {str(e)}")
            return np.zeros((len(sentences), len(sentences)))
    
    def generate_summary(self, text: str, max_sentences: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the text.
        
        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Dictionary containing summary and metadata
        """
        if max_sentences is None:
            max_sentences = SUMMARIZATION_SETTINGS["max_sentences"]
        
        try:
            logger.info("Generating summary...")
            
            # Preprocess text
            sentences = self.preprocess_text(text)
            
            if len(sentences) < 2:
                return {
                    "summary": text,
                    "keywords": [],
                    "action_items": [],
                    "named_entities": [],
                    "metadata": {"sentence_count": len(sentences)}
                }
            
            # Extract keywords
            keywords = self.extract_keywords(text)
            
            # Extract named entities
            named_entities = self.extract_named_entities(text)
            
            # Extract action items
            action_items = self.extract_action_items(text)
            
            # Calculate sentence scores
            sentence_scores = self.calculate_sentence_scores(sentences, keywords)
            
            # Build similarity matrix
            similarity_matrix = self.build_similarity_matrix(sentences)
            
            # Use PageRank to rank sentences
            graph = nx.from_numpy_array(similarity_matrix)
            pagerank_scores = nx.pagerank(graph)
            
            # Combine scores
            final_scores = {}
            for i in range(len(sentences)):
                final_scores[i] = sentence_scores[i] + pagerank_scores[i]
            
            # Select top sentences
            top_sentences = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            selected_indices = [i for i, _ in top_sentences[:max_sentences]]
            selected_indices.sort()  # Maintain original order
            
            # Create summary
            summary_sentences = [sentences[i] for i in selected_indices]
            summary = " ".join(summary_sentences)
            
            # Create metadata
            metadata = {
                "original_sentence_count": len(sentences),
                "summary_sentence_count": len(summary_sentences),
                "compression_ratio": len(summary_sentences) / len(sentences),
                "keyword_count": len(keywords),
                "action_item_count": len(action_items),
                "named_entity_count": len(named_entities)
            }
            
            result = {
                "summary": summary,
                "keywords": keywords,
                "action_items": action_items,
                "named_entities": named_entities,
                "metadata": metadata
            }
            
            logger.info("Summary generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
    
    def save_summary(self, summary_data: Dict[str, Any], 
                    output_name: Optional[str] = None) -> Path:
        """
        Save summary to file.
        
        Args:
            summary_data: Summary data dictionary
            output_name: Optional custom name for output file
            
        Returns:
            Path to the saved summary file
        """
        if output_name is None:
            output_name = "summary.txt"
        elif not output_name.endswith(".txt"):
            output_name += ".txt"
        
        output_path = self.output_dir / output_name
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=== MEETING SUMMARY ===\n\n")
                f.write(summary_data["summary"])
                f.write("\n\n=== KEYWORDS ===\n")
                f.write(", ".join(summary_data["keywords"]))
                f.write("\n\n=== ACTION ITEMS ===\n")
                for item in summary_data["action_items"]:
                    f.write(f"• {item}\n")
                f.write("\n=== NAMED ENTITIES ===\n")
                for entity, label in summary_data["named_entities"]:
                    f.write(f"• {entity} ({label})\n")
                f.write("\n=== METADATA ===\n")
                for key, value in summary_data["metadata"].items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Summary saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
            raise

def main():
    """Example usage of the AdvancedSummarizer class."""
    summarizer = AdvancedSummarizer()
    
    # Example: Summarize a transcript
    try:
        transcript_path = "output/transcripts/transcript4_transcript.txt"  # Replace with your transcript path
        with open(transcript_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Generate summary
        summary_data = summarizer.generate_summary(text)
        
        # Save summary
        summary_path = summarizer.save_summary(summary_data)
        print(f"Summary saved to: {summary_path}")
        
        # Print summary
        print("\n=== SUMMARY ===")
        print(summary_data["summary"])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
