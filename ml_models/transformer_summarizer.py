"""
Advanced transformer-based text summarization module.
Supports BART, T5, Pegasus, and other state-of-the-art models.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    pipeline, AutoModel, AutoConfig
)
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import mlflow
import mlflow.transformers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

class TransformerSummarizer:
    """Advanced transformer-based text summarization."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", 
                 device: str = "auto", max_length: int = 1024):
        """
        Initialize the transformer summarizer.
        
        Args:
            model_name: Name of the transformer model to use
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            max_length: Maximum input length for the model
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.sentence_model = None
        
        # Initialize models
        self._load_models()
        
        # Initialize evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        logger.info(f"TransformerSummarizer initialized with {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_models(self):
        """Load the transformer model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Load sentence transformer for similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def summarize_text(self, text: str, max_length: int = 150, 
                      min_length: int = 30, num_beams: int = 4,
                      temperature: float = 1.0, do_sample: bool = False) -> str:
        """
        Generate summary using transformer model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            do_sample: Whether to use sampling
            
        Returns:
            Generated summary text
        """
        try:
            # Truncate input if too long
            inputs = self.tokenizer(
                text, 
                max_length=self.max_length, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=do_sample,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
    
    def extractive_summarize(self, text: str, num_sentences: int = 5) -> str:
        """
        Generate extractive summary using sentence similarity.
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            Extractive summary
        """
        try:
            # Split into sentences
            sentences = text.split('. ')
            if len(sentences) <= num_sentences:
                return text
            
            # Get sentence embeddings
            sentence_embeddings = self.sentence_model.encode(sentences)
            
            # Calculate sentence importance scores
            scores = []
            for i, embedding in enumerate(sentence_embeddings):
                # Calculate similarity to all other sentences
                similarities = np.dot(sentence_embeddings, embedding) / (
                    np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(embedding)
                )
                # Score is average similarity (excluding self)
                score = np.mean(similarities[np.arange(len(similarities)) != i])
                scores.append(score)
            
            # Select top sentences
            top_indices = np.argsort(scores)[-num_sentences:]
            top_indices = sorted(top_indices)  # Maintain order
            
            # Combine selected sentences
            summary_sentences = [sentences[i] for i in top_indices]
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            raise
    
    def evaluate_summary(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Evaluate summary quality using ROUGE and BERTScore.
        
        Args:
            reference: Reference summary
            candidate: Generated summary
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference, candidate)
            rouge_metrics = {
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure
            }
            
            # BERTScore
            P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
            bert_metrics = {
                'bert_precision': P.item(),
                'bert_recall': R.item(),
                'bert_f1': F1.item()
            }
            
            # Combine metrics
            metrics = {**rouge_metrics, **bert_metrics}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating summary: {str(e)}")
            return {}
    
    def generate_abstractive_summary(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Generate abstractive summary with metadata.
        
        Args:
            text: Input text
            **kwargs: Additional parameters for summarization
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            # Generate summary
            summary = self.summarize_text(text, **kwargs)
            
            # Calculate compression ratio
            original_length = len(text.split())
            summary_length = len(summary.split())
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            # Generate extractive summary for comparison
            extractive_summary = self.extractive_summarize(text, num_sentences=5)
            
            # Evaluate both summaries
            abstractive_metrics = self.evaluate_summary(text, summary)
            extractive_metrics = self.evaluate_summary(text, extractive_summary)
            
            result = {
                'abstractive_summary': summary,
                'extractive_summary': extractive_summary,
                'compression_ratio': compression_ratio,
                'original_length': original_length,
                'summary_length': summary_length,
                'abstractive_metrics': abstractive_metrics,
                'extractive_metrics': extractive_metrics,
                'model_name': self.model_name
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating abstractive summary: {str(e)}")
            raise
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input texts
            **kwargs: Additional parameters for summarization
            
        Returns:
            List of summary dictionaries
        """
        results = []
        for i, text in enumerate(texts):
            try:
                logger.info(f"Processing text {i+1}/{len(texts)}")
                result = self.generate_abstractive_summary(text, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing text {i+1}: {str(e)}")
                results.append({
                    'abstractive_summary': '',
                    'extractive_summary': '',
                    'error': str(e)
                })
        
        return results
    
    def save_model(self, save_path: Union[str, Path]):
        """Save the model and tokenizer."""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def log_to_mlflow(self, experiment_name: str = "summarization", 
                     run_name: Optional[str] = None):
        """Log model to MLflow."""
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=run_name):
                # Log model
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": self.model,
                        "tokenizer": self.tokenizer
                    },
                    artifact_path="summarizer",
                    registered_model_name=f"summarizer_{self.model_name.replace('/', '_')}"
                )
                
                # Log parameters
                mlflow.log_params({
                    "model_name": self.model_name,
                    "max_length": self.max_length,
                    "device": self.device
                })
                
                logger.info("Model logged to MLflow")
                
        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
            raise

class MultiModelSummarizer:
    """Ensemble summarizer using multiple transformer models."""
    
    def __init__(self, model_names: List[str] = None):
        """
        Initialize multi-model summarizer.
        
        Args:
            model_names: List of model names to use
        """
        if model_names is None:
            model_names = [
                "facebook/bart-large-cnn",
                "google/pegasus-xsum",
                "t5-base"
            ]
        
        self.models = {}
        for model_name in model_names:
            try:
                self.models[model_name] = TransformerSummarizer(model_name)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
    
    def ensemble_summarize(self, text: str, method: str = "voting") -> Dict[str, Any]:
        """
        Generate ensemble summary using multiple models.
        
        Args:
            text: Input text
            method: Ensemble method ('voting', 'averaging', 'best')
            
        Returns:
            Ensemble summary results
        """
        try:
            summaries = {}
            metrics = {}
            
            # Generate summaries from all models
            for model_name, model in self.models.items():
                try:
                    result = model.generate_abstractive_summary(text)
                    summaries[model_name] = result['abstractive_summary']
                    metrics[model_name] = result['abstractive_metrics']
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {str(e)}")
                    continue
            
            if not summaries:
                raise ValueError("No models produced valid summaries")
            
            # Combine summaries based on method
            if method == "voting":
                # Simple voting - use the most common words
                all_words = []
                for summary in summaries.values():
                    all_words.extend(summary.lower().split())
                
                word_counts = {}
                for word in all_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                # Select top words and reconstruct
                top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:50]
                ensemble_summary = " ".join([word for word, count in top_words])
                
            elif method == "averaging":
                # Average the summaries (simplified)
                all_summaries = list(summaries.values())
                ensemble_summary = " ".join(all_summaries)
                
            elif method == "best":
                # Use the model with best ROUGE score
                best_model = max(metrics.keys(), 
                               key=lambda x: metrics[x].get('rouge1', 0))
                ensemble_summary = summaries[best_model]
            
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
            
            return {
                'ensemble_summary': ensemble_summary,
                'individual_summaries': summaries,
                'individual_metrics': metrics,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble summarization: {str(e)}")
            raise

def main():
    """Example usage of the TransformerSummarizer."""
    # Initialize summarizer
    summarizer = TransformerSummarizer("facebook/bart-large-cnn")
    
    # Example text
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    
    The scope of AI is disputed: as machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
    """
    
    # Generate summary
    result = summarizer.generate_abstractive_summary(text)
    
    print("=== ABSTRACTIVE SUMMARY ===")
    print(result['abstractive_summary'])
    print(f"\nCompression ratio: {result['compression_ratio']:.2f}")
    print(f"ROUGE-1: {result['abstractive_metrics'].get('rouge1', 0):.3f}")

if __name__ == "__main__":
    main()
