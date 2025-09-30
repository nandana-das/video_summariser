"""
Comprehensive model evaluation and validation framework for video summarization.
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
import optuna
from optuna.integration import MLflowCallback
import mlflow
import mlflow.pytorch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize model evaluator.
        
        Args:
            device: Device for evaluation
        """
        self.device = self._get_device(device)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Download required NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass
        
        logger.info(f"ModelEvaluator initialized on {self.device}")
    
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
    
    def evaluate_summarization(self, references: List[str], 
                              predictions: List[str],
                              model_name: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate text summarization performance.
        
        Args:
            references: List of reference summaries
            predictions: List of predicted summaries
            model_name: Name of the model being evaluated
            
        Returns:
            Comprehensive evaluation metrics
        """
        try:
            logger.info(f"Evaluating summarization for {len(references)} samples")
            
            # ROUGE scores
            rouge_scores = self._calculate_rouge_scores(references, predictions)
            
            # BERTScore
            bert_scores = self._calculate_bert_scores(references, predictions)
            
            # BLEU scores
            bleu_scores = self._calculate_bleu_scores(references, predictions)
            
            # METEOR score
            meteor_scores = self._calculate_meteor_scores(references, predictions)
            
            # NIST score
            nist_scores = self._calculate_nist_scores(references, predictions)
            
            # Compression ratio
            compression_ratios = self._calculate_compression_ratios(references, predictions)
            
            # Readability metrics
            readability_metrics = self._calculate_readability_metrics(predictions)
            
            # Combine all metrics
            evaluation_results = {
                'model_name': model_name,
                'num_samples': len(references),
                'rouge_scores': rouge_scores,
                'bert_scores': bert_scores,
                'bleu_scores': bleu_scores,
                'meteor_scores': meteor_scores,
                'nist_scores': nist_scores,
                'compression_ratios': compression_ratios,
                'readability_metrics': readability_metrics,
                'overall_metrics': self._calculate_overall_metrics(
                    rouge_scores, bert_scores, bleu_scores, 
                    meteor_scores, nist_scores, compression_ratios
                )
            }
            
            logger.info("Summarization evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating summarization: {str(e)}")
            raise
    
    def _calculate_rouge_scores(self, references: List[str], 
                               predictions: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for ref, pred in zip(references, predictions):
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1_mean': np.mean(rouge1_scores),
                'rouge1_std': np.std(rouge1_scores),
                'rouge2_mean': np.mean(rouge2_scores),
                'rouge2_std': np.std(rouge2_scores),
                'rougeL_mean': np.mean(rougeL_scores),
                'rougeL_std': np.std(rougeL_scores),
                'rouge1_scores': rouge1_scores,
                'rouge2_scores': rouge2_scores,
                'rougeL_scores': rougeL_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            return {}
    
    def _calculate_bert_scores(self, references: List[str], 
                              predictions: List[str]) -> Dict[str, float]:
        """Calculate BERTScore."""
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            
            return {
                'bert_precision_mean': P.mean().item(),
                'bert_precision_std': P.std().item(),
                'bert_recall_mean': R.mean().item(),
                'bert_recall_std': R.std().item(),
                'bert_f1_mean': F1.mean().item(),
                'bert_f1_std': F1.std().item(),
                'bert_precision_scores': P.tolist(),
                'bert_recall_scores': R.tolist(),
                'bert_f1_scores': F1.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {str(e)}")
            return {}
    
    def _calculate_bleu_scores(self, references: List[str], 
                              predictions: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores."""
        try:
            bleu1_scores = []
            bleu2_scores = []
            bleu3_scores = []
            bleu4_scores = []
            
            smoothing = SmoothingFunction().method1
            
            for ref, pred in zip(references, predictions):
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                
                bleu1_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing))
                bleu2_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing))
                bleu3_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing))
                bleu4_scores.append(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
            
            return {
                'bleu1_mean': np.mean(bleu1_scores),
                'bleu1_std': np.std(bleu1_scores),
                'bleu2_mean': np.mean(bleu2_scores),
                'bleu2_std': np.std(bleu2_scores),
                'bleu3_mean': np.mean(bleu3_scores),
                'bleu3_std': np.std(bleu3_scores),
                'bleu4_mean': np.mean(bleu4_scores),
                'bleu4_std': np.std(bleu4_scores),
                'bleu1_scores': bleu1_scores,
                'bleu2_scores': bleu2_scores,
                'bleu3_scores': bleu3_scores,
                'bleu4_scores': bleu4_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating BLEU scores: {str(e)}")
            return {}
    
    def _calculate_meteor_scores(self, references: List[str], 
                                predictions: List[str]) -> Dict[str, float]:
        """Calculate METEOR scores."""
        try:
            meteor_scores = []
            
            for ref, pred in zip(references, predictions):
                score = meteor_score([ref.split()], pred.split())
                meteor_scores.append(score)
            
            return {
                'meteor_mean': np.mean(meteor_scores),
                'meteor_std': np.std(meteor_scores),
                'meteor_scores': meteor_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating METEOR scores: {str(e)}")
            return {}
    
    def _calculate_nist_scores(self, references: List[str], 
                              predictions: List[str]) -> Dict[str, float]:
        """Calculate NIST scores."""
        try:
            nist_scores = []
            
            for ref, pred in zip(references, predictions):
                score = sentence_nist([ref.split()], pred.split())
                nist_scores.append(score)
            
            return {
                'nist_mean': np.mean(nist_scores),
                'nist_std': np.std(nist_scores),
                'nist_scores': nist_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating NIST scores: {str(e)}")
            return {}
    
    def _calculate_compression_ratios(self, references: List[str], 
                                    predictions: List[str]) -> Dict[str, float]:
        """Calculate compression ratios."""
        try:
            ratios = []
            
            for ref, pred in zip(references, predictions):
                ref_length = len(ref.split())
                pred_length = len(pred.split())
                ratio = pred_length / ref_length if ref_length > 0 else 0
                ratios.append(ratio)
            
            return {
                'compression_ratio_mean': np.mean(ratios),
                'compression_ratio_std': np.std(ratios),
                'compression_ratios': ratios
            }
            
        except Exception as e:
            logger.error(f"Error calculating compression ratios: {str(e)}")
            return {}
    
    def _calculate_readability_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Calculate readability metrics."""
        try:
            import textstat
            
            flesch_scores = []
            fk_scores = []
            gunning_fog_scores = []
            
            for pred in predictions:
                flesch_scores.append(textstat.flesch_reading_ease(pred))
                fk_scores.append(textstat.flesch_kincaid_grade(pred))
                gunning_fog_scores.append(textstat.gunning_fog(pred))
            
            return {
                'flesch_mean': np.mean(flesch_scores),
                'flesch_std': np.std(flesch_scores),
                'fk_mean': np.mean(fk_scores),
                'fk_std': np.std(fk_scores),
                'gunning_fog_mean': np.mean(gunning_fog_scores),
                'gunning_fog_std': np.std(gunning_fog_scores),
                'flesch_scores': flesch_scores,
                'fk_scores': fk_scores,
                'gunning_fog_scores': gunning_fog_scores
            }
            
        except ImportError:
            logger.warning("textstat not available for readability metrics")
            return {}
        except Exception as e:
            logger.error(f"Error calculating readability metrics: {str(e)}")
            return {}
    
    def _calculate_overall_metrics(self, rouge_scores: Dict, bert_scores: Dict,
                                  bleu_scores: Dict, meteor_scores: Dict,
                                  nist_scores: Dict, compression_ratios: Dict) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        try:
            overall_metrics = {}
            
            # ROUGE-1 as primary metric
            if 'rouge1_mean' in rouge_scores:
                overall_metrics['primary_metric'] = rouge_scores['rouge1_mean']
            
            # Composite score (weighted average)
            composite_score = 0
            weights = {'rouge1': 0.3, 'rouge2': 0.2, 'rougeL': 0.2, 'bert_f1': 0.2, 'meteor': 0.1}
            
            if 'rouge1_mean' in rouge_scores:
                composite_score += weights['rouge1'] * rouge_scores['rouge1_mean']
            if 'rouge2_mean' in rouge_scores:
                composite_score += weights['rouge2'] * rouge_scores['rouge2_mean']
            if 'rougeL_mean' in rouge_scores:
                composite_score += weights['rougeL'] * rouge_scores['rougeL_mean']
            if 'bert_f1_mean' in bert_scores:
                composite_score += weights['bert_f1'] * bert_scores['bert_f1_mean']
            if 'meteor_mean' in meteor_scores:
                composite_score += weights['meteor'] * meteor_scores['meteor_mean']
            
            overall_metrics['composite_score'] = composite_score
            
            # Quality indicators
            if 'compression_ratio_mean' in compression_ratios:
                ratio = compression_ratios['compression_ratio_mean']
                if 0.1 <= ratio <= 0.5:
                    overall_metrics['compression_quality'] = 'good'
                elif 0.05 <= ratio <= 0.7:
                    overall_metrics['compression_quality'] = 'acceptable'
                else:
                    overall_metrics['compression_quality'] = 'poor'
            
            return overall_metrics
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {str(e)}")
            return {}
    
    def evaluate_classification(self, y_true: List[int], y_pred: List[int],
                               class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            Classification evaluation metrics
        """
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            report = classification_report(y_true, y_pred, 
                                        target_names=class_names, 
                                        output_dict=True)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Error evaluating classification: {str(e)}")
            raise
    
    def evaluate_regression(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Evaluate regression performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Regression evaluation metrics
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2
            }
            
        except Exception as e:
            logger.error(f"Error evaluating regression: {str(e)}")
            raise
    
    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            if scoring in ['accuracy', 'precision', 'recall', 'f1']:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return {
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'cv_folds': cv_folds,
                'scoring': scoring
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def plot_evaluation_results(self, evaluation_results: Dict[str, Any],
                               save_path: Optional[Union[str, Path]] = None):
        """Plot evaluation results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # ROUGE scores
            if 'rouge_scores' in evaluation_results:
                rouge_data = evaluation_results['rouge_scores']
                rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
                rouge_means = [rouge_data.get(f'{metric}_mean', 0) for metric in rouge_metrics]
                rouge_stds = [rouge_data.get(f'{metric}_std', 0) for metric in rouge_metrics]
                
                axes[0, 0].bar(rouge_metrics, rouge_means, yerr=rouge_stds, capsize=5)
                axes[0, 0].set_title('ROUGE Scores')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_ylim(0, 1)
            
            # BERT scores
            if 'bert_scores' in evaluation_results:
                bert_data = evaluation_results['bert_scores']
                bert_metrics = ['bert_precision', 'bert_recall', 'bert_f1']
                bert_means = [bert_data.get(f'{metric}_mean', 0) for metric in bert_metrics]
                bert_stds = [bert_data.get(f'{metric}_std', 0) for metric in bert_metrics]
                
                axes[0, 1].bar(bert_metrics, bert_means, yerr=bert_stds, capsize=5)
                axes[0, 1].set_title('BERT Scores')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_ylim(0, 1)
            
            # BLEU scores
            if 'bleu_scores' in evaluation_results:
                bleu_data = evaluation_results['bleu_scores']
                bleu_metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
                bleu_means = [bleu_data.get(f'{metric}_mean', 0) for metric in bleu_metrics]
                bleu_stds = [bleu_data.get(f'{metric}_std', 0) for metric in bleu_metrics]
                
                axes[1, 0].bar(bleu_metrics, bleu_means, yerr=bleu_stds, capsize=5)
                axes[1, 0].set_title('BLEU Scores')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_ylim(0, 1)
            
            # Compression ratios
            if 'compression_ratios' in evaluation_results:
                comp_data = evaluation_results['compression_ratios']
                ratios = comp_data.get('compression_ratios', [])
                
                axes[1, 1].hist(ratios, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Compression Ratios Distribution')
                axes[1, 1].set_xlabel('Compression Ratio')
                axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Evaluation plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting evaluation results: {str(e)}")
            raise
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 save_path: Union[str, Path]):
        """Generate comprehensive evaluation report."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== MODEL EVALUATION REPORT ===\n\n")
                
                f.write(f"Model: {evaluation_results.get('model_name', 'Unknown')}\n")
                f.write(f"Number of samples: {evaluation_results.get('num_samples', 0)}\n\n")
                
                # ROUGE scores
                if 'rouge_scores' in evaluation_results:
                    f.write("=== ROUGE SCORES ===\n")
                    rouge_data = evaluation_results['rouge_scores']
                    f.write(f"ROUGE-1: {rouge_data.get('rouge1_mean', 0):.4f} ± {rouge_data.get('rouge1_std', 0):.4f}\n")
                    f.write(f"ROUGE-2: {rouge_data.get('rouge2_mean', 0):.4f} ± {rouge_data.get('rouge2_std', 0):.4f}\n")
                    f.write(f"ROUGE-L: {rouge_data.get('rougeL_mean', 0):.4f} ± {rouge_data.get('rougeL_std', 0):.4f}\n\n")
                
                # BERT scores
                if 'bert_scores' in evaluation_results:
                    f.write("=== BERT SCORES ===\n")
                    bert_data = evaluation_results['bert_scores']
                    f.write(f"BERT Precision: {bert_data.get('bert_precision_mean', 0):.4f} ± {bert_data.get('bert_precision_std', 0):.4f}\n")
                    f.write(f"BERT Recall: {bert_data.get('bert_recall_mean', 0):.4f} ± {bert_data.get('bert_recall_std', 0):.4f}\n")
                    f.write(f"BERT F1: {bert_data.get('bert_f1_mean', 0):.4f} ± {bert_data.get('bert_f1_std', 0):.4f}\n\n")
                
                # Overall metrics
                if 'overall_metrics' in evaluation_results:
                    f.write("=== OVERALL METRICS ===\n")
                    overall = evaluation_results['overall_metrics']
                    f.write(f"Primary Metric (ROUGE-1): {overall.get('primary_metric', 0):.4f}\n")
                    f.write(f"Composite Score: {overall.get('composite_score', 0):.4f}\n")
                    f.write(f"Compression Quality: {overall.get('compression_quality', 'Unknown')}\n\n")
            
            logger.info(f"Evaluation report saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, direction: str = "maximize"):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            direction: Optimization direction ('maximize' or 'minimize')
        """
        self.direction = direction
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def optimize(self, objective_func: Callable, n_trials: int = 100,
                study_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Objective function to optimize
            n_trials: Number of trials
            study_name: Name for the study
            
        Returns:
            Optimization results
        """
        try:
            # Create study
            self.study = optuna.create_study(
                direction=self.direction,
                study_name=study_name
            )
            
            # Add MLflow callback
            mlflow_callback = MLflowCallback(
                tracking_uri="sqlite:///mlflow.db",
                metric_name="objective_value"
            )
            
            # Optimize
            self.study.optimize(
                objective_func, 
                n_trials=n_trials,
                callbacks=[mlflow_callback]
            )
            
            # Get best results
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            
            return {
                'best_params': self.best_params,
                'best_value': self.best_value,
                'n_trials': n_trials,
                'study_name': study_name
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise
    
    def plot_optimization_history(self, save_path: Optional[Union[str, Path]] = None):
        """Plot optimization history."""
        try:
            if self.study is None:
                raise ValueError("No study found. Run optimize() first.")
            
            fig = optuna.visualization.plot_optimization_history(self.study)
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Optimization history plot saved to {save_path}")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting optimization history: {str(e)}")
            raise

def main():
    """Example usage of the ModelEvaluator."""
    evaluator = ModelEvaluator()
    
    # Example evaluation
    references = [
        "This is a reference summary about machine learning.",
        "Another reference summary about deep learning."
    ]
    predictions = [
        "This is a predicted summary about machine learning.",
        "Another predicted summary about deep learning."
    ]
    
    results = evaluator.evaluate_summarization(references, predictions, "example_model")
    
    print("Evaluation Results:")
    print(f"ROUGE-1: {results['rouge_scores']['rouge1_mean']:.4f}")
    print(f"BERT F1: {results['bert_scores']['bert_f1_mean']:.4f}")
    print(f"Composite Score: {results['overall_metrics']['composite_score']:.4f}")

if __name__ == "__main__":
    main()
