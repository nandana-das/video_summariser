"""
Automated model training and fine-tuning capabilities for video summarization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import yaml
from datetime import datetime
import wandb
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset
import optuna
# from optuna.integration import PyTorchLightningPruningCallback  # Optional dependency
import mlflow
import mlflow.pytorch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

class VideoSummarizationDataset(Dataset):
    """Custom dataset for video summarization."""
    
    def __init__(self, texts: List[str], summaries: List[str], 
                 tokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of input texts
            summaries: List of target summaries
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        summary = str(self.summaries[idx])
        
        # Tokenize inputs
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            summary,
            max_length=self.max_length // 2,  # Summaries are typically shorter
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class ModelTrainer:
    """Automated model training and fine-tuning."""
    
    def __init__(self, model_name: str = "facebook/bart-base",
                 device: str = "auto", use_wandb: bool = False):
        """
        Initialize model trainer.
        
        Args:
            model_name: Name of the base model
            device: Device for training
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_wandb = use_wandb
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.training_args = None
        self.trainer = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        logger.info(f"ModelTrainer initialized with {model_name} on {self.device}")
    
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
    
    def load_model(self, model_name: Optional[str] = None):
        """Load model and tokenizer."""
        try:
            if model_name is None:
                model_name = self.model_name
            
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_data(self, train_texts: List[str], train_summaries: List[str],
                    val_texts: Optional[List[str]] = None,
                    val_summaries: Optional[List[str]] = None,
                    test_size: float = 0.2) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data for training.
        
        Args:
            train_texts: Training input texts
            train_summaries: Training target summaries
            val_texts: Validation input texts (optional)
            val_summaries: Validation target summaries (optional)
            test_size: Fraction of data to use for validation if val data not provided
            
        Returns:
            Training and validation data loaders
        """
        try:
            # Split data if validation data not provided
            if val_texts is None or val_summaries is None:
                from sklearn.model_selection import train_test_split
                train_texts, val_texts, train_summaries, val_summaries = train_test_split(
                    train_texts, train_summaries, test_size=test_size, random_state=42
                )
            
            # Create datasets
            train_dataset = VideoSummarizationDataset(
                train_texts, train_summaries, self.tokenizer
            )
            val_dataset = VideoSummarizationDataset(
                val_texts, val_summaries, self.tokenizer
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=8, shuffle=True, num_workers=2
            )
            val_loader = DataLoader(
                val_dataset, batch_size=8, shuffle=False, num_workers=2
            )
            
            logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def setup_training_args(self, output_dir: Union[str, Path],
                           num_epochs: int = 3,
                           learning_rate: float = 5e-5,
                           batch_size: int = 8,
                           warmup_steps: int = 500,
                           weight_decay: float = 0.01,
                           logging_steps: int = 100,
                           eval_steps: int = 500,
                           save_steps: int = 1000,
                           **kwargs) -> TrainingArguments:
        """
        Setup training arguments.
        
        Args:
            output_dir: Directory to save model outputs
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            logging_steps: Steps between logging
            eval_steps: Steps between evaluation
            save_steps: Steps between saving
            **kwargs: Additional training arguments
            
        Returns:
            Training arguments
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Default training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                logging_dir=str(output_dir / "logs"),
                logging_steps=logging_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb" if self.use_wandb else None,
                **kwargs
            )
            
            self.training_args = training_args
            logger.info("Training arguments configured")
            
            return training_args
            
        except Exception as e:
            logger.error(f"Error setting up training arguments: {str(e)}")
            raise
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              training_args: Optional[TrainingArguments] = None,
              custom_metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            training_args: Training arguments
            custom_metrics: Custom evaluation metrics
            
        Returns:
            Training results
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            if training_args is None:
                training_args = self.training_args
            
            if training_args is None:
                raise ValueError("Training arguments not set. Call setup_training_args() first.")
            
            # Initialize Weights & Biases if enabled
            if self.use_wandb:
                wandb.init(
                    project="video-summarization",
                    config=training_args.to_dict()
                )
            
            # Create data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_loader.dataset,
                eval_dataset=val_loader.dataset if val_loader else None,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics if custom_metrics else None,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Start training
            logger.info("Starting training...")
            training_result = self.trainer.train()
            
            # Save model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(training_args.output_dir)
            
            # Update training history
            self.training_history['train_loss'].extend(training_result.log_history)
            
            # Log to MLflow
            self._log_training_to_mlflow(training_result)
            
            logger.info("Training completed successfully")
            
            return {
                'training_result': training_result,
                'model_path': training_args.output_dir,
                'training_history': self.training_history
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        try:
            predictions, labels = eval_pred
            
            # Decode predictions
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Calculate ROUGE scores
            from rouge_score import rouge_scorer
            rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, label in zip(decoded_preds, decoded_labels):
                scores = rouge.score(label, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(rouge1_scores),
                'rouge2': np.mean(rouge2_scores),
                'rougeL': np.mean(rougeL_scores)
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {}
    
    def _log_training_to_mlflow(self, training_result):
        """Log training results to MLflow."""
        try:
            with mlflow.start_run():
                # Log model
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    artifact_path="model",
                    registered_model_name=f"summarizer_{self.model_name.replace('/', '_')}"
                )
                
                # Log parameters
                mlflow.log_params({
                    "model_name": self.model_name,
                    "num_epochs": self.training_args.num_train_epochs,
                    "learning_rate": self.training_args.learning_rate,
                    "batch_size": self.training_args.per_device_train_batch_size
                })
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": training_result.training_loss,
                    "eval_loss": training_result.eval_loss if hasattr(training_result, 'eval_loss') else 0
                })
                
                logger.info("Training results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"Error logging to MLflow: {str(e)}")
    
    def fine_tune(self, train_texts: List[str], train_summaries: List[str],
                  val_texts: Optional[List[str]] = None,
                  val_summaries: Optional[List[str]] = None,
                  num_epochs: int = 3,
                  learning_rate: float = 5e-5,
                  output_dir: Union[str, Path] = "fine_tuned_model") -> Dict[str, Any]:
        """
        Fine-tune the model on custom data.
        
        Args:
            train_texts: Training input texts
            train_summaries: Training target summaries
            val_texts: Validation input texts
            val_summaries: Validation target summaries
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            output_dir: Output directory for fine-tuned model
            
        Returns:
            Fine-tuning results
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Prepare data
            train_loader, val_loader = self.prepare_data(
                train_texts, train_summaries, val_texts, val_summaries
            )
            
            # Setup training arguments
            training_args = self.setup_training_args(
                output_dir=output_dir,
                num_epochs=num_epochs,
                learning_rate=learning_rate
            )
            
            # Train model
            results = self.train(train_loader, val_loader, training_args)
            
            logger.info("Fine-tuning completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def hyperparameter_search(self, train_texts: List[str], train_summaries: List[str],
                             val_texts: List[str], val_summaries: List[str],
                             n_trials: int = 20) -> Dict[str, Any]:
        """
        Perform hyperparameter search using Optuna.
        
        Args:
            train_texts: Training input texts
            train_summaries: Training target summaries
            val_texts: Validation input texts
            val_summaries: Validation target summaries
            n_trials: Number of trials for hyperparameter search
            
        Returns:
            Best hyperparameters and results
        """
        try:
            def objective(trial):
                # Suggest hyperparameters
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
                batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
                num_epochs = trial.suggest_int('num_epochs', 1, 5)
                weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
                
                # Load model
                self.load_model()
                
                # Prepare data
                train_loader, val_loader = self.prepare_data(
                    train_texts, train_summaries, val_texts, val_summaries
                )
                
                # Setup training arguments
                training_args = self.setup_training_args(
                    output_dir=f"hyperparameter_search/trial_{trial.number}",
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    weight_decay=weight_decay
                )
                
                # Train model
                results = self.train(train_loader, val_loader, training_args)
                
                # Return validation loss as objective
                return results['training_result'].eval_loss
            
            # Create study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Hyperparameter search completed. Best value: {best_value}")
            logger.info(f"Best parameters: {best_params}")
            
            return {
                'best_params': best_params,
                'best_value': best_value,
                'study': study
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter search: {str(e)}")
            raise
    
    def evaluate_model(self, test_texts: List[str], test_summaries: List[str],
                      model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            test_texts: Test input texts
            test_summaries: Test target summaries
            model_path: Path to trained model (optional)
            
        Returns:
            Evaluation results
        """
        try:
            # Load model if path provided
            if model_path is not None:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.model.to(self.device)
            
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model not loaded")
            
            # Generate predictions
            predictions = []
            self.model.eval()
            
            with torch.no_grad():
                for text in test_texts:
                    inputs = self.tokenizer(
                        text,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predictions.append(prediction)
            
            # Calculate metrics
            from ml_models.model_evaluation import ModelEvaluator
            evaluator = ModelEvaluator()
            
            evaluation_results = evaluator.evaluate_summarization(
                test_summaries, predictions, f"fine_tuned_{self.model_name}"
            )
            
            logger.info("Model evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def save_model(self, save_path: Union[str, Path]):
        """Save the trained model."""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            if self.model is not None:
                self.model.save_pretrained(save_path)
            
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_path)
            
            # Save training history
            with open(save_path / "training_history.json", "w") as f:
                json.dump(self.training_history, f, indent=2)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_trained_model(self, model_path: Union[str, Path]):
        """Load a trained model."""
        try:
            model_path = Path(model_path)
            
            # Load model and tokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model.to(self.device)
            
            # Load training history if available
            history_path = model_path / "training_history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    self.training_history = json.load(f)
            
            logger.info(f"Trained model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            raise

class AutoMLTrainer:
    """Automated ML training with multiple model types."""
    
    def __init__(self, task_type: str = "summarization"):
        """
        Initialize AutoML trainer.
        
        Args:
            task_type: Type of task ('summarization', 'classification', 'regression')
        """
        self.task_type = task_type
        self.models = {}
        self.results = {}
        
        logger.info(f"AutoMLTrainer initialized for {task_type}")
    
    def train_multiple_models(self, train_data: Dict[str, Any],
                             val_data: Optional[Dict[str, Any]] = None,
                             model_types: List[str] = None) -> Dict[str, Any]:
        """
        Train multiple models and compare performance.
        
        Args:
            train_data: Training data
            val_data: Validation data
            model_types: List of model types to train
            
        Returns:
            Training results for all models
        """
        try:
            if model_types is None:
                model_types = ["bart", "t5", "pegasus"]
            
            for model_type in model_types:
                logger.info(f"Training {model_type} model...")
                
                # Map model type to model name
                model_name_map = {
                    "bart": "facebook/bart-base",
                    "t5": "t5-base",
                    "pegasus": "google/pegasus-xsum"
                }
                
                if model_type not in model_name_map:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                # Create trainer
                trainer = ModelTrainer(model_name_map[model_type])
                
                # Train model
                results = trainer.fine_tune(
                    train_data["texts"],
                    train_data["summaries"],
                    val_data["texts"] if val_data else None,
                    val_data["summaries"] if val_data else None
                )
                
                # Store results
                self.models[model_type] = trainer
                self.results[model_type] = results
                
                logger.info(f"{model_type} training completed")
            
            # Compare models
            comparison = self._compare_models()
            
            return {
                'models': self.models,
                'results': self.results,
                'comparison': comparison
            }
            
        except Exception as e:
            logger.error(f"Error in multi-model training: {str(e)}")
            raise
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare performance of different models."""
        try:
            comparison = {}
            
            for model_type, results in self.results.items():
                if 'training_result' in results:
                    training_result = results['training_result']
                    comparison[model_type] = {
                        'train_loss': training_result.training_loss,
                        'eval_loss': getattr(training_result, 'eval_loss', None),
                        'model_path': results.get('model_path', '')
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {}

def main():
    """Example usage of the ModelTrainer."""
    # Initialize trainer
    trainer = ModelTrainer("facebook/bart-base")
    
    # Example data
    train_texts = [
        "This is a long text about machine learning and artificial intelligence.",
        "Another long text about deep learning and neural networks."
    ]
    train_summaries = [
        "Text about ML and AI.",
        "Text about deep learning."
    ]
    
    # Fine-tune model
    results = trainer.fine_tune(train_texts, train_summaries)
    
    print("Fine-tuning completed!")
    print(f"Model saved to: {results['model_path']}")

if __name__ == "__main__":
    main()
