"""
MLflow integration for experiment tracking, model management, and MLOps.
"""
import mlflow
import mlflow.pytorch
import mlflow.transformers
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.onnx
import os
import json
import pickle
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from config import OUTPUT_DIR

logger = setup_logger(__name__)

class MLflowManager:
    """MLflow integration for experiment tracking and model management."""
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", 
                 experiment_name: str = "video_summarizer"):
        """
        Initialize MLflow manager.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.current_run = None
        
        # Set up MLflow
        # Use HTTP URI if MLflow server is running, otherwise use SQLite
        if tracking_uri.startswith('sqlite:'):
            # Check if MLflow server is running on localhost:5000
            try:
                import requests
                response = requests.get('http://localhost:5000/health', timeout=2)
                if response.status_code == 200:
                    # MLflow server is running, use HTTP URI
                    mlflow.set_tracking_uri('http://localhost:5000')
                    logger.info("Using MLflow server at http://localhost:5000")
                else:
                    # Use SQLite as fallback
                    mlflow.set_tracking_uri(tracking_uri)
                    logger.info("Using SQLite backend for MLflow")
            except:
                # MLflow server not running, use SQLite as fallback
                mlflow.set_tracking_uri(tracking_uri)
                logger.info("Using SQLite backend for MLflow")
        else:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"Using MLflow tracking URI: {tracking_uri}")
        
        self.experiment_id = self._get_or_create_experiment()
        
        logger.info(f"MLflowManager initialized with experiment: {experiment_name}")
    
    def _get_or_create_experiment(self) -> str:
        """Get or create MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error setting up experiment: {str(e)}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            Run ID
        """
        try:
            if run_name is None:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                # Clean run name to avoid Unicode issues
                run_name = "".join(c for c in run_name if c.isalnum() or c in (' ', '-', '_')).strip()
                if not run_name:
                    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name) as run:
                self.current_run = run
                
                # Add tags
                if tags:
                    mlflow.set_tags(tags)
                
                # Add default tags
                mlflow.set_tags({
                    "framework": "pytorch",
                    "task": "video_summarization",
                    "created_at": datetime.now().isoformat()
                })
                
                logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Error starting run: {str(e)}")
            raise
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            if self.current_run:
                mlflow.end_run()
                self.current_run = None
                logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Error ending run: {str(e)}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            if self.current_run is None:
                raise ValueError("No active run. Call start_run() first.")
            
            # Convert non-serializable parameters
            serializable_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    serializable_params[key] = value
                else:
                    serializable_params[key] = str(value)
            
            mlflow.log_params(serializable_params)
            logger.info(f"Logged {len(serializable_params)} parameters")
            
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        try:
            if self.current_run is None:
                raise ValueError("No active run. Call start_run() first.")
            
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise
    
    def log_model(self, model: Any, model_name: str, 
                  model_type: str = "pytorch", 
                  artifact_path: str = "model",
                  **kwargs):
        """
        Log model to MLflow.
        
        Args:
            model: Model to log
            model_name: Name for the model
            model_type: Type of model ('pytorch', 'transformers', 'sklearn', 'tensorflow', 'onnx')
            artifact_path: Path for the model artifact
            **kwargs: Additional arguments for model logging
        """
        try:
            if self.current_run is None:
                raise ValueError("No active run. Call start_run() first.")
            
            if model_type == "pytorch":
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=model_name,
                    **kwargs
                )
            elif model_type == "transformers":
                mlflow.transformers.log_model(
                    transformers_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=model_name,
                    **kwargs
                )
            elif model_type == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=model_name,
                    **kwargs
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    model=model,
                    artifact_path=artifact_path,
                    registered_model_name=model_name,
                    **kwargs
                )
            elif model_type == "onnx":
                mlflow.onnx.log_model(
                    onnx_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=model_name,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Logged {model_type} model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise
    
    def log_artifacts(self, local_dir: Union[str, Path], 
                     artifact_path: Optional[str] = None):
        """Log artifacts to MLflow."""
        try:
            if self.current_run is None:
                raise ValueError("No active run. Call start_run() first.")
            
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from {local_dir}")
            
        except Exception as e:
            logger.error(f"Error logging artifacts: {str(e)}")
            raise
    
    def log_figure(self, figure, artifact_file: str = "plot.png"):
        """Log matplotlib figure to MLflow."""
        try:
            if self.current_run is None:
                raise ValueError("No active run. Call start_run() first.")
            
            mlflow.log_figure(figure, artifact_file)
            logger.info(f"Logged figure: {artifact_file}")
            
        except Exception as e:
            logger.error(f"Error logging figure: {str(e)}")
            raise
    
    def log_text(self, text: str, artifact_file: str = "text.txt"):
        """Log text to MLflow."""
        try:
            if self.current_run is None:
                raise ValueError("No active run. Call start_run() first.")
            
            # Sanitize text to remove emoji and special characters that cause encoding issues
            import re
            sanitized_text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
            sanitized_text = sanitized_text.strip()
            
            if not sanitized_text:
                sanitized_text = "Text content processed successfully"
            
            mlflow.log_text(sanitized_text, artifact_file)
            logger.info(f"Logged text: {artifact_file}")
            
        except Exception as e:
            logger.error(f"Error logging text: {str(e)}")
            raise
    
    def log_json(self, data: Dict[str, Any], artifact_file: str = "data.json"):
        """Log JSON data to MLflow."""
        try:
            if self.current_run is None:
                raise ValueError("No active run. Call start_run() first.")
            
            # Sanitize data to remove problematic characters
            def sanitize_data(obj):
                if isinstance(obj, str):
                    import re
                    return re.sub(r'[^\x00-\x7F]+', '', obj)  # Remove non-ASCII characters
                elif isinstance(obj, dict):
                    return {k: sanitize_data(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_data(item) for item in obj]
                else:
                    return obj
            
            sanitized_data = sanitize_data(data)
            mlflow.log_dict(sanitized_data, artifact_file)
            logger.info(f"Logged JSON: {artifact_file}")
            
        except Exception as e:
            logger.error(f"Error logging JSON: {str(e)}")
            raise
    
    def log_video_summarization_results(self, results: Dict[str, Any], 
                                      video_path: str, 
                                      model_name: str):
        """
        Log video summarization results to MLflow.
        
        Args:
            results: Summarization results
            video_path: Path to input video
            model_name: Name of the model used
        """
        try:
            # Log parameters
            params = {
                "video_path": video_path,
                "model_name": model_name,
                "input_type": "video"
            }
            
            if "summary_data" in results:
                summary_data = results["summary_data"]
                params.update({
                    "max_sentences": summary_data.get("metadata", {}).get("summary_sentence_count", 0),
                    "compression_ratio": summary_data.get("metadata", {}).get("compression_ratio", 0)
                })
            
            self.log_parameters(params)
            
            # Log metrics
            metrics = {}
            if "summary_data" in results and "metadata" in results["summary_data"]:
                metadata = results["summary_data"]["metadata"]
                metrics.update({
                    "original_sentence_count": metadata.get("original_sentence_count", 0),
                    "summary_sentence_count": metadata.get("summary_sentence_count", 0),
                    "compression_ratio": metadata.get("compression_ratio", 0),
                    "keyword_count": metadata.get("keyword_count", 0),
                    "action_item_count": metadata.get("action_item_count", 0),
                    "named_entity_count": metadata.get("named_entity_count", 0)
                })
            
            # Add evaluation metrics if available
            if "summary_data" in results and "abstractive_metrics" in results["summary_data"]:
                eval_metrics = results["summary_data"]["abstractive_metrics"]
                metrics.update({
                    "rouge1": eval_metrics.get("rouge1", 0),
                    "rouge2": eval_metrics.get("rouge2", 0),
                    "rougeL": eval_metrics.get("rougeL", 0),
                    "bert_f1": eval_metrics.get("bert_f1", 0)
                })
            
            self.log_metrics(metrics)
            
            # Log artifacts
            if "summary_path" in results:
                self.log_artifacts(str(Path(results["summary_path"]).parent))
            
            # Log summary text
            if "summary_data" in results and "summary" in results["summary_data"]:
                self.log_text(
                    results["summary_data"]["summary"], 
                    "summary.txt"
                )
            
            # Log full results as JSON
            self.log_json(results, "full_results.json")
            
            logger.info("Logged video summarization results to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging video summarization results: {str(e)}")
            raise
    
    def log_model_training(self, model: Any, training_history: Dict[str, List[float]], 
                          model_name: str, model_type: str = "pytorch"):
        """
        Log model training results to MLflow.
        
        Args:
            model: Trained model
            training_history: Training history with metrics
            model_name: Name for the model
            model_type: Type of model
        """
        try:
            # Log model
            self.log_model(model, model_name, model_type)
            
            # Log training metrics
            for epoch, metrics in enumerate(training_history.get("metrics", [])):
                self.log_metrics(metrics, step=epoch)
            
            # Log training history
            self.log_json(training_history, "training_history.json")
            
            # Log training plots
            if "loss" in training_history:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(training_history["loss"], label="Training Loss")
                if "val_loss" in training_history:
                    ax.plot(training_history["val_loss"], label="Validation Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Progress")
                ax.legend()
                ax.grid(True)
                self.log_figure(fig, "training_plot.png")
                plt.close(fig)
            
            logger.info("Logged model training to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging model training: {str(e)}")
            raise
    
    def get_best_model(self, metric_name: str = "rouge1", 
                      ascending: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best model based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether lower values are better
            
        Returns:
            Best model information
        """
        try:
            # Get all runs
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
            
            if runs.empty:
                logger.warning("No runs found")
                return None
            
            # Filter runs with the metric
            runs_with_metric = runs.dropna(subset=[f"metrics.{metric_name}"])
            
            if runs_with_metric.empty:
                logger.warning(f"No runs found with metric: {metric_name}")
                return None
            
            # Find best run
            best_run = runs_with_metric.loc[
                runs_with_metric[f"metrics.{metric_name}"].idxmax() if not ascending
                else runs_with_metric[f"metrics.{metric_name}"].idxmin()
            ]
            
            return {
                "run_id": best_run["run_id"],
                "metric_value": best_run[f"metrics.{metric_name}"],
                "model_name": best_run.get("params.model_name", "unknown"),
                "run_name": best_run.get("tags.mlflow.runName", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error getting best model: {str(e)}")
            return None
    
    def load_model(self, run_id: str, artifact_path: str = "model") -> Any:
        """
        Load a model from MLflow.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to the model artifact
            
        Returns:
            Loaded model
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from run {run_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison DataFrame
        """
        try:
            runs = mlflow.search_runs(run_ids=run_ids)
            
            # Select relevant columns
            comparison_cols = ["run_id", "run_name", "status", "start_time", "end_time"]
            
            # Add metric columns
            metric_cols = [col for col in runs.columns if col.startswith("metrics.")]
            comparison_cols.extend(metric_cols)
            
            # Add parameter columns
            param_cols = [col for col in runs.columns if col.startswith("params.")]
            comparison_cols.extend(param_cols)
            
            comparison_df = runs[comparison_cols]
            
            logger.info(f"Compared {len(run_ids)} runs")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing runs: {str(e)}")
            raise
    
    def export_experiment(self, output_path: Union[str, Path]):
        """Export experiment data to files."""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get all runs
            runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
            
            # Save runs data
            runs.to_csv(output_path / "runs.csv", index=False)
            
            # Save experiment metadata
            experiment = mlflow.get_experiment(self.experiment_id)
            experiment_data = {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "creation_time": experiment.creation_time,
                "last_update_time": experiment.last_update_time
            }
            
            with open(output_path / "experiment_metadata.json", "w") as f:
                json.dump(experiment_data, f, indent=2, default=str)
            
            logger.info(f"Exported experiment to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting experiment: {str(e)}")
            raise

def main():
    """Example usage of the MLflowManager."""
    # Initialize MLflow manager
    mlflow_manager = MLflowManager()
    
    # Start a run
    run_id = mlflow_manager.start_run("example_run", {"task": "summarization"})
    
    # Log parameters
    mlflow_manager.log_parameters({
        "model_name": "facebook/bart-large-cnn",
        "max_length": 150,
        "temperature": 1.0
    })
    
    # Log metrics
    mlflow_manager.log_metrics({
        "rouge1": 0.85,
        "rouge2": 0.72,
        "rougeL": 0.82,
        "bert_f1": 0.88
    })
    
    # Log text
    mlflow_manager.log_text("This is an example summary.", "summary.txt")
    
    # End run
    mlflow_manager.end_run()
    
    print(f"Example run completed: {run_id}")

if __name__ == "__main__":
    main()
