"""
Complete ML Suite for Video Summarization

This package provides a comprehensive machine learning suite for video summarization,
including modern transformer models, computer vision, speech recognition, and MLOps capabilities.
"""

from .transformer_summarizer import TransformerSummarizer, MultiModelSummarizer
from .advanced_speech_recognition import AdvancedSpeechRecognizer, MultiModelSpeechRecognizer
from .video_analysis import VideoAnalyzer
from .mlflow_integration import MLflowManager
from .data_preprocessing import (
    AudioPreprocessor, VideoPreprocessor, TextPreprocessor, FeatureEngineer
)
from .model_evaluation import ModelEvaluator, HyperparameterOptimizer
from .model_training import ModelTrainer, AutoMLTrainer, VideoSummarizationDataset
from .model_serving import (
    ModelOptimizer, ONNXInferenceEngine, InferenceServer, 
    ModelServingAPI, ModelRegistry
)

__version__ = "2.0.0"
__author__ = "Video Summarizer Team"

__all__ = [
    # Transformer-based summarization
    "TransformerSummarizer",
    "MultiModelSummarizer",
    
    # Advanced speech recognition
    "AdvancedSpeechRecognizer", 
    "MultiModelSpeechRecognizer",
    
    # Computer vision and video analysis
    "VideoAnalyzer",
    
    # MLOps and experiment tracking
    "MLflowManager",
    
    # Data preprocessing and feature engineering
    "AudioPreprocessor",
    "VideoPreprocessor", 
    "TextPreprocessor",
    "FeatureEngineer",
    
    # Model evaluation and optimization
    "ModelEvaluator",
    "HyperparameterOptimizer",
    
    # Model training and fine-tuning
    "ModelTrainer",
    "AutoMLTrainer",
    "VideoSummarizationDataset",
    
    # Model serving and inference
    "ModelOptimizer",
    "ONNXInferenceEngine",
    "InferenceServer",
    "ModelServingAPI", 
    "ModelRegistry"
]
