"""
Configuration settings for the Video Summarizer project with complete ML suite.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SERVING_DIR = PROJECT_ROOT / "serving"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR, ML_MODELS_DIR, EXPERIMENTS_DIR, SERVING_DIR]:
    directory.mkdir(exist_ok=True)

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "chunk_size": 500,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".wav", ".mp3", ".m4a", ".flac"],
    "max_duration": 300.0,  # seconds
    "preprocessing": {
        "normalize": True,
        "trim_silence": True,
        "remove_noise": False
    }
}

# Video settings
VIDEO_SETTINGS = {
    "target_size": (224, 224),
    "max_frames": 100,
    "fps": 1.0,
    "preprocessing": {
        "resize": True,
        "normalize": True,
        "extract_scenes": True
    }
}

# Transcription settings
TRANSCRIPTION_SETTINGS = {
    "model_name": "vosk-model-small-en-us-0.15",
    "model_url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "confidence_threshold": 0.5,
    "whisper_models": {
        "tiny": "whisper-tiny",
        "base": "whisper-base",
        "small": "whisper-small",
        "medium": "whisper-medium",
        "large": "whisper-large"
    },
    "wav2vec2_models": {
        "base": "facebook/wav2vec2-base-960h",
        "large": "facebook/wav2vec2-large-960h-lv60-self"
    }
}

# Summarization settings
SUMMARIZATION_SETTINGS = {
    "max_sentences": 5,
    "min_sentence_length": 10,
    "similarity_threshold": 0.3,
    "transformer_models": {
        "bart": "facebook/bart-large-cnn",
        "t5": "t5-base",
        "pegasus": "google/pegasus-xsum",
        "distilbart": "sshleifer/distilbart-cnn-12-6"
    },
    "max_length": 1024,
    "min_length": 30,
    "num_beams": 4,
    "temperature": 1.0,
    "do_sample": False
}

# Computer Vision settings
CV_SETTINGS = {
    "object_detection": {
        "model": "yolov8n.pt",
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45
    },
    "scene_detection": {
        "threshold": 0.3,
        "min_scene_duration": 2.0
    },
    "caption_generation": {
        "model": "Salesforce/blip-image-captioning-base",
        "max_length": 50
    }
}

# ML Experiment Tracking
MLFLOW_SETTINGS = {
    "tracking_uri": "sqlite:///mlflow.db",
    "experiment_name": "video_summarizer",
    "artifact_location": str(EXPERIMENTS_DIR),
    "registered_model_prefix": "video_summarizer"
}

# Weights & Biases settings
WANDB_SETTINGS = {
    "project": "video-summarization",
    "entity": None,  # Set your W&B entity
    "tags": ["video", "summarization", "ml"]
}

# Model Training settings
TRAINING_SETTINGS = {
    "default_epochs": 3,
    "default_batch_size": 8,
    "default_learning_rate": 5e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100,
    "early_stopping_patience": 3
}

# Model Evaluation settings
EVALUATION_SETTINGS = {
    "metrics": ["rouge1", "rouge2", "rougeL", "bert_f1", "bleu1", "bleu2", "bleu3", "bleu4", "meteor"],
    "cross_validation_folds": 5,
    "test_size": 0.2,
    "random_state": 42
}

# Model Serving settings
SERVING_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_workers": 4,
    "queue_size": 100,
    "timeout": 30,
    "optimization": {
        "quantization": True,
        "pruning": False,
        "onnx_conversion": False
    }
}

# Data Preprocessing settings
PREPROCESSING_SETTINGS = {
    "text": {
        "language": "english",
        "remove_stopwords": True,
        "stemming": True,
        "lemmatization": True,
        "min_word_length": 2
    },
    "audio": {
        "target_sr": 16000,
        "normalize": True,
        "trim_silence": True,
        "extract_features": True
    },
    "video": {
        "target_size": (224, 224),
        "max_frames": 100,
        "extract_scenes": True,
        "extract_objects": True
    }
}

# Feature Engineering settings
FEATURE_ENGINEERING_SETTINGS = {
    "text_features": {
        "tfidf_max_features": 1000,
        "embedding_model": "all-MiniLM-L6-v2",
        "ngram_range": (1, 2)
    },
    "dimensionality_reduction": {
        "pca_components": 50,
        "svd_components": 50
    },
    "clustering": {
        "n_clusters": 5,
        "algorithm": "kmeans"
    }
}

# Email settings (to be configured by user)
EMAIL_SETTINGS = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "",
    "sender_password": "",
    "use_tls": True
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "video_summarizer.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# Device settings
DEVICE_SETTINGS = {
    "auto_detect": True,
    "preferred_device": "auto",  # auto, cpu, cuda, mps
    "mixed_precision": True,
    "memory_efficient": True
}

# Performance settings
PERFORMANCE_SETTINGS = {
    "batch_processing": True,
    "parallel_processing": True,
    "cache_results": True,
    "max_cache_size": 1000,
    "async_processing": True
}

# Security settings
SECURITY_SETTINGS = {
    "api_key_required": False,
    "rate_limiting": True,
    "max_requests_per_minute": 60,
    "cors_origins": ["*"],
    "allowed_hosts": ["*"]
}
