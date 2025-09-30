"""
Configuration settings for the Video Summarizer project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "chunk_size": 500,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
}

# Transcription settings
TRANSCRIPTION_SETTINGS = {
    "model_name": "vosk-model-small-en-us-0.15",
    "model_url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "confidence_threshold": 0.5
}

# Summarization settings
SUMMARIZATION_SETTINGS = {
    "max_sentences": 5,
    "min_sentence_length": 10,
    "similarity_threshold": 0.3
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
    "file": LOGS_DIR / "video_summarizer.log"
}
