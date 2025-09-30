#!/usr/bin/env python3
"""
Setup script for the Complete ML Suite for Video Summarization.
This script handles installation, configuration, and initial setup.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"[SUCCESS] Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_gpu_availability():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[SUCCESS] CUDA GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("[WARNING] No CUDA GPU detected. CPU will be used.")
            return False
    except ImportError:
        print("[WARNING] PyTorch not installed yet. GPU check will be performed after installation.")
        return False

def install_requirements():
    """Install Python requirements."""
    print("\n[INFO] Installing Python packages...")
    
    # Install basic requirements
    if not run_command("python -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    # Install additional ML packages
    ml_packages = [
        "spacy",
        "transformers",
        "torch",
        "torchaudio",
        "torchvision",
        "mlflow",
        "wandb",
        "optuna",
        "fastapi",
        "uvicorn"
    ]
    
    for package in ml_packages:
        if not run_command(f"python -m pip install {package}", f"Installing {package}"):
            print(f"[WARNING] Failed to install {package}, continuing...")
    
    return True

def download_models():
    """Download required ML models."""
    print("\n[INFO] Downloading ML models...")
    
    try:
        # Download spaCy model
        run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")
        
        # Download Whisper models (optional)
        print("[INFO] Downloading Whisper models (this may take a while)...")
        run_command("python -c \"import whisper; whisper.load_model('base')\"", "Downloading Whisper base model")
        
        # Download sentence transformer model
        run_command("python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\"", "Downloading sentence transformer model")
        
        return True
    except Exception as e:
        print(f"[WARNING] Model download failed: {e}")
        print("You can download models manually later.")
        return True

def setup_directories():
    """Create necessary directories."""
    print("\n[INFO] Setting up directories...")
    
    directories = [
        "data",
        "models", 
        "output",
        "logs",
        "ml_models",
        "experiments",
        "serving",
        "output/audio",
        "output/transcripts", 
        "output/summaries"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"[SUCCESS] Created directory: {directory}")
    
    return True

def setup_mlflow():
    """Initialize MLflow."""
    print("\n[INFO] Setting up MLflow...")
    
    try:
        # Initialize MLflow database
        run_command("python -c \"import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db')\"", "Initializing MLflow database")
        
        # Create default experiment
        run_command("python -c \"import mlflow; mlflow.create_experiment('video_summarizer')\"", "Creating MLflow experiment")
        
        print("[SUCCESS] MLflow setup completed")
        print("[INFO] Start MLflow UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        return True
    except Exception as e:
        print(f"[WARNING] MLflow setup failed: {e}")
        return True

def create_config_file():
    """Create configuration file if it doesn't exist."""
    print("\n[INFO] Setting up configuration...")
    
    config_file = Path("config.py")
    if config_file.exists():
        print("[SUCCESS] Configuration file already exists")
        return True
    
    # Create basic config
    config_content = '''"""
Configuration settings for the Video Summarizer project with complete ML suite.
"""
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

# Basic settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "chunk_size": 500,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".wav", ".mp3", ".m4a", ".flac"]
}

SUMMARIZATION_SETTINGS = {
    "max_sentences": 5,
    "min_sentence_length": 10,
    "similarity_threshold": 0.3
}

MLFLOW_SETTINGS = {
    "tracking_uri": "sqlite:///mlflow.db",
    "experiment_name": "video_summarizer"
}
'''
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("[SUCCESS] Configuration file created")
    return True

def run_tests():
    """Run basic tests to verify installation."""
    print("\n[INFO] Running tests...")
    
    test_commands = [
        ("python -c \"import torch; print('PyTorch:', torch.__version__)\"", "Testing PyTorch"),
        ("python -c \"import transformers; print('Transformers:', transformers.__version__)\"", "Testing Transformers"),
        ("python -c \"import mlflow; print('MLflow:', mlflow.__version__)\"", "Testing MLflow"),
        ("python -c \"from ml_models import TransformerSummarizer; print('ML Models: OK')\"", "Testing ML Models")
    ]
    
    for command, description in test_commands:
        if not run_command(command, description):
            print(f"[WARNING] {description} failed, but installation may still work")
    
    return True

def print_next_steps():
    """Print next steps for the user."""
    print("\n[SUCCESS] ML Suite setup completed!")
    print("\n[INFO] Next Steps:")
    print("1. Test the installation:")
    print("   python ml_main.py --help")
    print("\n2. Process a video:")
    print("   python ml_main.py video.mp4 --comprehensive")
    print("\n3. Start MLflow UI:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("\n4. Start inference server:")
    print("   python ml_main.py --serve")
    print("\n5. Read the documentation:")
    print("   cat ML_SUITE_README.md")
    print("\n[INFO] For more information, see ML_SUITE_README.md")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Complete ML Suite for Video Summarization")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading ML models")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--gpu", action="store_true", help="Force GPU installation")
    
    args = parser.parse_args()
    
    print("Setting up Complete ML Suite for Video Summarization")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Setup directories
    if not setup_directories():
        print("[ERROR] Failed to setup directories")
        sys.exit(1)
    
    # Create config file
    if not create_config_file():
        print("[ERROR] Failed to create configuration file")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("[ERROR] Failed to install requirements")
        sys.exit(1)
    
    # Download models (optional)
    if not args.skip_models:
        download_models()
    
    # Setup MLflow
    setup_mlflow()
    
    # Run tests (optional)
    if not args.skip_tests:
        run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
