# Clean Project Structure

## 🎯 AI Video Summarizer with Complete ML Suite

### 📁 Core Application Files
- `ml_main.py` - Main ML application with enhanced video summarizer
- `streamlit_app.py` - Web interface for video processing
- `config.py` - Configuration settings for all ML components
- `requirements.txt` - Python dependencies

### 📁 ML Models Directory (`ml_models/`)
- `transformer_summarizer.py` - BART, T5, Pegasus text summarization
- `advanced_speech_recognition.py` - Whisper and Wav2Vec2 speech recognition
- `video_analysis.py` - Computer vision and scene detection
- `data_preprocessing.py` - Audio, video, and text preprocessing
- `model_evaluation.py` - Model evaluation and validation
- `model_training.py` - Automated training and fine-tuning
- `model_serving.py` - Model serving and inference optimization
- `mlflow_integration.py` - MLflow experiment tracking

### 📁 Utilities (`utils/`)
- `logger.py` - Logging configuration

### 📁 Data Directories
- `data/` - Input data storage
- `output/` - Generated summaries, transcripts, and audio
  - `audio/` - Extracted audio files
  - `summaries/` - Generated summaries
  - `transcripts/` - Speech-to-text transcripts
- `models/` - Pre-trained model storage
- `logs/` - Application logs

### 📁 Configuration Files
- `mlflow.db` - MLflow SQLite database
- `setup_ml_suite.py` - ML suite installation script
- `test_ml_suite.py` - Test suite for ML components

### 📁 Documentation
- `README.md` - Main project documentation
- `ML_SUITE_README.md` - ML suite documentation

## 🚀 Quick Start

1. **Install Dependencies**: `python -m pip install -r requirements.txt`
2. **Setup ML Suite**: `python setup_ml_suite.py`
3. **Run Web App**: `python -m streamlit run streamlit_app.py --server.port 8501`
4. **Run MLflow UI**: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

## 🌐 Access Points
- **Streamlit App**: http://localhost:8501
- **MLflow Dashboard**: http://localhost:5000

## ✨ Features
- Universal video content recognition (13+ categories)
- Fast and Comprehensive processing modes
- YouTube URL support
- Advanced ML models (BART, Whisper, YOLO, etc.)
- Real-time experiment tracking
- Intelligent content generation
