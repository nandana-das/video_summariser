# 🚀 Complete ML Suite for Video Summarization

This project has been upgraded with a comprehensive machine learning suite that transforms it from a basic video summarization tool into a state-of-the-art AI platform. The ML suite includes modern transformer models, computer vision, advanced speech recognition, MLOps capabilities, and much more.

## 🎯 What's New in the ML Suite

### 🔥 Core ML Capabilities
- **Modern Transformer Models**: BART, T5, Pegasus, and other state-of-the-art models
- **Advanced Speech Recognition**: Whisper, Wav2Vec2, and ensemble methods
- **Computer Vision**: Video analysis, scene detection, object detection, and captioning
- **MLOps Integration**: MLflow, Weights & Biases, experiment tracking, and model management
- **Automated Training**: Custom model training, fine-tuning, and hyperparameter optimization
- **Model Serving**: High-performance inference with optimization and API serving

### 📊 Advanced Features
- **Data Preprocessing**: Comprehensive pipelines for audio, video, and text
- **Feature Engineering**: Automated feature extraction and dimensionality reduction
- **Model Evaluation**: ROUGE, BERTScore, BLEU, METEOR, and other metrics
- **Batch Processing**: Efficient processing of multiple files
- **Real-time Inference**: Fast API-based model serving
- **Model Registry**: Centralized model management and versioning

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- FFmpeg (for video processing)

### Quick Install
```bash
# Install the complete ML suite
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Install additional ML models (optional)
python -c "import whisper; whisper.load_model('base')"
```

### Development Install
```bash
# Install with development dependencies
pip install -e ".[dev,ml,serving]"

# Initialize MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 🚀 Quick Start

### Basic Usage
```python
from ml_main import EnhancedVideoSummarizer

# Initialize the enhanced summarizer
summarizer = EnhancedVideoSummarizer(use_mlflow=True)

# Load models
summarizer.load_models()

# Process a video with full ML suite
results = summarizer.process_video_comprehensive(
    "meeting.mp4",
    include_visual_analysis=True
)

print(f"Summary: {results['summary']['abstractive_summary']}")
print(f"Visual Analysis: {results['visual_analysis']['content_summary']}")
```

### Command Line Interface
```bash
# Comprehensive video processing
python ml_main.py video.mp4 --comprehensive --visual-analysis

# Batch processing
python ml_main.py --batch video1.mp4 video2.mp4 video3.mp4

# Start inference server
python ml_main.py --serve

# Train custom model
python ml_main.py --train
```

## 📚 ML Suite Components

### 1. Transformer-Based Summarization
```python
from ml_models import TransformerSummarizer, MultiModelSummarizer

# Single model
summarizer = TransformerSummarizer("facebook/bart-large-cnn")
result = summarizer.generate_abstractive_summary(text)

# Ensemble of multiple models
ensemble = MultiModelSummarizer([
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
    "t5-base"
])
result = ensemble.ensemble_summarize(text)
```

### 2. Advanced Speech Recognition
```python
from ml_models import AdvancedSpeechRecognizer

# Whisper-based recognition
recognizer = AdvancedSpeechRecognizer("whisper-base")
result = recognizer.transcribe_audio("audio.wav")

# Wav2Vec2-based recognition
recognizer = AdvancedSpeechRecognizer("wav2vec2-base")
result = recognizer.transcribe_audio("audio.wav")
```

### 3. Computer Vision & Video Analysis
```python
from ml_models import VideoAnalyzer

analyzer = VideoAnalyzer()
analysis = analyzer.analyze_video_content("video.mp4")

print(f"Scenes detected: {len(analysis['scenes'])}")
print(f"Objects found: {analysis['analysis_metadata']['objects_detected']}")
print(f"Content summary: {analysis['content_summary']}")
```

### 4. MLOps & Experiment Tracking
```python
from ml_models import MLflowManager

# Initialize MLflow
mlflow_manager = MLflowManager()

# Start experiment
run_id = mlflow_manager.start_run("my_experiment")

# Log parameters and metrics
mlflow_manager.log_parameters({"learning_rate": 0.001, "batch_size": 32})
mlflow_manager.log_metrics({"accuracy": 0.95, "loss": 0.05})

# Log model
mlflow_manager.log_model(model, "my_model")

# End run
mlflow_manager.end_run()
```

### 5. Data Preprocessing & Feature Engineering
```python
from ml_models import AudioPreprocessor, VideoPreprocessor, TextPreprocessor, FeatureEngineer

# Audio preprocessing
audio_processor = AudioPreprocessor()
audio_data = audio_processor.preprocess_audio("audio.wav")

# Video preprocessing
video_processor = VideoPreprocessor()
video_data = video_processor.preprocess_video("video.mp4")

# Text preprocessing
text_processor = TextPreprocessor()
text_data = text_processor.preprocess_text("Long text to process...")

# Feature engineering
feature_engineer = FeatureEngineer()
text_features = feature_engineer.create_text_features(texts, "tfidf")
```

### 6. Model Training & Fine-tuning
```python
from ml_models import ModelTrainer

# Initialize trainer
trainer = ModelTrainer("facebook/bart-base")

# Prepare data
train_loader, val_loader = trainer.prepare_data(
    train_texts, train_summaries, val_texts, val_summaries
)

# Fine-tune model
results = trainer.fine_tune(
    train_texts, train_summaries,
    num_epochs=3,
    learning_rate=5e-5
)
```

### 7. Model Evaluation
```python
from ml_models import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate summarization
results = evaluator.evaluate_summarization(
    references, predictions, "my_model"
)

print(f"ROUGE-1: {results['rouge_scores']['rouge1_mean']:.4f}")
print(f"BERT F1: {results['bert_scores']['bert_f1_mean']:.4f}")
```

### 8. Model Serving & Inference
```python
from ml_models import InferenceServer, ModelServingAPI

# Initialize inference server
server = InferenceServer("path/to/model")

# Generate prediction
result = server.predict("Text to summarize")

# Start API server
api = ModelServingAPI("path/to/model")
api.run()  # Starts FastAPI server on localhost:8000
```

## 🔧 Configuration

The ML suite is highly configurable through `config.py`. Key configuration sections:

### Model Settings
```python
# Summarization models
SUMMARIZATION_SETTINGS = {
    "transformer_models": {
        "bart": "facebook/bart-large-cnn",
        "t5": "t5-base",
        "pegasus": "google/pegasus-xsum"
    }
}

# Speech recognition models
TRANSCRIPTION_SETTINGS = {
    "whisper_models": {
        "tiny": "whisper-tiny",
        "base": "whisper-base",
        "large": "whisper-large"
    }
}
```

### Training Settings
```python
TRAINING_SETTINGS = {
    "default_epochs": 3,
    "default_batch_size": 8,
    "default_learning_rate": 5e-5,
    "early_stopping_patience": 3
}
```

### Serving Settings
```python
SERVING_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_workers": 4,
    "optimization": {
        "quantization": True,
        "onnx_conversion": False
    }
}
```

## 📊 Performance & Optimization

### Model Optimization
- **Quantization**: Reduce model size and inference time
- **Pruning**: Remove unnecessary parameters
- **ONNX Conversion**: Cross-platform optimization
- **Mixed Precision**: Faster training and inference

### Inference Optimization
- **Batch Processing**: Process multiple inputs efficiently
- **Async Processing**: Non-blocking inference
- **Caching**: Store frequently used results
- **GPU Acceleration**: Automatic device detection

### Performance Monitoring
- **MLflow Tracking**: Comprehensive experiment tracking
- **Weights & Biases**: Advanced visualization and monitoring
- **Custom Metrics**: ROUGE, BERTScore, BLEU, and more

## 🧪 Testing & Validation

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=ml_models tests/

# Run specific test categories
pytest tests/test_summarization.py
pytest tests/test_speech_recognition.py
```

### Model Validation
```python
# Cross-validation
from ml_models import ModelEvaluator

evaluator = ModelEvaluator()
cv_results = evaluator.cross_validate(model, X, y, cv_folds=5)

# Hyperparameter optimization
from ml_models import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()
best_params = optimizer.optimize(objective_function, n_trials=100)
```

## 🚀 Deployment

### Local Deployment
```bash
# Start inference server
python ml_main.py --serve

# Access API at http://localhost:8000
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text to summarize"}'
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "ml_main.py", "--serve"]
```

### Cloud Deployment
- **AWS**: SageMaker, EC2, Lambda
- **Google Cloud**: AI Platform, Compute Engine
- **Azure**: Machine Learning, Container Instances
- **Kubernetes**: Scalable container orchestration

## 📈 Monitoring & Maintenance

### MLflow UI
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Access at http://localhost:5000
```

### Model Registry
```python
from ml_models import ModelRegistry

registry = ModelRegistry()

# List all models
models = registry.list_models()

# Get model info
info = registry.get_model_info("model_id")

# Set active model
registry.set_active_model("model_id")
```

### Performance Monitoring
- **Real-time Metrics**: Processing time, accuracy, throughput
- **Resource Usage**: CPU, GPU, memory utilization
- **Error Tracking**: Comprehensive error logging and monitoring
- **Alerting**: Automated alerts for performance issues

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd video_summariser

# Install development dependencies
pip install -e ".[dev,ml,serving]"

# Run tests
pytest

# Format code
black ml_models/
flake8 ml_models/
```

### Adding New Models
1. Create model class in `ml_models/`
2. Implement required methods
3. Add to `ml_models/__init__.py`
4. Update configuration in `config.py`
5. Add tests and documentation

## 📚 API Reference

### EnhancedVideoSummarizer
Main class for comprehensive video processing.

#### Methods
- `load_models()`: Load ML models
- `process_video_comprehensive()`: Full video processing pipeline
- `train_custom_model()`: Train custom models
- `evaluate_model_performance()`: Evaluate model performance
- `start_inference_server()`: Start model serving
- `batch_process_videos()`: Process multiple videos

### TransformerSummarizer
Advanced transformer-based text summarization.

#### Methods
- `generate_abstractive_summary()`: Generate abstractive summary
- `extractive_summarize()`: Generate extractive summary
- `evaluate_summary()`: Evaluate summary quality
- `batch_summarize()`: Process multiple texts

### AdvancedSpeechRecognizer
State-of-the-art speech recognition.

#### Methods
- `transcribe_audio()`: Transcribe audio to text
- `batch_transcribe()`: Process multiple audio files
- `evaluate_transcription()`: Evaluate transcription quality

### VideoAnalyzer
Comprehensive video analysis and understanding.

#### Methods
- `analyze_video_content()`: Full video analysis
- `detect_scenes()`: Scene change detection
- `detect_objects()`: Object detection
- `generate_captions()`: Image captioning

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
TRAINING_SETTINGS["default_batch_size"] = 4

# Use gradient accumulation
TRAINING_SETTINGS["gradient_accumulation_steps"] = 4
```

#### Model Loading Errors
```python
# Check device compatibility
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

#### Performance Issues
```python
# Enable optimization
SERVING_SETTINGS["optimization"]["quantization"] = True
SERVING_SETTINGS["optimization"]["onnx_conversion"] = True
```

### Getting Help
- **Documentation**: Check this README and inline docstrings
- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions
- **Email**: Contact the development team

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For transformer models and libraries
- **OpenAI**: For Whisper speech recognition
- **MLflow**: For experiment tracking and model management
- **Weights & Biases**: For advanced ML monitoring
- **spaCy**: For natural language processing
- **PyTorch**: For deep learning framework

---

**🚀 The Complete ML Suite transforms your video summarization project into a production-ready AI platform!**
