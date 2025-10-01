# 🎥 AI Video Summarizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Multi-Platform](https://img.shields.io/badge/Platforms-1000+-green.svg)](https://github.com/yt-dlp/yt-dlp)

An advanced AI-powered video summarization tool that supports **1000+ video platforms** including YouTube, Vimeo, Instagram, TikTok, Facebook, Twitter, Twitch, and many more. Extract audio from videos, generate transcripts using speech recognition, and create intelligent summaries with action items, keywords, and named entity extraction.

## ✨ Features

### 🎯 Core Functionality
- **🌐 Multi-Platform Support**: Process videos from 1000+ platforms including YouTube, Vimeo, Instagram, TikTok, Facebook, Twitter, Twitch, Dailymotion, Bilibili, Rumble, Odysee, LBRY, and many more!
- **Video Processing**: Extract audio from multiple video formats (MP4, AVI, MOV, MKV, WMV, FLV)
- **Speech Recognition**: High-accuracy transcription using Vosk speech recognition
- **Intelligent Summarization**: Advanced NLP-based text summarization with multiple algorithms
- **Action Item Extraction**: Automatically identify and extract tasks and action items
- **Keyword Analysis**: Extract key topics and important terms
- **Named Entity Recognition**: Identify people, organizations, and locations mentioned

### 🚀 Advanced Features
- **Web Interface**: Modern Streamlit-based web application
- **CLI Interface**: Command-line tool for batch processing
- **Email Integration**: Send summaries to meeting participants
- **Multiple Output Formats**: Text, JSON, and HTML summaries
- **Configurable Settings**: Customizable summarization parameters
- **Progress Tracking**: Real-time processing status and logging

### 📊 Analytics & Insights
- **Compression Metrics**: Track summarization efficiency
- **Content Analysis**: Detailed statistics about processed content
- **Visualization**: Interactive charts and graphs for data insights
- **Export Options**: Download results in multiple formats

## 🌐 Supported Platforms

This tool supports **1000+ video platforms** through yt-dlp integration, including:

### Popular Platforms
- **📺 YouTube** - The world's largest video platform
- **🎬 Vimeo** - High-quality video hosting and sharing
- **📷 Instagram** - Photo and video sharing platform
- **🎵 TikTok** - Short-form video platform
- **👥 Facebook** - Social media video content
- **🐦 Twitter/X** - Social media video posts
- **🎮 Twitch** - Live streaming and gaming content

### Additional Platforms
- **🎥 Dailymotion** - Video sharing platform
- **🇨🇳 Bilibili** - Chinese video platform
- **⚡ Rumble** - Video platform and creator network
- **🔍 Odysee** - Decentralized video platform
- **🔗 LBRY** - Decentralized content platform
- **And 1000+ more!** - Full list available in yt-dlp documentation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for video processing)
- Git

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/video-summarizer.git
cd video-summarizer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Development Install
```bash
# Install with development dependencies
pip install -e ".[dev,web,nlp]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🌐 Live Demo

**Try the app online**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourusername-video-summariser-app-xxxxx.streamlit.app/)

*Replace `yourusername` with your GitHub username after deployment*

## 🚀 Quick Start

### Web Interface
```bash
# Launch the Streamlit web app
streamlit run streamlit_app.py
```

### Command Line Interface

#### Process Videos from URLs
```bash
# YouTube video
python ml_main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Vimeo video
python ml_main.py "https://vimeo.com/123456789" --comprehensive

# Instagram video
python ml_main.py "https://www.instagram.com/p/ABC123/" -s 10

# TikTok video
python ml_main.py "https://www.tiktok.com/@user/video/1234567890" --comprehensive

# Facebook video
python ml_main.py "https://www.facebook.com/watch/?v=123456789"

# Twitter video
python ml_main.py "https://twitter.com/user/status/1234567890"

# Twitch video
python ml_main.py "https://www.twitch.tv/videos/123456789" --comprehensive
```

#### Process Local Files
```bash
# Process a video file
python ml_main.py video.mp4

# Process an audio file
python ml_main.py audio.wav -t audio

# Process a transcript
python ml_main.py transcript.txt -t transcript

# Customize summary length
python ml_main.py video.mp4 -s 10

# Specify output name
python ml_main.py video.mp4 -o my_meeting
```

### Python API

#### Multi-Platform Video Processing
```python
from ml_main import EnhancedVideoSummarizer

# Initialize the enhanced summarizer
summarizer = EnhancedVideoSummarizer()
summarizer.load_models()

# Process video from any supported platform
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
result = summarizer.process_video(url, max_sentences=5, comprehensive=True)

if result["success"]:
    print(f"Summary: {result['summary_data']['summary']}")
    print(f"Platform: {result['platform_info']['platform']}")
    print(f"Video Title: {result['platform_info']['video_title']}")
    print(f"Action Items: {result['summary_data']['action_items']}")
else:
    print(f"Error: {result['error']}")
```

#### Local File Processing
```python
from main import VideoSummarizer

# Initialize the basic summarizer
summarizer = VideoSummarizer()

# Process a local video file
results = summarizer.process_video("meeting.mp4")

# Access results
print(f"Summary: {results['summary_data']['summary']}")
print(f"Action items: {results['summary_data']['action_items']}")
```

## 📖 Usage Examples

### Basic Video Processing
```python
from main import VideoSummarizer

summarizer = VideoSummarizer()
results = summarizer.process_video("meeting.mp4", max_sentences=5)

if results["success"]:
    print("Summary:", results["summary_data"]["summary"])
    print("Action items:", results["summary_data"]["action_items"])
```

### Email Integration
```python
from email_sender import EmailSender

# Configure email settings
sender = EmailSender(
    sender_email="your_email@gmail.com",
    sender_password="your_app_password"
)

# Send summary email
success = sender.send_summary_email(
    recipients=["participant1@example.com", "participant2@example.com"],
    summary_data=results["summary_data"],
    meeting_title="Weekly Team Meeting"
)
```

### Custom Summarization
```python
from transcriptSummariser import AdvancedSummarizer

summarizer = AdvancedSummarizer()

# Generate summary with custom parameters
summary_data = summarizer.generate_summary(
    text=transcript_text,
    max_sentences=8
)

# Extract specific information
keywords = summary_data["keywords"]
action_items = summary_data["action_items"]
entities = summary_data["named_entities"]
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Email Configuration
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Model Configuration
VOSK_MODEL_PATH=models/vosk-model-small-en-us-0.15
SPACY_MODEL=en_core_web_sm
```

### Configuration File
Modify `config.py` to customize default settings:

```python
# Summarization settings
SUMMARIZATION_SETTINGS = {
    "max_sentences": 5,
    "min_sentence_length": 10,
    "similarity_threshold": 0.3
}

# Audio settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "chunk_size": 500,
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
}
```

## 📁 Project Structure

```
video-summarizer/
├── ml_main.py                    # Enhanced main application with ML suite
├── streamlit_app.py              # Web interface
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── setup_ml_suite.py             # Installation script
├── test_ml_suite.py              # Test suite
├── ML_SUITE_README.md            # Complete ML suite documentation
├── README.md                     # This file
├── audioExtraction.py            # Audio extraction module
├── transcriptCreation.py         # Speech recognition module
├── transcriptSummariser_fixed.py # Summarization module
├── email_sender.py               # Email functionality
├── ml_models/                    # Complete ML Suite
│   ├── transformer_summarizer.py
│   ├── advanced_speech_recognition.py
│   ├── video_analysis.py
│   ├── mlflow_integration.py
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── model_serving.py
├── utils/                        # Utility modules
│   ├── __init__.py
│   └── logger.py
├── data/                         # Input data directory
├── models/                       # AI models directory
├── output/                       # Output files directory
│   ├── audio/                   # Extracted audio files
│   ├── transcripts/             # Generated transcripts
│   └── summaries/               # Generated summaries
├── logs/                        # Log files
└── examples/                    # Usage examples
    ├── __init__.py
    └── basic_usage.py
```

## 🔧 API Reference

### VideoSummarizer Class
Main class for processing videos, audio, and transcripts.

#### Methods
- `process_video(video_path, output_name=None, max_sentences=None)`: Process a video file
- `process_audio(audio_path, output_name=None, max_sentences=None)`: Process an audio file
- `process_transcript(transcript_path, output_name=None, max_sentences=None)`: Process a transcript

### AdvancedSummarizer Class
Advanced text summarization with multiple NLP techniques.

#### Methods
- `generate_summary(text, max_sentences=None)`: Generate comprehensive summary
- `extract_keywords(text, top_k=20)`: Extract keywords using TF-IDF
- `extract_action_items(text)`: Extract action items and tasks
- `extract_named_entities(text)`: Extract named entities

### EmailSender Class
Send summary emails to meeting participants.

#### Methods
- `send_summary_email(recipients, summary_data, meeting_title, attachments=None)`: Send single email
- `send_bulk_emails(email_list)`: Send multiple emails

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_summarizer.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🚀 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
3. **Click "New app"**
4. **Select your repository**: `yourusername/video_summariser`
5. **Main file path**: `streamlit_app.py`
6. **Click "Deploy!"**

Your app will be live at: `https://yourusername-video-summariser-app-xxxxx.streamlit.app/`

### Deploy to Heroku

1. **Create a `Procfile`**:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create a `runtime.txt`**:
   ```
   python-3.9.18
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) for speech recognition
- [spaCy](https://spacy.io/) for natural language processing
- [Streamlit](https://streamlit.io/) for the web interface
- [MoviePy](https://zulko.github.io/moviepy/) for video processing

## 📚 References

    1. Shah, M., & Patel, D. (2023). Efficient meeting insights: NLP-Enhanced summarization of voice and Text. Retrieved from https://www.researchgate.net/profile/Manan-Shah-44/publication/376621969_Efficient_meeting_insights_NLP_-Enhanced_summarization_of_voice_and_Text/links/658188813c472d2e8e707105/Efficient-meeting-insights-NLP-Enhanced-summarization-of-voice-and-Text.pdf

    2. O'Donovan, P. (2023). An investigation into the use of NLP to extract meeting action items for project management. Retrieved from https://norma.ncirl.ie/id/eprint/6262

    3. Chen, H., & Rangarajan, S. (2023). A novel approach to summarizing business meeting transcripts. Retrieved from https://dl.acm.org/doi/abs/10.1145/3474085.3478556

    4. Kumar, A., & Srinivas, T. (2023). Leveraging AI for effective meeting summarization. Retrieved from https://ieeexplore.ieee.org/abstract/document/9777155/

## 📞 Support

For support, email support@videosummarizer.ai or create an issue on GitHub.

---

**Made with ❤️ by the AI Video Summarizer Team**
