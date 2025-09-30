# 🎥 AI Video Summarizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

An advanced AI-powered video summarization tool that extracts audio from videos, generates transcripts using speech recognition, and creates intelligent summaries with action items, keywords, and named entity extraction.

## ✨ Features

### 🎯 Core Functionality
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

## 🚀 Quick Start

### Web Interface
```bash
# Launch the Streamlit web app
streamlit run streamlit_app.py
```

### Command Line Interface
```bash
# Process a video file
python main.py video.mp4

# Process an audio file
python main.py audio.wav -t audio

# Process a transcript
python main.py transcript.txt -t transcript

# Customize summary length
python main.py video.mp4 -s 10

# Specify output name
python main.py video.mp4 -o my_meeting
```

### Python API
```python
from main import VideoSummarizer

# Initialize the summarizer
summarizer = VideoSummarizer()

# Process a video
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
├── main.py                 # Main CLI application
├── streamlit_app.py        # Web interface
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── README.md             # This file
├── audioExtraction.py    # Audio extraction module
├── transcriptCreation.py # Speech recognition module
├── transcriptSummariser.py # Summarization module
├── email_sender.py       # Email functionality
├── utils/                # Utility modules
│   ├── __init__.py
│   └── logger.py
├── data/                 # Input data directory
├── models/               # AI models directory
├── output/               # Output files directory
│   ├── audio/           # Extracted audio files
│   ├── transcripts/     # Generated transcripts
│   └── summaries/       # Generated summaries
└── logs/                # Log files
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
