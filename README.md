# 🎥 AI Video Summarizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

A powerful AI-powered video summarization tool that transforms any video into actionable insights using advanced machine learning and natural language processing. Built with Flask and state-of-the-art ML models including Whisper for speech recognition and BART for intelligent summarization. Upload your video files or paste URLs to get instant summaries with keywords, action items, and detailed analysis.

## ✨ Features

### 🎯 Core Functionality
- **📁 Video Upload**: Support for multiple video formats (MP4, AVI, MOV, MKV, WMV, FLV)
- **🔗 URL Processing**: Support for 1000+ video platforms (YouTube, Vimeo, Instagram, TikTok, Facebook, Twitter, Twitch, etc.)
- **🎤 Real Speech Recognition**: Advanced audio transcription using OpenAI's Whisper AI
- **🤖 ML-Powered Summarization**: Intelligent text summarization using BART transformer model
- **🎯 Smart Action Item Extraction**: NLP-powered identification of tasks and action items
- **🔑 Advanced Keyword Analysis**: Extract key topics using NLTK and advanced NLP techniques
- **📊 Comprehensive Analytics**: Visual metrics, ROUGE scores, and compression ratios
- **💾 Export Results**: Download summaries, transcripts, and analysis data

### 🚀 Advanced Features
- **🌐 Modern Web Interface**: Beautiful, responsive Flask-based web application
- **📱 Mobile Friendly**: Works perfectly on all devices
- **⚡ Real-time Processing**: Live video download, audio extraction, and ML processing
- **🎨 Professional UI**: Modern design with smooth animations and drag-and-drop
- **🔄 Live Progress Tracking**: Real-time status updates during processing
- **🔧 RESTful API**: Complete API for integration with other applications
- **🛡️ Robust Error Handling**: Graceful fallbacks and comprehensive error management
- **📈 Performance Metrics**: ROUGE scores, compression ratios, and quality metrics

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:5000**

## 📁 Project Structure

```
video_summariser/
├── app.py                 # Flask application with ML integration
├── video_processor.py     # Core ML/NLP processing engine
├── run.py                 # Run script
├── requirements.txt       # ML/NLP dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Beautiful CSS styling
│   └── js/
│       └── app.js        # JavaScript functionality
└── README.md             # This file
```

## 🧠 ML/NLP Technologies

### Core Models
- **Whisper AI**: OpenAI's state-of-the-art speech recognition model
- **BART**: Facebook's transformer model for text summarization
- **NLTK**: Natural Language Toolkit for text processing
- **Transformers**: Hugging Face transformers library

### Processing Pipeline
1. **Video Download**: yt-dlp for multi-platform video downloading
2. **Audio Extraction**: MoviePy for high-quality audio extraction
3. **Speech Recognition**: Whisper for accurate transcription
4. **Text Summarization**: BART for intelligent summarization
5. **Keyword Extraction**: NLTK for advanced keyword analysis
6. **Action Item Detection**: Pattern matching and NLP techniques

## 🎨 UI Features

### Upload Interface
- **Drag & Drop**: Simply drag video files onto the upload area
- **File Browser**: Click to browse and select files
- **File Validation**: Automatic file type and size validation
- **Progress Tracking**: Real-time upload and processing status

### URL Processing
- **Multi-Platform Support**: Works with 1000+ video platforms
- **Platform Cards**: Visual representation of supported platforms
- **URL Validation**: Automatic URL format validation

### Results Display
- **Tabbed Interface**: Organized results in easy-to-navigate tabs
- **Summary View**: Clean, readable AI-generated summaries
- **Action Items**: Highlighted tasks and action items
- **Keywords**: Tag-based keyword display
- **Transcript**: Full scrollable transcript view
- **Quick Stats**: Visual metrics and analytics

## 🔧 Configuration

### Processing Settings
- **Summary Length**: Adjustable from 3 to 20 sentences (configurable tokens)
- **Processing Mode**: Fast or Comprehensive analysis
- **File Size Limit**: 500MB maximum file size
- **ML Model Selection**: Automatic model loading with fallback options
- **Audio Quality**: High-quality audio extraction and processing

### Supported Platforms
- YouTube, Vimeo, Instagram, TikTok
- Facebook, Twitter, Twitch, Dailymotion
- And 1000+ more platforms via yt-dlp

## 🚀 Deployment

### Local Development
```bash
python run.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t video-summarizer .
docker run -p 5000:5000 video-summarizer
```

### Cloud Deployment
- **Heroku**: Add `Procfile` and deploy
- **Railway**: Connect GitHub repository
- **DigitalOcean**: Use App Platform
- **AWS**: Deploy to Elastic Beanstalk

## 🎯 API Endpoints

### POST /api/process-video
Process uploaded video file
- **Input**: Multipart form data with video file
- **Output**: JSON with summary data

### POST /api/process-url
Process video from URL
- **Input**: JSON with video URL
- **Output**: JSON with summary data

### GET /api/platforms
Get supported video platforms
- **Output**: JSON with platform information

## 🎨 Customization

### Styling
Edit `static/css/style.css` to customize:
- Colors and themes
- Layout and spacing
- Animations and transitions
- Responsive breakpoints

### Functionality
Edit `static/js/app.js` to customize:
- UI interactions
- API calls
- Data processing
- Error handling

## 🔧 Troubleshooting

### Common Issues
1. **File Upload Fails**: Check file size and format
2. **URL Processing Fails**: Verify URL format and platform support
3. **ML Model Loading**: Ensure all dependencies are installed correctly
4. **Audio Processing**: Check if ffmpeg is installed for audio extraction
5. **Memory Issues**: Large videos may require more RAM for processing
6. **UI Issues**: Clear browser cache and reload

### Debug Mode
```bash
export FLASK_DEBUG=1
python run.py
```

## 📱 Mobile Support

The application is fully responsive and works on:
- 📱 Mobile phones (iOS, Android)
- 📱 Tablets (iPad, Android tablets)
- 💻 Desktop computers
- 🖥️ Large screens

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

## 🎯 What Makes This Special

### Real AI Processing
- **Actual Speech Recognition**: Uses OpenAI's Whisper to transcribe real audio
- **Intelligent Summarization**: BART transformer model generates meaningful summaries
- **Advanced NLP**: NLTK and transformers for sophisticated text analysis
- **Multi-Platform Support**: Downloads from 1000+ video platforms
- **Production Ready**: Robust error handling and graceful fallbacks

### Technical Excellence
- **State-of-the-Art Models**: Latest ML/NLP models for best results
- **Scalable Architecture**: Flask-based backend with modular design
- **Real-time Processing**: Live progress tracking and status updates
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Memory Efficient**: Optimized for processing large video files

---

**Built with ❤️ using Flask, Python, Whisper AI, BART, and modern ML technologies.**