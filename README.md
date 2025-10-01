# 🎥 AI Video Summarizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Multi-Platform](https://img.shields.io/badge/Platforms-1000+-green.svg)](https://github.com/yt-dlp/yt-dlp)

A beautiful and intelligent video summarization tool that transforms any video into actionable insights. Built with Flask for maximum flexibility and performance. Upload your video files or paste URLs to get instant summaries with keywords, action items, and detailed analysis.

## ✨ Features

### 🎯 Core Functionality
- **📁 Video Upload**: Support for multiple video formats (MP4, AVI, MOV, MKV, WMV, FLV)
- **🔗 URL Processing**: Support for 1000+ video platforms (YouTube, Vimeo, Instagram, TikTok, Facebook, Twitter, Twitch, etc.)
- **🤖 Intelligent Summarization**: Advanced AI-powered text summarization using transformer models
- **🎯 Action Item Extraction**: Automatically identify and extract tasks and action items
- **🔑 Keyword Analysis**: Extract key topics and important terms from content
- **📊 Beautiful Analytics**: Visual metrics and insights about your content
- **💾 Download Results**: Export summaries, transcripts, and analysis data

### 🚀 Advanced Features
- **🌐 Modern Web Interface**: Beautiful, responsive Flask-based web application
- **📱 Mobile Friendly**: Works perfectly on all devices
- **⚡ Fast Processing**: Quick analysis and summarization
- **🎨 Professional UI**: Modern design with smooth animations and drag-and-drop
- **🔄 Real-time Updates**: Live progress tracking and status updates
- **🔧 API Endpoints**: RESTful API for integration with other applications

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
├── app.py                 # Flask application
├── run.py                 # Run script
├── requirements.txt       # Dependencies
├── ml_main.py            # Core ML functionality
├── config.py             # Configuration settings
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Beautiful CSS styling
│   └── js/
│       └── app.js        # JavaScript functionality
├── ml_models/            # ML model implementations
├── utils/                # Utility modules
└── README.md             # This file
```

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
- **Summary Length**: Adjustable from 3 to 20 sentences
- **Processing Mode**: Fast or Comprehensive analysis
- **File Size Limit**: 500MB maximum file size

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
3. **Processing Errors**: Check ML model dependencies
4. **UI Issues**: Clear browser cache and reload

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

**Built with ❤️ using Flask, Python, and modern web technologies.**