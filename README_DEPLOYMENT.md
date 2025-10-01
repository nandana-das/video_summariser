# 🚀 Deployment Guide: AI Video Summarizer

## 🌐 Deploy to Streamlit Cloud (Recommended)

### Quick Setup
1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add beautiful multi-platform video summarizer"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Repository: `yourusername/video_summariser`
   - Main file: `streamlit_app.py`
   - Click "Deploy!"

3. **Your app will be live at**: `https://yourusername-video-summariser-app-xxxxx.streamlit.app/`

## 🔧 Configuration Files

- `.streamlit/config.toml` - Streamlit configuration
- `packages.txt` - System dependencies (ffmpeg)
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

## 📋 Features

- ✅ **Multi-Platform Support** - 1000+ video platforms
- ✅ **Beautiful UI** - Modern, responsive design
- ✅ **AI-Powered** - Advanced ML models
- ✅ **Real-time Processing** - Fast video analysis
- ✅ **Download Results** - Multiple export formats

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## 📱 Access Your App

Once deployed, your app will be available at:
- **Streamlit Cloud**: `https://yourusername-video-summariser-app-xxxxx.streamlit.app/`
- **Local**: `http://localhost:8501`
