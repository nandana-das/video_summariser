"""
Streamlit web interface for the Video Summarizer project.
"""
import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import subprocess
import sys
from ml_main import EnhancedVideoSummarizer
from utils.logger import setup_logger

# Try to import yt-dlp for video processing
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# Import video source manager
from utils.video_source_manager import VideoSourceManager

# Configure page
st.set_page_config(
    page_title="AI Video Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = setup_logger(__name__)

def is_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([^&\n?#]+)'
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)

def download_video_from_url(url: str, output_dir: str = "temp") -> Optional[str]:
    """Download video from any supported platform and return the file path."""
    if not YT_DLP_AVAILABLE:
        st.error("Video processing requires yt-dlp. Please install it: pip install yt-dlp")
        return None
    
    try:
        # Initialize video source manager
        video_manager = VideoSourceManager(output_dir)
        
        # Validate URL
        is_valid, message = video_manager.validate_url(url)
        if not is_valid:
            st.error(f"Invalid URL: {message}")
            return None
        
        # Detect platform
        platform = video_manager.detect_platform(url)
        st.info(f"🎥 Detected platform: {video_manager.get_platform_icon(platform)} {platform.title()}")
        
        # Get video info
        video_info = video_manager.get_platform_info(url)
        if 'error' in video_info:
            st.error(f"Error getting video info: {video_info['error']}")
            return None
        
        # Show video info
        with st.expander("📋 Video Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Title:** {video_info['title']}")
                st.write(f"**Duration:** {video_info['duration']} seconds")
                st.write(f"**Uploader:** {video_info['uploader']}")
            with col2:
                st.write(f"**Platform:** {video_info['platform']}")
                st.write(f"**Views:** {video_info['view_count']:,}")
                if video_info['thumbnail']:
                    st.image(video_info['thumbnail'], width=200)
        
        # Download the video
        st.info(f"📥 Downloading video from {platform.title()}...")
        video_path = video_manager.download_video(url)
        
        if video_path:
            st.success(f"✅ Video downloaded successfully!")
            return video_path
        else:
            st.error("❌ Failed to download video")
            return None
        
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return None

# Custom CSS for modern, professional design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Header Styles */
    .main-header {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        text-align: center;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }
    
    .description {
        font-size: 1.1rem;
        text-align: center;
        color: #718096;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    /* Platform Showcase */
    .platform-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .platform-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .platform-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .platform-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .platform-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .platform-desc {
        font-size: 0.9rem;
        color: #718096;
        line-height: 1.4;
    }
    
    /* Input Section */
    .input-section {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #e2e8f0;
    }
    
    .input-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Summary Box */
    .summary-box {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        color: #2d3748 !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .action-item {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
        transition: transform 0.2s ease;
        color: #1a202c !important;
        font-weight: 500;
    }
    
    .action-item:hover {
        transform: translateX(5px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px dashed #cbd5e0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
    }
    
    /* Text Input */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Radio Buttons */
    .stRadio > div {
        gap: 1rem;
    }
    
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] {
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .platform-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
        
        .main-container {
            margin: 0.5rem;
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def create_platform_showcase():
    """Create an impressive platform showcase section."""
    video_manager = VideoSourceManager()
    platforms = video_manager.get_supported_platforms()
    
    st.markdown("### 🌐 Supported Platforms")
    st.markdown("Process videos from **1000+ platforms** with a single click!")
    
    # Create platform grid
    cols = st.columns(4)
    for i, platform in enumerate(platforms[:8]):  # Show first 8 platforms
        with cols[i % 4]:
            icon = video_manager.get_platform_icon(platform)
            st.markdown(f"""
            <div class="platform-card">
                <div class="platform-icon">{icon}</div>
                <div class="platform-name">{platform.title()}</div>
                <div class="platform-desc">{video_manager.get_platform_description(platform)[:50]}...</div>
            </div>
            """, unsafe_allow_html=True)
    
    if len(platforms) > 8:
        st.markdown(f"<p style='text-align: center; color: #718096; margin-top: 1rem;'>... and {len(platforms) - 8} more platforms!</p>", unsafe_allow_html=True)

def create_hero_section():
    """Create an impressive hero section."""
    st.markdown('<h1 class="main-header">🎥 AI Video Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Multi-Platform Video Analysis & Summarization</h2>', unsafe_allow_html=True)
    st.markdown('<p class="description">Transform any video into intelligent summaries with action items, keywords, and insights. Supporting 1000+ platforms including YouTube, Vimeo, Instagram, TikTok, Facebook, Twitter, Twitch, and many more!</p>', unsafe_allow_html=True)
    
    # Add some impressive stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1000+</div>
            <div class="metric-label">Platforms</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">AI</div>
            <div class="metric-label">Powered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Fast</div>
            <div class="metric-label">Processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Free</div>
            <div class="metric-label">To Use</div>
        </div>
        """, unsafe_allow_html=True)

def create_input_section():
    """Create the main input section."""
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="input-header">🚀 Get Started</h3>', unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose your input method:",
        ["📁 Upload File", "🌐 Video URL"],
        horizontal=True
    )
    
    uploaded_file = None
    video_url = None
    
    if input_method == "📁 Upload File":
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video, audio, or transcript file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'wav', 'mp3', 'm4a', 'flac', 'txt'],
            help="Upload a file to process"
        )
    else:
        # Video URL input
        video_url = st.text_input(
            "Enter Video URL:",
            placeholder="https://www.youtube.com/watch?v=... or any supported platform",
            help="Paste a video URL from any supported platform"
        )
        
        # Show supported platforms and validate URL
        if video_url:
            video_manager = VideoSourceManager()
            is_valid, message = video_manager.validate_url(video_url)
            
            if is_valid:
                platform = video_manager.detect_platform(video_url)
                st.success(f"✅ {message}")
                st.info(f"🎥 Platform: {video_manager.get_platform_icon(platform)} {platform.title()}")
            else:
                st.error(f"❌ {message}")
                video_url = None
                
                # Show supported platforms
                with st.expander("📋 Supported Platforms", expanded=True):
                    create_platform_showcase()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file, video_url

def main():
    """Main Streamlit application."""
    
    # Create hero section
    create_hero_section()
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create platform showcase
    create_platform_showcase()
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create input section
    uploaded_file, video_url = create_input_section()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Processing Settings")
        
        # Processing options
        max_sentences = st.slider(
            "📝 Summary Length",
            min_value=3,
            max_value=20,
            value=5,
            help="Control the length of the generated summary"
        )
        
        # File type selection
        file_type = st.selectbox(
            "📁 File Type",
            ["Video", "Audio", "Transcript"],
            help="Select the type of file you want to process"
        )
        
        # Processing mode
        processing_mode = st.radio(
            "🚀 Processing Mode",
            ["⚡ Fast (Quick Results)", "🧠 Comprehensive (Full ML Analysis)"],
            help="Fast mode provides quick results, Comprehensive mode uses full ML suite but takes longer"
        )
        comprehensive = processing_mode == "🧠 Comprehensive (Full ML Analysis)"
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        ✨ **Multi-Platform Support**  
        🎥 **Video Processing**  
        🎵 **Audio Extraction**  
        📝 **Smart Summarization**  
        🎯 **Action Items**  
        🔑 **Keyword Analysis**  
        👥 **Entity Recognition**  
        📊 **Analytics & Insights**
        """)
        
        st.markdown("---")
        st.markdown("### 🌐 Quick Platform Access")
        video_manager = VideoSourceManager()
        platforms = video_manager.get_supported_platforms()
        
        # Create a compact platform list
        platform_text = ""
        for i, platform in enumerate(platforms[:8]):
            icon = video_manager.get_platform_icon(platform)
            platform_text += f"{icon} {platform.title()}"
            if i < 7:
                platform_text += " • "
        
        st.markdown(platform_text)
        if len(platforms) > 8:
            st.caption(f"... and {len(platforms) - 8} more!")
    
    # Main processing section
    if uploaded_file is not None or video_url is not None:
        st.markdown("---")
        
        # Display file info with better styling
        if uploaded_file:
            st.success(f"✅ **File Ready:** {uploaded_file.name}")
        elif video_url:
            st.success(f"✅ **Video URL Ready:** {video_url}")
        
        # Process button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Video", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing your video... This may take a few minutes."):
                    try:
                        # Handle file or video URL
                        if uploaded_file:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                        elif video_url:
                            # Download video from any supported platform
                            st.info("📥 Downloading video from platform...")
                            tmp_path = download_video_from_url(video_url)
                            if not tmp_path:
                                st.error("Failed to download video")
                                return
                        
                        # Initialize summarizer
                        summarizer = EnhancedVideoSummarizer()
                        
                        # Load models (including speech recognizer)
                        summarizer.load_models()
                        
                        # Process based on file type
                        if file_type == "Video":
                            results = summarizer.process_video(tmp_path, max_sentences=max_sentences, comprehensive=comprehensive)
                        elif file_type == "Audio":
                            results = summarizer.process_audio(tmp_path, max_sentences=max_sentences)
                        else:  # Transcript
                            results = summarizer.process_transcript(tmp_path, max_sentences=max_sentences)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        if results["success"]:
                            st.success("🎉 Processing completed successfully!")
                            
                            # Store results in session state
                            st.session_state['results'] = results
                            st.session_state['summary_data'] = results['summary_data']
                            
                            # Show summary immediately with better styling
                            st.markdown("### 📝 Summary")
                            st.markdown(f'<div class="summary-box">{results["summary_data"]["summary"]}</div>', unsafe_allow_html=True)
                            
                        else:
                            st.error(f"❌ Error: {results['error']}")
                            
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
                        logger.error(f"Streamlit processing error: {str(e)}")
    
    # Results section with better layout
    if 'summary_data' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Create two-column layout for results
        col1, col2 = st.columns([2, 1])
    
        with col2:
            st.markdown("### 📈 Quick Stats")
            
            if 'summary_data' in st.session_state:
                summary_data = st.session_state['summary_data']
                metadata = summary_data.get('metadata', {})
                
                # Create beautiful metric cards
                st.markdown("""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Sentences</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Keywords</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Action Items</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Compression</div>
                    </div>
                </div>
                """.format(
                    metadata.get('summary_sentence_count', 0),
                    metadata.get('keyword_count', 0),
                    metadata.get('action_item_count', 0),
                    f"{metadata.get('compression_ratio', 0):.1%}"
                ), unsafe_allow_html=True)
                
                # Compression ratio visualization
                if 'original_sentence_count' in metadata and 'summary_sentence_count' in metadata:
                    fig = go.Figure(data=[
                        go.Bar(name='Original', x=['Sentences'], y=[metadata['original_sentence_count']], 
                               marker_color='rgba(102, 126, 234, 0.6)'),
                        go.Bar(name='Summary', x=['Sentences'], y=[metadata['summary_sentence_count']], 
                               marker_color='rgba(102, 126, 234, 1)')
                    ])
                    fig.update_layout(
                        title="Text Compression",
                        height=300,
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload and process a file to see statistics")
    
        with col1:
            # Detailed results with beautiful tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Summary", "🎯 Action Items", "🔑 Keywords", "👥 Entities", "📊 Analysis"])
            
            with tab1:
                st.markdown("### 📝 AI-Generated Summary")
                st.markdown(f'<div class="summary-box">{summary_data["summary"]}</div>', unsafe_allow_html=True)
                
                st.markdown("### 📄 Full Transcript")
                with st.expander("View Full Transcript", expanded=False):
                    st.write(summary_data['transcript'])
            
            with tab2:
                st.markdown("### 🎯 Action Items")
                action_items = summary_data.get('action_items', [])
                if action_items:
                    for i, item in enumerate(action_items, 1):
                        st.markdown(f'<div class="action-item"><strong>{i}.</strong> {item}</div>', unsafe_allow_html=True)
                else:
                    st.info("No action items found in the content")
            
            with tab3:
                st.markdown("### 🔑 Keywords & Topics")
                keywords = summary_data.get('keywords', [])
                if keywords:
                    # Create keyword visualization
                    keyword_df = pd.DataFrame({
                        'keyword': keywords[:15],  # Top 15 keywords
                        'frequency': range(len(keywords[:15]), 0, -1)
                    })
                    
                    # Keyword bar chart with better styling
                    fig = px.bar(keyword_df, x='frequency', y='keyword', orientation='h',
                               title="Top Keywords", color='frequency',
                               color_continuous_scale='Blues')
                    fig.update_layout(
                        height=400, 
                        yaxis={'categoryorder': 'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No keywords extracted")
            
            with tab4:
                st.markdown("### 👥 Named Entities")
                entities = summary_data.get('named_entities', [])
                if entities:
                    # Group entities by type
                    entity_df = pd.DataFrame(entities, columns=['Entity', 'Type'])
                    entity_counts = entity_df['Type'].value_counts()
                    
                    # Entity type pie chart with better styling
                    fig = px.pie(values=entity_counts.values, names=entity_counts.index,
                               title="Named Entity Types",
                               color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Entity table
                    st.markdown("#### All Named Entities")
                    st.dataframe(entity_df, use_container_width=True)
                else:
                    st.info("No named entities found")
            
            with tab5:
                st.markdown("### 📊 Detailed Analysis")
                metadata = summary_data.get('metadata', {})
                
                # Analysis metrics in a grid
                st.markdown("""
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 2rem;">
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Original Sentences</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Summary Sentences</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Compression Ratio</div>
                    </div>
                </div>
                """.format(
                    metadata.get('original_sentence_count', 0),
                    metadata.get('summary_sentence_count', 0),
                    f"{metadata.get('compression_ratio', 0):.1%}"
                ), unsafe_allow_html=True)
                
                # Download section
                st.markdown("### 📥 Download Results")
                
                if 'results' in st.session_state:
                    results = st.session_state['results']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'transcript_path' in results:
                            with open(results['transcript_path'], 'r', encoding='utf-8') as f:
                                transcript_content = f.read()
                            st.download_button(
                                label="📄 Download Transcript",
                                data=transcript_content,
                                file_name="transcript.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                    
                    with col2:
                        if 'summary_path' in results:
                            with open(results['summary_path'], 'r', encoding='utf-8') as f:
                                summary_content = f.read()
                            st.download_button(
                                label="📋 Download Summary",
                                data=summary_content,
                                file_name="summary.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                    
                    with col3:
                        # Download JSON data
                        import json
                        json_data = json.dumps(summary_data, indent=2)
                        st.download_button(
                            label="📊 Download JSON Data",
                            data=json_data,
                            file_name="summary_data.json",
                            mime="application/json",
                            use_container_width=True
                        )
    
    # Beautiful Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-bottom: 2rem;">
            <div>
                <h4 style="color: #2d3748; margin-bottom: 1rem;">🎥 AI Video Summarizer</h4>
                <p style="color: #718096; line-height: 1.6;">Transform any video into intelligent summaries with action items, keywords, and insights.</p>
            </div>
            <div>
                <h4 style="color: #2d3748; margin-bottom: 1rem;">🌐 Supported Platforms</h4>
                <p style="color: #718096; line-height: 1.6;">YouTube, Vimeo, Instagram, TikTok, Facebook, Twitter, Twitch, and 1000+ more platforms!</p>
            </div>
            <div>
                <h4 style="color: #2d3748; margin-bottom: 1rem;">🚀 Features</h4>
                <p style="color: #718096; line-height: 1.6;">Multi-platform support, AI-powered analysis, real-time processing, and comprehensive insights.</p>
            </div>
        </div>
        <div style="border-top: 1px solid #e2e8f0; padding-top: 1rem; color: #718096;">
            <p>🤖 Powered by AI | Built with Streamlit | Video Summarizer v2.0 | Multi-Platform Edition</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
