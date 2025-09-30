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

# Try to import yt-dlp for YouTube processing
try:
    import yt_dlp
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

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

def download_youtube_video(url: str, output_dir: str = "temp") -> Optional[str]:
    """Download YouTube video and return the file path."""
    if not YOUTUBE_AVAILABLE:
        st.error("YouTube processing requires yt-dlp. Please install it: pip install yt-dlp")
        return None
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'format': 'best[height<=720]',  # Limit to 720p for faster processing
            'noplaylist': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            
            # Download the video
            ydl.download([url])
            
            # Find the downloaded file
            for file in os.listdir(output_dir):
                if file.endswith(('.mp4', '.webm', '.mkv', '.avi')):
                    return os.path.join(output_dir, file)
        
        return None
        
    except Exception as e:
        st.error(f"Error downloading YouTube video: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .summary-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .action-item {
        background: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎥 AI Video Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Processing options
        max_sentences = st.slider(
            "Maximum sentences in summary",
            min_value=3,
            max_value=20,
            value=5,
            help="Control the length of the generated summary"
        )
        
        # File type selection
        file_type = st.selectbox(
            "Input file type",
            ["Video", "Audio", "Transcript"],
            help="Select the type of file you want to process"
        )
        
        # Processing mode
        processing_mode = st.radio(
            "Processing Mode:",
            ["⚡ Fast (Quick Results)", "🧠 Comprehensive (Full ML Analysis)"],
            help="Fast mode provides quick results, Comprehensive mode uses full ML suite but takes longer"
        )
        comprehensive = processing_mode == "🧠 Comprehensive (Full ML Analysis)"
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        This AI-powered tool can:
        - Extract audio from videos
        - Generate transcripts using speech recognition
        - Create intelligent summaries
        - Extract action items and keywords
        - Identify named entities
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload File or YouTube URL")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload File", "🎥 YouTube URL"],
            horizontal=True
        )
        
        uploaded_file = None
        youtube_url = None
        
        if input_method == "📁 Upload File":
            # File upload
            uploaded_file = st.file_uploader(
                f"Choose a {file_type.lower()} file",
                type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'wav', 'mp3', 'm4a', 'flac', 'txt'],
                help=f"Upload a {file_type.lower()} file to process"
            )
        else:
            # YouTube URL input
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste a YouTube video URL to process"
            )
            
            if youtube_url and not is_youtube_url(youtube_url):
                st.error("Please enter a valid YouTube URL")
                youtube_url = None
        
        if uploaded_file is not None or youtube_url is not None:
            # Display file info
            if uploaded_file:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
            elif youtube_url:
                st.success(f"✅ YouTube URL: {youtube_url}")
            
            # Process button
            if st.button("🚀 Process File", type="primary"):
                with st.spinner("Processing file... This may take a few minutes."):
                    try:
                        # Handle file or YouTube URL
                        if uploaded_file:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                        elif youtube_url:
                            # Download YouTube video
                            st.info("📥 Downloading YouTube video...")
                            tmp_path = download_youtube_video(youtube_url)
                            if not tmp_path:
                                st.error("Failed to download YouTube video")
                                return
                        
                        # Initialize summarizer
                        summarizer = EnhancedVideoSummarizer()
                        
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
                            st.success("✅ Processing completed successfully!")
                            
                            # Store results in session state
                            st.session_state['results'] = results
                            st.session_state['summary_data'] = results['summary_data']
                            
                            # Show summary immediately
                            st.markdown("### 📝 Summary")
                            st.write(results['summary_data']['summary'])
                            
                            
                        else:
                            st.error(f"❌ Error: {results['error']}")
                            
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
                        logger.error(f"Streamlit processing error: {str(e)}")
    
    with col2:
        st.header("📈 Quick Stats")
        
        if 'summary_data' in st.session_state:
            summary_data = st.session_state['summary_data']
            metadata = summary_data.get('metadata', {})
            
            
            # Metrics
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Sentences", metadata.get('summary_sentence_count', 0))
                st.metric("Keywords", metadata.get('keyword_count', 0))
            
            with col_b:
                st.metric("Action Items", metadata.get('action_item_count', 0))
                st.metric("Compression", f"{metadata.get('compression_ratio', 0):.1%}")
            
            # Compression ratio visualization
            if 'original_sentence_count' in metadata and 'summary_sentence_count' in metadata:
                fig = go.Figure(data=[
                    go.Bar(name='Original', x=['Sentences'], y=[metadata['original_sentence_count']], marker_color='lightblue'),
                    go.Bar(name='Summary', x=['Sentences'], y=[metadata['summary_sentence_count']], marker_color='darkblue')
                ])
                fig.update_layout(title="Text Compression", height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload and process a file to see statistics")
    
    # Results section
    if 'summary_data' in st.session_state:
        st.markdown("---")
        st.header("📋 Results")
        
        summary_data = st.session_state['summary_data']
        
        # Show basic info immediately
        st.markdown("### 📝 Summary")
        st.write(summary_data['summary'])
        
        st.markdown("### 📄 Transcript")
        st.write(summary_data['transcript'])
        
        # Tabs for different result types
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Summary", "🎯 Action Items", "🔑 Keywords", "👥 Entities", "📊 Analysis"])
        
        with tab1:
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.write(summary_data['summary'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            action_items = summary_data.get('action_items', [])
            if action_items:
                for i, item in enumerate(action_items, 1):
                    st.markdown(f'<div class="action-item"><strong>{i}.</strong> {item}</div>', unsafe_allow_html=True)
            else:
                st.info("No action items found in the content")
        
        with tab3:
            keywords = summary_data.get('keywords', [])
            if keywords:
                # Create keyword cloud data
                keyword_df = pd.DataFrame({
                    'keyword': keywords[:20],  # Top 20 keywords
                    'frequency': range(len(keywords[:20]), 0, -1)
                })
                
                # Keyword bar chart
                fig = px.bar(keyword_df, x='frequency', y='keyword', orientation='h',
                           title="Top Keywords", color='frequency',
                           color_continuous_scale='Blues')
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No keywords extracted")
        
        with tab4:
            entities = summary_data.get('named_entities', [])
            if entities:
                # Group entities by type
                entity_df = pd.DataFrame(entities, columns=['Entity', 'Type'])
                entity_counts = entity_df['Type'].value_counts()
                
                # Entity type pie chart
                fig = px.pie(values=entity_counts.values, names=entity_counts.index,
                           title="Named Entity Types")
                st.plotly_chart(fig, use_container_width=True)
                
                # Entity table
                st.subheader("All Named Entities")
                st.dataframe(entity_df, use_container_width=True)
            else:
                st.info("No named entities found")
        
        with tab5:
            metadata = summary_data.get('metadata', {})
            
            # Analysis metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Sentences", metadata.get('original_sentence_count', 0))
                st.metric("Summary Sentences", metadata.get('summary_sentence_count', 0))
            
            with col2:
                st.metric("Compression Ratio", f"{metadata.get('compression_ratio', 0):.1%}")
                st.metric("Keywords Found", metadata.get('keyword_count', 0))
            
            with col3:
                st.metric("Action Items", metadata.get('action_item_count', 0))
                st.metric("Named Entities", metadata.get('named_entity_count', 0))
            
            # Download section
            st.subheader("📥 Download Results")
            
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
                            mime="text/plain"
                        )
                
                with col2:
                    if 'summary_path' in results:
                        with open(results['summary_path'], 'r', encoding='utf-8') as f:
                            summary_content = f.read()
                        st.download_button(
                            label="📋 Download Summary",
                            data=summary_content,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
                
                with col3:
                    # Download JSON data
                    import json
                    json_data = json.dumps(summary_data, indent=2)
                    st.download_button(
                        label="📊 Download JSON Data",
                        data=json_data,
                        file_name="summary_data.json",
                        mime="application/json"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🤖 Powered by AI | Built with Streamlit | Video Summarizer v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
