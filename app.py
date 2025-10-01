"""
Flask web application for Video Summarizer.
Modern web interface without Streamlit dependencies.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import json
from pathlib import Path
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Import our video processing modules
from ml_main import EnhancedVideoSummarizer
from utils.video_source_manager import VideoSourceManager
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process uploaded video file."""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
        
        # Get processing parameters
        max_sentences = int(request.form.get('max_sentences', 5))
        processing_mode = request.form.get('processing_mode', 'fast')
        comprehensive = processing_mode == 'comprehensive'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(tempfile.gettempdir(), unique_filename)
        file.save(file_path)
        
        try:
            # Initialize video summarizer
            summarizer = EnhancedVideoSummarizer(use_mlflow=False)
            summarizer.load_models()
            
            # Process video
            results = summarizer.process_video(
                video_path=file_path,
                max_sentences=max_sentences,
                comprehensive=comprehensive
            )
            
            if results['success']:
                # Clean up file
                os.unlink(file_path)
                
                return jsonify({
                    'success': True,
                    'data': results['summary_data']
                })
            else:
                os.unlink(file_path)
                return jsonify({'error': results['error']}), 500
                
        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.unlink(file_path)
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in process_video: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/process-url', methods=['POST'])
def process_url():
    """Process video from URL."""
    try:
        data = request.get_json()
        url = data.get('url')
        max_sentences = data.get('max_sentences', 5)
        processing_mode = data.get('processing_mode', 'fast')
        comprehensive = processing_mode == 'comprehensive'
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Initialize video source manager
        source_manager = VideoSourceManager()
        
        # Download video
        video_path = source_manager.download_video(url)
        if not video_path:
            return jsonify({'error': 'Failed to download video'}), 500
        
        try:
            # Initialize video summarizer
            summarizer = EnhancedVideoSummarizer(use_mlflow=False)
            summarizer.load_models()
            
            # Process video
            results = summarizer.process_video(
                video_path=video_path,
                max_sentences=max_sentences,
                comprehensive=comprehensive
            )
            
            if results['success']:
                # Clean up file
                os.unlink(video_path)
                
                return jsonify({
                    'success': True,
                    'data': results['summary_data']
                })
            else:
                os.unlink(video_path)
                return jsonify({'error': results['error']}), 500
                
        except Exception as e:
            # Clean up file on error
            if os.path.exists(video_path):
                os.unlink(video_path)
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in process_url: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/platforms')
def get_platforms():
    """Get supported video platforms."""
    platforms = [
        {
            'name': 'YouTube',
            'description': 'World\'s largest video platform',
            'icon': '🎥',
            'color': '#FF0000'
        },
        {
            'name': 'Vimeo',
            'description': 'High-quality video hosting',
            'icon': '🎬',
            'color': '#1AB7EA'
        },
        {
            'name': 'Instagram',
            'description': 'Social media videos',
            'icon': '📸',
            'color': '#E4405F'
        },
        {
            'name': 'TikTok',
            'description': 'Short-form video content',
            'icon': '🎵',
            'color': '#000000'
        },
        {
            'name': 'Facebook',
            'description': 'Social media videos',
            'icon': '👥',
            'color': '#1877F2'
        },
        {
            'name': 'Twitter',
            'description': 'Micro-video content',
            'icon': '🐦',
            'color': '#1DA1F2'
        },
        {
            'name': 'Twitch',
            'description': 'Live streaming platform',
            'icon': '🎮',
            'color': '#9146FF'
        },
        {
            'name': 'Dailymotion',
            'description': 'Video sharing platform',
            'icon': '🎞️',
            'color': '#0066CC'
        }
    ]
    
    return jsonify({'platforms': platforms})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
