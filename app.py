"""
Advanced Flask web application for Video Summarizer.
Real video processing with ML and NLP capabilities.
"""

from flask import Flask, render_template, request, jsonify
import os
import tempfile
import json
from pathlib import Path
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Import video processor
from video_processor import VideoProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize video processor
video_processor = VideoProcessor()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process uploaded video file with real ML processing."""
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
        max_length = int(request.form.get('max_sentences', 5)) * 30  # Convert to approximate tokens
        processing_mode = request.form.get('processing_mode', 'fast')
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        video_path = os.path.join(tempfile.gettempdir(), unique_filename)
        file.save(video_path)
        
        try:
            # Process video with real ML models
            result = video_processor.process_video_file(video_path, max_length)
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'data': result
                })
            else:
                return jsonify({'error': result.get('error', 'Processing failed')}), 500
                
        finally:
            # Clean up uploaded file
            if os.path.exists(video_path):
                os.unlink(video_path)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/process-url', methods=['POST'])
def process_url():
    """Process video from URL."""
    try:
        data = request.get_json()
        url = data.get('url')
        max_sentences = data.get('max_sentences', 5)
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Convert sentences to approximate tokens
        max_length = max_sentences * 30
        
        # Process video with real ML models
        result = video_processor.process_video_url(url, max_length)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': result
            })
        else:
            return jsonify({'error': result.get('error', 'Processing failed')}), 500
        
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

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
    
    print("Starting AI Video Summarizer (Simplified)...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
