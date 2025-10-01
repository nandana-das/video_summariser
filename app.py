"""
Simplified Flask web application for Video Summarizer.
No MLflow dependencies - just core functionality.
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

def simple_summarize(text, max_sentences=5):
    """Simple extractive summarization without ML dependencies."""
    import re
    from collections import Counter
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    # Simple scoring based on word count and position
    scores = []
    for i, sentence in enumerate(sentences):
        # Score based on length and position (first sentences are more important)
        length_score = len(sentence.split())
        position_score = 1.0 / (i + 1)  # First sentences get higher score
        score = length_score + position_score
        scores.append(score)
    
    # Select top sentences
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_sentences]
    top_indices = sorted(top_indices)  # Maintain order
    
    summary_sentences = [sentences[i] for i in top_indices]
    return '. '.join(summary_sentences) + '.'

def extract_keywords(text, num_keywords=10):
    """Extract keywords using simple word frequency."""
    import re
    from collections import Counter
    
    # Simple stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequency
    word_freq = Counter(words)
    return [word for word, freq in word_freq.most_common(num_keywords)]

def extract_action_items(text):
    """Extract potential action items."""
    import re
    
    sentences = re.split(r'[.!?]+', text)
    action_items = []
    
    # Look for action patterns
    action_patterns = [
        r'\b(?:need to|should|must|have to|will|going to)\b',
        r'\b(?:action|task|step|next|follow up|implement)\b',
        r'\b(?:please|make sure|ensure|remember)\b'
    ]
    
    for sentence in sentences:
        for pattern in action_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                action_items.append(sentence.strip())
                break
    
    return action_items[:10]

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
        
        # For now, create a mock transcript since we don't have audio processing
        mock_transcript = """
        This is a sample transcript of the video content. The video discusses important topics 
        related to technology and innovation. We need to focus on key areas that will drive 
        future growth and development. The main points include understanding market trends, 
        implementing new strategies, and ensuring customer satisfaction. 
        
        It's important to remember that we should always prioritize quality over quantity. 
        We must also ensure that our team is properly trained and equipped with the right tools. 
        The next steps involve reviewing our current processes and making necessary improvements.
        
        We should also consider the feedback from our customers and stakeholders. This will help us 
        make better decisions and improve our overall performance. The goal is to create a more 
        efficient and effective organization that can adapt to changing market conditions.
        """
        
        # Generate summary
        summary = simple_summarize(mock_transcript, max_sentences)
        
        # Extract keywords
        keywords = extract_keywords(mock_transcript)
        
        # Extract action items
        action_items = extract_action_items(mock_transcript)
        
        return jsonify({
            'success': True,
            'data': {
                'summary': summary,
                'transcript': mock_transcript,
                'keywords': keywords,
                'action_items': action_items,
                'metadata': {
                    'summary_sentence_count': len(summary.split('.')),
                    'keyword_count': len(keywords),
                    'action_item_count': len(action_items),
                    'compression_ratio': len(summary) / len(mock_transcript)
                }
            }
        })
        
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
        
        # For now, create a mock transcript since we don't have URL processing
        mock_transcript = f"""
        This is a sample transcript from the video at {url}. The video discusses important topics 
        related to technology and innovation. We need to focus on key areas that will drive 
        future growth and development. The main points include understanding market trends, 
        implementing new strategies, and ensuring customer satisfaction. 
        
        It's important to remember that we should always prioritize quality over quantity. 
        We must also ensure that our team is properly trained and equipped with the right tools. 
        The next steps involve reviewing our current processes and making necessary improvements.
        """
        
        # Generate summary
        summary = simple_summarize(mock_transcript, max_sentences)
        
        # Extract keywords
        keywords = extract_keywords(mock_transcript)
        
        # Extract action items
        action_items = extract_action_items(mock_transcript)
        
        return jsonify({
            'success': True,
            'data': {
                'summary': summary,
                'transcript': mock_transcript,
                'keywords': keywords,
                'action_items': action_items,
                'metadata': {
                    'summary_sentence_count': len(summary.split('.')),
                    'keyword_count': len(keywords),
                    'action_item_count': len(action_items),
                    'compression_ratio': len(summary) / len(mock_transcript)
                }
            }
        })
        
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
