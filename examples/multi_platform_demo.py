#!/usr/bin/env python3
"""
Multi-Platform Video Summarization Demo

This script demonstrates the enhanced video summarizer's ability to process
videos from multiple platforms including YouTube, Vimeo, Instagram, TikTok,
Facebook, Twitter, Twitch, and many more.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml_main import EnhancedVideoSummarizer
from utils.video_source_manager import VideoSourceManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

def demo_platform_detection():
    """Demonstrate platform detection capabilities."""
    print("🔍 Platform Detection Demo")
    print("=" * 50)
    
    # Sample URLs from different platforms
    sample_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://vimeo.com/123456789",
        "https://www.instagram.com/p/ABC123/",
        "https://www.tiktok.com/@user/video/1234567890",
        "https://www.facebook.com/watch/?v=123456789",
        "https://twitter.com/user/status/1234567890",
        "https://www.twitch.tv/videos/123456789",
        "https://www.dailymotion.com/video/abc123",
        "https://www.bilibili.com/video/BV1234567890",
        "https://rumble.com/v1234567890-abc",
        "https://odysee.com/@channel/video-title",
        "https://lbry.tv/@channel/video-title"
    ]
    
    video_manager = VideoSourceManager()
    
    for url in sample_urls:
        platform = video_manager.detect_platform(url)
        icon = video_manager.get_platform_icon(platform) if platform else "❌"
        description = video_manager.get_platform_description(platform) if platform else "Unsupported"
        
        print(f"{icon} {url}")
        print(f"   Platform: {platform or 'Unknown'}")
        print(f"   Description: {description}")
        print()

def demo_video_info_extraction():
    """Demonstrate video information extraction."""
    print("📋 Video Information Extraction Demo")
    print("=" * 50)
    
    # Note: This would require actual URLs to work
    print("To test video info extraction, provide a real video URL:")
    print("Example: python multi_platform_demo.py --url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
    print()

def demo_video_processing():
    """Demonstrate video processing from URLs."""
    print("🎥 Video Processing Demo")
    print("=" * 50)
    
    # Initialize the enhanced summarizer
    summarizer = EnhancedVideoSummarizer()
    summarizer.load_models()
    
    print("Enhanced Video Summarizer initialized with multi-platform support!")
    print("Supported platforms:")
    
    video_manager = VideoSourceManager()
    platforms = video_manager.get_supported_platforms()
    
    for i, platform in enumerate(platforms, 1):
        icon = video_manager.get_platform_icon(platform)
        print(f"  {i:2d}. {icon} {platform.title()}")
    
    print(f"\nTotal supported platforms: {len(platforms)}")
    print()

def demo_cli_usage():
    """Demonstrate CLI usage examples."""
    print("💻 CLI Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "description": "Process YouTube video (fast mode)",
            "command": "python ml_main.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'"
        },
        {
            "description": "Process Vimeo video (comprehensive mode)",
            "command": "python ml_main.py 'https://vimeo.com/123456789' --comprehensive"
        },
        {
            "description": "Process Instagram video with custom output",
            "command": "python ml_main.py 'https://www.instagram.com/p/ABC123/' -o my_summary -s 10"
        },
        {
            "description": "Process TikTok video with visual analysis",
            "command": "python ml_main.py 'https://www.tiktok.com/@user/video/1234567890' --comprehensive --visual-analysis"
        },
        {
            "description": "Process Twitter video",
            "command": "python ml_main.py 'https://twitter.com/user/status/1234567890'"
        },
        {
            "description": "Process Twitch video",
            "command": "python ml_main.py 'https://www.twitch.tv/videos/123456789' --comprehensive"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   Command: {example['command']}")
        print()

def demo_streamlit_usage():
    """Demonstrate Streamlit usage."""
    print("🌐 Streamlit Web Interface")
    print("=" * 50)
    
    print("To use the web interface with multi-platform support:")
    print("1. Start the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print()
    print("2. In the web interface:")
    print("   - Select '🌐 Video URL' input method")
    print("   - Paste any supported video URL")
    print("   - The system will automatically detect the platform")
    print("   - Choose processing mode (Fast or Comprehensive)")
    print("   - Click '🚀 Process File' to start")
    print()
    print("3. Supported platforms in the UI:")
    print("   - YouTube, Vimeo, Instagram, TikTok")
    print("   - Facebook, Twitter, Twitch")
    print("   - Dailymotion, Bilibili, Rumble")
    print("   - Odysee, LBRY, and 1000+ more!")
    print()

def demo_api_usage():
    """Demonstrate API usage."""
    print("🔧 Python API Usage")
    print("=" * 50)
    
    code_example = '''
from ml_main import EnhancedVideoSummarizer

# Initialize the summarizer
summarizer = EnhancedVideoSummarizer()
summarizer.load_models()

# Process video from any supported platform
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
results = summarizer.process_video(url, max_sentences=5, comprehensive=True)

if results["success"]:
    print(f"Summary: {results['summary_data']['summary']}")
    print(f"Platform: {results['platform_info']['platform']}")
    print(f"Video Title: {results['platform_info']['video_title']}")
else:
    print(f"Error: {results['error']}")
'''
    
    print("Example Python code:")
    print(code_example)

def main():
    """Main demo function."""
    print("🎥 Multi-Platform Video Summarization Demo")
    print("=" * 60)
    print()
    
    # Check if yt-dlp is available
    try:
        import yt_dlp
        print("✅ yt-dlp is available - Multi-platform support enabled!")
    except ImportError:
        print("❌ yt-dlp not found - Install with: pip install yt-dlp")
        print("   Multi-platform support will be limited.")
    print()
    
    # Run demos
    demo_platform_detection()
    demo_video_processing()
    demo_cli_usage()
    demo_streamlit_usage()
    demo_api_usage()
    
    print("🚀 Ready to process videos from 1000+ platforms!")
    print("Try it out with your favorite video URLs!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Platform Video Summarization Demo")
    parser.add_argument("--url", help="Test with a specific video URL")
    parser.add_argument("--info", action="store_true", help="Show video information only")
    parser.add_argument("--process", action="store_true", help="Process the video")
    
    args = parser.parse_args()
    
    if args.url:
        print(f"🎥 Testing with URL: {args.url}")
        print("=" * 50)
        
        video_manager = VideoSourceManager()
        
        # Validate URL
        is_valid, message = video_manager.validate_url(args.url)
        print(f"URL Validation: {'✅' if is_valid else '❌'} {message}")
        
        if is_valid:
            platform = video_manager.detect_platform(args.url)
            print(f"Platform: {video_manager.get_platform_icon(platform)} {platform.title()}")
            
            if args.info:
                print("\n📋 Video Information:")
                info = video_manager.get_platform_info(args.url)
                for key, value in info.items():
                    if key != 'error':
                        print(f"  {key}: {value}")
            
            if args.process:
                print("\n🎬 Processing video...")
                summarizer = EnhancedVideoSummarizer()
                summarizer.load_models()
                
                results = summarizer.process_video(args.url, max_sentences=5, comprehensive=False)
                
                if results["success"]:
                    print("✅ Processing completed!")
                    print(f"Summary: {results['summary_data']['summary']}")
                    if 'platform_info' in results:
                        print(f"Platform: {results['platform_info']['platform']}")
                        print(f"Title: {results['platform_info']['video_title']}")
                else:
                    print(f"❌ Error: {results['error']}")
    else:
        main()
