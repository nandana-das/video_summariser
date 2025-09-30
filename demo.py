"""
Demo script to show the Video Summarizer functionality.
"""
import sys
from pathlib import Path

def demo_basic_functionality():
    """Demonstrate basic functionality without problematic imports."""
    print("Video Summarizer - Demo")
    print("=" * 50)
    
    # Test basic imports
    print("Testing basic imports...")
    try:
        import moviepy
        print("[OK] MoviePy imported successfully")
    except ImportError as e:
        print(f"[ERROR] MoviePy import failed: {e}")
        return False
    
    try:
        import vosk
        print("[OK] Vosk imported successfully")
    except ImportError as e:
        print(f"[ERROR] Vosk import failed: {e}")
        return False
    
    try:
        import torch
        print("[OK] PyTorch imported successfully")
    except ImportError as e:
        print(f"[ERROR] PyTorch import failed: {e}")
        return False
    
    try:
        import streamlit
        print("[OK] Streamlit imported successfully")
    except ImportError as e:
        print(f"[ERROR] Streamlit import failed: {e}")
        return False
    
    # Test project modules that work
    print("\nTesting working project modules...")
    try:
        from config import AUDIO_SETTINGS, SUMMARIZATION_SETTINGS
        print("[OK] Config module imported successfully")
        print(f"  - Supported video formats: {AUDIO_SETTINGS['supported_formats']}")
        print(f"  - Max sentences in summary: {SUMMARIZATION_SETTINGS['max_sentences']}")
    except ImportError as e:
        print(f"[ERROR] Config import failed: {e}")
        return False
    
    try:
        from utils.logger import setup_logger
        logger = setup_logger("demo")
        print("[OK] Logger utility imported successfully")
        logger.info("Demo logger is working!")
    except ImportError as e:
        print(f"[ERROR] Logger utility import failed: {e}")
        return False
    
    try:
        from audioExtraction import AudioExtractor
        extractor = AudioExtractor()
        print("[OK] Audio extraction module imported successfully")
        print(f"  - Supported formats: {extractor.supported_formats}")
    except ImportError as e:
        print(f"[ERROR] Audio extraction import failed: {e}")
        return False
    
    # Test directory structure
    print("\nTesting directory structure...")
    required_dirs = ["data", "models", "output", "output/audio", "output/transcripts", "output/summaries", "logs", "utils"]
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"[OK] Directory exists: {dir_path}")
        else:
            print(f"[ERROR] Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def demo_web_interface():
    """Show how to run the web interface."""
    print("\n" + "=" * 50)
    print("Web Interface Demo")
    print("=" * 50)
    print("To run the web interface:")
    print("1. Open a new terminal/command prompt")
    print("2. Navigate to the project directory")
    print("3. Run: streamlit run streamlit_app.py")
    print("4. Open your browser to http://localhost:8501")
    print("\nThe web interface provides:")
    print("- File upload for videos, audio, or transcripts")
    print("- Real-time processing status")
    print("- Interactive visualizations")
    print("- Download options for results")

def demo_cli_usage():
    """Show CLI usage examples."""
    print("\n" + "=" * 50)
    print("Command Line Interface Demo")
    print("=" * 50)
    print("To use the CLI:")
    print("\n1. Process a video file:")
    print("   python main.py video.mp4")
    print("\n2. Process an audio file:")
    print("   python main.py audio.wav -t audio")
    print("\n3. Process a transcript:")
    print("   python main.py transcript.txt -t transcript")
    print("\n4. Customize summary length:")
    print("   python main.py video.mp4 -s 10")
    print("\n5. Specify output name:")
    print("   python main.py video.mp4 -o my_meeting")

def demo_python_api():
    """Show Python API usage."""
    print("\n" + "=" * 50)
    print("Python API Demo")
    print("=" * 50)
    print("To use the Python API:")
    print("\n```python")
    print("from main import VideoSummarizer")
    print("")
    print("# Initialize the summarizer")
    print("summarizer = VideoSummarizer()")
    print("")
    print("# Process a video")
    print("results = summarizer.process_video('meeting.mp4')")
    print("")
    print("if results['success']:")
    print("    print('Summary:', results['summary_data']['summary'])")
    print("    print('Action items:', results['summary_data']['action_items'])")
    print("```")

def main():
    """Run the demo."""
    print("Video Summarizer - Complete Demo")
    print("=" * 50)
    
    # Test basic functionality
    if demo_basic_functionality():
        print("\n[SUCCESS] Basic functionality test passed!")
    else:
        print("\n[WARNING] Some basic functionality tests failed.")
    
    # Show usage examples
    demo_web_interface()
    demo_cli_usage()
    demo_python_api()
    
    print("\n" + "=" * 50)
    print("Next Steps:")
    print("1. Download spaCy model: python -m spacy download en_core_web_sm")
    print("2. Run the web interface: streamlit run streamlit_app.py")
    print("3. Or use the CLI: python main.py your_video.mp4")
    print("4. Check the README.md for detailed documentation")
    print("=" * 50)

if __name__ == "__main__":
    main()
