"""
Test script to verify the Video Summarizer installation.
"""
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
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
        import nltk
        print("[OK] NLTK imported successfully")
    except ImportError as e:
        print(f"[ERROR] NLTK import failed: {e}")
        return False
    
    try:
        import streamlit
        print("[OK] Streamlit imported successfully")
    except ImportError as e:
        print(f"[ERROR] Streamlit import failed: {e}")
        return False
    
    try:
        import sklearn
        print("[OK] Scikit-learn imported successfully")
    except ImportError as e:
        print(f"[ERROR] Scikit-learn import failed: {e}")
        return False
    
    try:
        import spacy
        print("[OK] spaCy imported successfully")
    except ImportError as e:
        print(f"[ERROR] spaCy import failed: {e}")
        print("  Note: You may need to download a spaCy model: python -m spacy download en_core_web_sm")
    
    return True

def test_project_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        from config import AUDIO_SETTINGS, SUMMARIZATION_SETTINGS
        print("[OK] Config module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Config import failed: {e}")
        return False
    
    try:
        from utils.logger import setup_logger
        print("[OK] Logger utility imported successfully")
    except ImportError as e:
        print(f"[ERROR] Logger utility import failed: {e}")
        return False
    
    try:
        from audioExtraction import AudioExtractor
        print("[OK] Audio extraction module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Audio extraction import failed: {e}")
        return False
    
    try:
        from transcriptCreation import TranscriptGenerator
        print("[OK] Transcript creation module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Transcript creation import failed: {e}")
        return False
    
    try:
        from transcriptSummariser import AdvancedSummarizer
        print("[OK] Summarization module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Summarization import failed: {e}")
        return False
    
    try:
        from main import VideoSummarizer
        print("[OK] Main application imported successfully")
    except ImportError as e:
        print(f"[ERROR] Main application import failed: {e}")
        return False
    
    return True

def test_directory_structure():
    """Test if required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "data",
        "models", 
        "output",
        "output/audio",
        "output/transcripts",
        "output/summaries",
        "logs",
        "utils"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"[OK] Directory exists: {dir_path}")
        else:
            print(f"[ERROR] Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def test_basic_functionality():
    """Test basic functionality without processing files."""
    print("\nTesting basic functionality...")
    
    try:
        from main import VideoSummarizer
        summarizer = VideoSummarizer()
        print("[OK] VideoSummarizer initialized successfully")
    except Exception as e:
        print(f"[ERROR] VideoSummarizer initialization failed: {e}")
        return False
    
    try:
        from transcriptSummariser import AdvancedSummarizer
        summarizer = AdvancedSummarizer()
        print("[OK] AdvancedSummarizer initialized successfully")
    except Exception as e:
        print(f"[ERROR] AdvancedSummarizer initialization failed: {e}")
        return False
    
    try:
        from email_sender import EmailSender
        sender = EmailSender()
        print("[OK] EmailSender initialized successfully")
    except Exception as e:
        print(f"[ERROR] EmailSender initialization failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Video Summarizer - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
    
    if not test_project_modules():
        all_passed = False
    
    if not test_directory_structure():
        all_passed = False
    
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Download spaCy model: python -m spacy download en_core_web_sm")
        print("2. Run the web interface: streamlit run streamlit_app.py")
        print("3. Or use the CLI: python main.py your_video.mp4")
    else:
        print("Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that you're in the correct directory")
        print("3. Verify Python version is 3.8 or higher")

if __name__ == "__main__":
    main()
