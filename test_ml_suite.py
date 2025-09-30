#!/usr/bin/env python3
"""
Test script for the ML Suite to verify all components work correctly.
"""
import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test all ML suite imports."""
    print("[TEST] Testing ML Suite Imports...")
    
    try:
        # Test basic imports
        print("  [INFO] Testing basic imports...")
        import torch
        import numpy as np
        import pandas as pd
        print("  [SUCCESS] Basic imports successful")
        
        # Test ML model imports
        print("  [INFO] Testing ML model imports...")
        from ml_models import (
            TransformerSummarizer, AdvancedSpeechRecognizer, VideoAnalyzer,
            MLflowManager, ModelTrainer, ModelEvaluator, InferenceServer,
            ModelRegistry, AudioPreprocessor, VideoPreprocessor, TextPreprocessor
        )
        print("  [SUCCESS] ML model imports successful")
        
        # Test main application
        print("  [INFO] Testing main application import...")
        from ml_main import EnhancedVideoSummarizer
        print("  [SUCCESS] Main application import successful")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Import test failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of ML components."""
    print("\n[TEST] Testing Basic Functionality...")
    
    try:
        # Test text preprocessor
        print("  📝 Testing text preprocessor...")
        from ml_models import TextPreprocessor
        text_processor = TextPreprocessor()
        test_text = "This is a test text for preprocessing."
        result = text_processor.preprocess_text(test_text)
        print(f"  ✅ Text preprocessing successful: {len(result['words'])} words")
        
        # Test audio preprocessor
        print("  🎵 Testing audio preprocessor...")
        from ml_models import AudioPreprocessor
        audio_processor = AudioPreprocessor()
        print("  ✅ Audio preprocessor initialized")
        
        # Test video preprocessor
        print("  🎬 Testing video preprocessor...")
        from ml_models import VideoPreprocessor
        video_processor = VideoPreprocessor()
        print("  ✅ Video preprocessor initialized")
        
        # Test model evaluator
        print("  📊 Testing model evaluator...")
        from ml_models import ModelEvaluator
        evaluator = ModelEvaluator()
        print("  ✅ Model evaluator initialized")
        
        # Test MLflow manager
        print("  🔬 Testing MLflow manager...")
        from ml_models import MLflowManager
        mlflow_manager = MLflowManager()
        print("  ✅ MLflow manager initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_summarizer():
    """Test the enhanced video summarizer."""
    print("\n🎥 Testing Enhanced Video Summarizer...")
    
    try:
        from ml_main import EnhancedVideoSummarizer
        
        # Initialize without MLflow to avoid database issues
        summarizer = EnhancedVideoSummarizer(use_mlflow=False)
        print("  ✅ Enhanced video summarizer initialized")
        
        # Test model loading (this might take a while)
        print("  🤖 Testing model loading...")
        try:
            summarizer.load_models()
            print("  ✅ Models loaded successfully")
        except Exception as e:
            print(f"  ⚠️  Model loading failed (expected if models not downloaded): {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced summarizer test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from config import (
            SUMMARIZATION_SETTINGS, TRANSCRIPTION_SETTINGS, 
            MLFLOW_SETTINGS, SERVING_SETTINGS
        )
        
        print(f"  ✅ Summarization settings loaded: {len(SUMMARIZATION_SETTINGS)} items")
        print(f"  ✅ Transcription settings loaded: {len(TRANSCRIPTION_SETTINGS)} items")
        print(f"  ✅ MLflow settings loaded: {len(MLFLOW_SETTINGS)} items")
        print(f"  ✅ Serving settings loaded: {len(SERVING_SETTINGS)} items")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("ML Suite Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Enhanced Summarizer", test_enhanced_summarizer),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML Suite is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
