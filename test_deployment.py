#!/usr/bin/env python3
"""
Simple test script to verify deployment compatibility.
"""

import sys
import os

def test_imports():
    """Test that all critical imports work."""
    try:
        print("Testing critical imports...")
        
        # Test basic imports
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        import transformers
        print("✅ Transformers imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        # Test optional imports
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ Sentence Transformers available")
        except ImportError:
            print("⚠️  Sentence Transformers not available (optional)")
        
        try:
            from bert_score import score
            print("✅ BERT Score available")
        except ImportError:
            print("⚠️  BERT Score not available (optional)")
        
        # Test main app import
        try:
            from ml_main import EnhancedVideoSummarizer
            print("✅ Main app module imported successfully")
        except Exception as e:
            print(f"❌ Main app import failed: {e}")
            return False
        
        print("\n🎉 All critical imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_app_initialization():
    """Test that the app can initialize without errors."""
    try:
        print("\nTesting app initialization...")
        
        from ml_main import EnhancedVideoSummarizer
        
        # Initialize summarizer
        summarizer = EnhancedVideoSummarizer()
        print("✅ EnhancedVideoSummarizer initialized")
        
        # Test loading models (this might take a while)
        print("Loading models...")
        summarizer.load_models()
        print("✅ Models loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ App initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Video Summarizer Deployment Compatibility\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check dependencies.")
        sys.exit(1)
    
    # Test app initialization
    if not test_app_initialization():
        print("\n❌ App initialization failed. Check configuration.")
        sys.exit(1)
    
    print("\n🎉 All tests passed! App is ready for deployment.")
