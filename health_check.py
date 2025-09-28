#!/usr/bin/env python3
"""
Health check script for the Review Search application.
Returns 0 if healthy, 1 if unhealthy.
"""
import sys
import requests
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_streamlit_health(host="localhost", port=8501, timeout=10):
    """Check if Streamlit app is responding."""
    try:
        # Try to connect to the app
        url = f"http://{host}:{port}"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            print(f"✅ Streamlit app responding at {url}")
            return True
        else:
            print(f"❌ Streamlit app returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Streamlit app at {host}:{port}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Timeout connecting to Streamlit app")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def check_data_files():
    """Check if critical data files exist."""
    try:
        from config import config
        
        critical_files = [
            config.PRODUCT_EMB_PATH,
            config.PRODUCT_META_PATH
        ]
        
        for file_path in critical_files:
            if not file_path.exists():
                print(f"❌ Critical file missing: {file_path}")
                return False
        
        print("✅ All critical data files present")
        return True
        
    except Exception as e:
        print(f"❌ Data file check failed: {e}")
        return False

def check_imports():
    """Check if critical imports are working."""
    try:
        import numpy as np
        import pandas as pd
        import streamlit as st
        print("✅ Core dependencies available")
        
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ SentenceTransformers available")
        except ImportError:
            print("⚠️ SentenceTransformers not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Import check failed: {e}")
        return False

def main():
    """Run all health checks."""
    print("🔍 Running health checks...")
    
    checks = [
        ("Data files", check_data_files),
        ("Imports", check_imports),
        ("Streamlit app", lambda: check_streamlit_health()),
    ]
    
    all_healthy = True
    
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            if not check_func():
                all_healthy = False
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            all_healthy = False
    
    if all_healthy:
        print("\n🎉 All health checks passed!")
        return 0
    else:
        print("\n❌ Some health checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())