#!/usr/bin/env python3
"""
Simple test runner for basic functionality testing.
Can be used when pytest is not available.
"""
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_config_tests():
    """Run basic configuration tests."""
    print("Testing configuration...")
    
    try:
        from config import Config, config
        
        # Test basic instantiation
        test_config = Config()
        assert test_config.ENVIRONMENT in ["development", "production", "test"]
        assert test_config.APP_PORT > 0
        assert test_config.DEFAULT_K > 0
        print("✅ Config instantiation: PASS")
        
        # Test global config
        assert config is not None
        print("✅ Global config: PASS")
        
        # Test boolean parsing
        import os
        from config import Config
        old_value = os.environ.get('ENABLE_BM25')
        os.environ['ENABLE_BM25'] = 'false'
        # Create a fresh config instance
        fresh_config = Config()
        assert fresh_config.ENABLE_BM25 is False
        # Restore original value
        if old_value is not None:
            os.environ['ENABLE_BM25'] = old_value
        else:
            os.environ.pop('ENABLE_BM25', None)
        print("✅ Boolean parsing: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Config tests failed: {e}")
        traceback.print_exc()
        return False

def run_utils_tests():
    """Run basic utility function tests."""
    print("\nTesting utilities...")
    
    try:
        import numpy as np
        from utils import (
            l2_normalize, minmax_normalize, tokenize_query, 
            build_gate_groups, calculate_gate_factor
        )
        
        # Test L2 normalization
        x = np.array([[3.0, 4.0]])
        normalized = l2_normalize(x, axis=1)
        norm = np.linalg.norm(normalized[0])
        assert abs(norm - 1.0) < 1e-5
        print("✅ L2 normalization: PASS")
        
        # Test tokenization
        tokens = tokenize_query("best wireless headphones for music")
        assert "best" in tokens
        assert "wireless" in tokens
        assert "for" not in tokens  # stop word removed
        print("✅ Query tokenization: PASS")
        
        # Test gate groups
        groups = build_gate_groups("yellow cat socks")
        assert len(groups) > 0
        yellow_found = any("yellow" in group for group in groups)
        assert yellow_found
        print("✅ Gate group building: PASS")
        
        # Test gate factor
        factor, hits, total = calculate_gate_factor(
            "yellow cat socks", 
            [{"yellow"}, {"cat"}, {"sock", "socks"}]
        )
        assert hits == 3
        assert factor == 1.0
        print("✅ Gate factor calculation: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils tests failed: {e}")
        traceback.print_exc()
        return False

def run_import_tests():
    """Test that critical imports work."""
    print("\nTesting imports...")
    
    try:
        # Test core dependencies
        import numpy as np
        import pandas as pd
        import streamlit as st
        print("✅ Core dependencies: PASS")
        
        # Test ML dependencies
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ Sentence transformers: PASS")
        except ImportError:
            print("⚠️ Sentence transformers: SKIP (not installed)")
        
        try:
            from rank_bm25 import BM25Okapi
            print("✅ BM25: PASS")
        except ImportError:
            print("❌ BM25: FAIL (not installed)")
            return False
        
        # Test app imports
        try:
            sys.path.append(str(Path(__file__).parent.parent / "app"))
            from app_product_search import _l2norm, _minmax, _tokenize
            print("✅ App functions: PASS")
        except ImportError as e:
            print(f"⚠️ App functions: SKIP ({e})")
        
        return True
        
    except Exception as e:
        print(f"❌ Import tests failed: {e}")
        traceback.print_exc()
        return False

def run_data_validation():
    """Validate data files if they exist."""
    print("\nValidating data files...")
    
    try:
        from config import config
        
        # Check if critical files exist
        critical_files = [
            config.PRODUCT_EMB_PATH,
            config.PRODUCT_META_PATH
        ]
        
        all_exist = True
        for file_path in critical_files:
            if file_path.exists():
                print(f"✅ {file_path.name}: EXISTS")
            else:
                print(f"⚠️ {file_path.name}: MISSING")
                all_exist = False
        
        # Check optional files
        optional_files = [
            config.REVIEWS_EMB_PATH,
            config.BM25_PATH
        ]
        
        for file_path in optional_files:
            if file_path.exists():
                print(f"✅ {file_path.name}: EXISTS (optional)")
            else:
                print(f"ℹ️ {file_path.name}: MISSING (optional)")
        
        if all_exist:
            print("✅ All critical data files present")
        else:
            print("⚠️ Some critical data files missing (run data preprocessing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("🔍 Running basic functionality tests...\n")
    
    results = []
    results.append(run_config_tests())
    results.append(run_utils_tests())
    results.append(run_import_tests())
    results.append(run_data_validation())
    
    print(f"\n📊 Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())