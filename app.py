#!/usr/bin/env python3
"""
HF Spaces entry point for the Review Search Copilot.
This file serves as the main entry point for Hugging Face Spaces deployment.
Updated: 2025-09-28 - Force rebuild
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the main Streamlit app
from app.app_product_search import *