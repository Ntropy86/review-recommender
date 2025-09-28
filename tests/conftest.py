"""
Shared pytest fixtures and configuration.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

@pytest.fixture
def sample_product_data():
    """Create sample product data for testing."""
    return pd.DataFrame({
        'sku': ['SKU001', 'SKU002', 'SKU003'],
        'n_reviews': [25, 50, 100],
        'avg_stars': [4.2, 3.8, 4.5],
        'agg_text': [
            'wireless headphones bluetooth noise cancelling',
            'yellow cat socks soft comfortable cute',
            'gaming keyboard mechanical rgb backlight'
        ]
    })

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    # 3 products x 384 dimensions (typical for small embedding models)
    return np.random.rand(3, 384).astype(np.float32)

@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "best wireless headphones for music"

@pytest.fixture
def sample_query_vector():
    """Sample query vector for testing."""
    return np.random.rand(384).astype(np.float32)

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        yield data_dir

@pytest.fixture
def mock_config(temp_data_dir):
    """Mock configuration for testing."""
    class MockConfig:
        DATA_DIR = temp_data_dir
        PRODUCT_EMB_PATH = temp_data_dir / "product_emb.npy"
        PRODUCT_META_PATH = temp_data_dir / "product_emb_meta.parquet"
        REVIEWS_EMB_PATH = temp_data_dir / "reviews_with_embeddings.parquet"
        BM25_PATH = temp_data_dir / "product_bm25.pkl"
        
        EMB_MODEL = "BAAI/bge-small-en-v1.5"
        RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        DEFAULT_K = 10
        DEFAULT_RERANK_K = 50
        DEFAULT_MIN_REVIEWS = 8
        DEFAULT_W_DENSE = 0.55
        DEFAULT_W_BM25 = 0.20
        DEFAULT_W_RERANK = 0.20
        DEFAULT_W_PRIOR = 0.20
        DEFAULT_W_BEST = 0.10
        DEFAULT_GATE_PENALTY = 0.5
        
        MAX_REVIEWS_SCAN = 100000
        ENABLE_BM25 = True
        ENABLE_RERANKING = True
        ENABLE_SNIPPETS = True
        
        LOG_LEVEL = "DEBUG"
        ENVIRONMENT = "test"
        
        @classmethod
        def is_production(cls):
            return False
            
        @classmethod
        def is_development(cls):
            return False
    
    return MockConfig()

@pytest.fixture
def sample_bm25_data():
    """Sample BM25 data for testing."""
    return {
        'corpus': [
            ['wireless', 'headphones', 'bluetooth'],
            ['yellow', 'cat', 'socks', 'soft'],
            ['gaming', 'keyboard', 'mechanical']
        ],
        'skus': ['SKU001', 'SKU002', 'SKU003']
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    os.environ['ENVIRONMENT'] = 'test'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    yield
    # Cleanup
    for key in ['ENVIRONMENT', 'LOG_LEVEL']:
        os.environ.pop(key, None)