"""
Configuration management for the Review Search application.
Handles environment variables, defaults, and validation.
"""
import os
from pathlib import Path
from typing import Optional
import logging

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue with system env vars
    pass

class Config:
    """Application configuration class."""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Model Configuration
    EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Application Configuration
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8501"))
    APP_TITLE = os.getenv("APP_TITLE", "Review Search Copilot")
    
    # Data Paths - HF Datasets
    HF_DATASET = os.getenv("HF_DATASET", "ntropy86/AmazonReviewsCompiled")
    DATA_DIR = Path(os.getenv("DATA_DIR", "hf://datasets/ntropy86/AmazonReviewsCompiled"))
    PRODUCT_EMB_FILE = os.getenv("PRODUCT_EMB_FILE", "product_emb.npy")
    PRODUCT_META_FILE = os.getenv("PRODUCT_META_FILE", "product_emb_meta.parquet")
    REVIEWS_EMB_FILE = os.getenv("REVIEWS_EMB_FILE", "reviews_with_embeddings.parquet")
    BM25_FILE = os.getenv("BM25_FILE", "product_bm25.pkl")
    
    # Derived paths
    PRODUCT_EMB_PATH = DATA_DIR / PRODUCT_EMB_FILE
    PRODUCT_META_PATH = DATA_DIR / PRODUCT_META_FILE
    REVIEWS_EMB_PATH = DATA_DIR / REVIEWS_EMB_FILE
    BM25_PATH = DATA_DIR / BM25_FILE
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    
    # Performance Settings
    MAX_REVIEWS_SCAN = int(os.getenv("MAX_REVIEWS_SCAN", "300000"))
    DEFAULT_POOL_SIZE = int(os.getenv("DEFAULT_POOL_SIZE", "150"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Feature Flags
    ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"
    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    ENABLE_SNIPPETS = os.getenv("ENABLE_SNIPPETS", "true").lower() == "true"
    ENABLE_METRICS_TAB = os.getenv("ENABLE_METRICS_TAB", "true").lower() == "true"
    
    # Search Defaults
    DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))
    DEFAULT_RERANK_K = int(os.getenv("DEFAULT_RERANK_K", "50"))
    DEFAULT_MIN_REVIEWS = int(os.getenv("DEFAULT_MIN_REVIEWS", "8"))
    DEFAULT_W_DENSE = float(os.getenv("DEFAULT_W_DENSE", "0.55"))
    DEFAULT_W_BM25 = float(os.getenv("DEFAULT_W_BM25", "0.20"))
    DEFAULT_W_RERANK = float(os.getenv("DEFAULT_W_RERANK", "0.20"))
    DEFAULT_W_PRIOR = float(os.getenv("DEFAULT_W_PRIOR", "0.20"))
    DEFAULT_W_BEST = float(os.getenv("DEFAULT_W_BEST", "0.10"))
    DEFAULT_GATE_PENALTY = float(os.getenv("DEFAULT_GATE_PENALTY", "0.5"))
    
    # Security Settings
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",") if os.getenv("ALLOWED_HOSTS") else []
    HTTPS_ONLY = os.getenv("HTTPS_ONLY", "false").lower() == "true"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration and create necessary directories."""
        # Create logs directory
        log_dir = Path(cls.LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # For HF datasets, we don't need to validate file existence locally
        # The datasets library will handle loading from HF
        if not cls.HF_DATASET:
            raise ValueError("HF_DATASET environment variable must be set")
        
        # Test HF dataset access (optional, only if datasets library is available)
        try:
            import datasets
            # Quick test to see if dataset is accessible
            datasets.load_dataset(cls.HF_DATASET, split="train", streaming=True).take(1)
        except ImportError:
            # datasets not available, skip validation
            pass
        except Exception as e:
            logging.warning(f"Could not validate HF dataset access: {e}")
            # Continue anyway - might work at runtime
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup application logging."""
        # Create logs directory
        log_dir = Path(cls.LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return cls.ENVIRONMENT.lower() == "production"
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment."""
        return cls.ENVIRONMENT.lower() == "development"

# Global config instance
config = Config()