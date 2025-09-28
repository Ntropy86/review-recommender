"""
Unit tests for configuration module.
"""
import pytest
import os
from pathlib import Path
import tempfile
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config, config

class TestConfig:
    """Test configuration class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        test_config = Config()
        assert test_config.ENVIRONMENT == "development"
        assert test_config.EMB_MODEL == "BAAI/bge-small-en-v1.5"
        assert test_config.RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert test_config.APP_PORT == 8501
        assert test_config.DEFAULT_K == 10
    
    def test_environment_variables(self):
        """Test that environment variables override defaults."""
        os.environ['ENVIRONMENT'] = 'test'
        os.environ['APP_PORT'] = '9000'
        os.environ['DEFAULT_K'] = '20'
        
        test_config = Config()
        assert test_config.ENVIRONMENT == 'test'
        assert test_config.APP_PORT == 9000
        assert test_config.DEFAULT_K == 20
        
        # Cleanup
        del os.environ['ENVIRONMENT']
        del os.environ['APP_PORT']
        del os.environ['DEFAULT_K']
    
    def test_boolean_parsing(self):
        """Test that boolean environment variables are parsed correctly."""
        os.environ['ENABLE_BM25'] = 'false'
        os.environ['ENABLE_RERANKING'] = 'True'
        os.environ['HTTPS_ONLY'] = 'FALSE'
        
        test_config = Config()
        assert test_config.ENABLE_BM25 is False
        assert test_config.ENABLE_RERANKING is True
        assert test_config.HTTPS_ONLY is False
        
        # Cleanup
        del os.environ['ENABLE_BM25']
        del os.environ['ENABLE_RERANKING']
        del os.environ['HTTPS_ONLY']
    
    def test_path_construction(self):
        """Test that paths are constructed correctly."""
        test_config = Config()
        expected_emb_path = test_config.DATA_DIR / test_config.PRODUCT_EMB_FILE
        assert test_config.PRODUCT_EMB_PATH == expected_emb_path
    
    def test_is_production(self):
        """Test production environment detection."""
        os.environ['ENVIRONMENT'] = 'production'
        test_config = Config()
        assert test_config.is_production() is True
        assert test_config.is_development() is False
        
        del os.environ['ENVIRONMENT']
    
    def test_is_development(self):
        """Test development environment detection."""
        os.environ['ENVIRONMENT'] = 'development'
        test_config = Config()
        assert test_config.is_development() is True
        assert test_config.is_production() is False
        
        del os.environ['ENVIRONMENT']
    
    def test_validate_creates_directories(self):
        """Test that validate method creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['DATA_DIR'] = str(tmpdir) + '/data/processed'
            os.environ['LOG_FILE'] = str(tmpdir) + '/logs/app.log'
            
            test_config = Config()
            test_config.validate()
            
            assert Path(test_config.DATA_DIR).exists()
            assert Path(test_config.LOG_FILE).parent.exists()
            
            del os.environ['DATA_DIR']
            del os.environ['LOG_FILE']
    
    def test_validate_fails_on_missing_files_in_production(self):
        """Test that validate fails when critical files are missing in production."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['ENVIRONMENT'] = 'production'
            os.environ['DATA_DIR'] = str(tmpdir) + '/data/processed'
            
            test_config = Config()
            
            with pytest.raises(FileNotFoundError):
                test_config.validate()
            
            del os.environ['ENVIRONMENT']
            del os.environ['DATA_DIR']

class TestGlobalConfig:
    """Test global config instance."""
    
    def test_global_config_exists(self):
        """Test that global config instance exists."""
        assert config is not None
        assert isinstance(config, Config)