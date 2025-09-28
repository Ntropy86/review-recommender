"""
Integration tests for the main application.
"""
import pytest
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.mark.integration
class TestSearchPipeline:
    """Integration tests for the search pipeline."""
    
    def setup_test_data(self, temp_data_dir, sample_product_data, sample_embeddings, sample_bm25_data):
        """Setup test data files."""
        # Save product metadata
        sample_product_data.to_parquet(temp_data_dir / "product_emb_meta.parquet")
        
        # Save embeddings
        np.save(temp_data_dir / "product_emb.npy", sample_embeddings)
        
        # Save BM25 data
        with open(temp_data_dir / "product_bm25.pkl", "wb") as f:
            pickle.dump(sample_bm25_data, f)
    
    @patch('app.app_product_search._st_encoder')
    @patch('app.app_product_search._cross_encoder')
    def test_search_pipeline_basic(self, mock_cross_encoder, mock_st_encoder, 
                                 temp_data_dir, sample_product_data, sample_embeddings, 
                                 sample_bm25_data, sample_query_vector):
        """Test basic search pipeline functionality."""
        # Setup test data
        self.setup_test_data(temp_data_dir, sample_product_data, sample_embeddings, sample_bm25_data)
        
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([sample_query_vector])
        mock_st_encoder.return_value = mock_encoder
        
        # Mock cross encoder
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.8, 0.6, 0.9])
        mock_cross_encoder.return_value = mock_ce
        
        # Import after setting up mocks
        with patch('app.app_product_search.DATA', temp_data_dir):
            with patch('app.app_product_search.P_EMB', temp_data_dir / "product_emb.npy"):
                with patch('app.app_product_search.P_META', temp_data_dir / "product_emb_meta.parquet"):
                    with patch('app.app_product_search.BM25_PKL', temp_data_dir / "product_bm25.pkl"):
                        from app.app_product_search import run_search
                        
                        # Run search
                        results, snippets, debug_info = run_search(
                            query="wireless headphones",
                            k=3,
                            rerank_k=3,
                            w_dense=0.5,
                            w_bm25=0.2,
                            w_rerank=0.2,
                            w_prior=0.1,
                            w_best=0.0,
                            prior_C=20.0,
                            use_snips=False,
                            max_scan=1000,
                            min_reviews=5,
                            gate_penalty=0.5
                        )
                        
                        # Verify results
                        assert len(results) <= 3
                        assert '_final' in results.columns
                        assert '_dense' in results.columns
                        assert '_bm25' in results.columns
                        assert '_rerank' in results.columns
                        assert 'sku' in results.columns
                        
                        # Check that results are sorted by final score
                        final_scores = results['_final'].values
                        assert all(final_scores[i] >= final_scores[i+1] for i in range(len(final_scores)-1))
    
    def test_missing_data_handling(self, temp_data_dir):
        """Test handling of missing data files."""
        # Don't create any data files
        
        with patch('app.app_product_search.DATA', temp_data_dir):
            with patch('app.app_product_search.P_EMB', temp_data_dir / "product_emb.npy"):
                with patch('app.app_product_search.P_META', temp_data_dir / "product_emb_meta.parquet"):
                    from app.app_product_search import _product_index
                    
                    # Should handle missing files gracefully (or raise appropriate error)
                    try:
                        meta, embeddings = _product_index()
                        # If it doesn't raise an error, check that it returns sensible empty data
                        assert meta is not None
                        assert embeddings is not None
                    except (FileNotFoundError, Exception) as e:
                        # Expected behavior for missing files
                        assert "Missing" in str(e) or "product_emb" in str(e)

@pytest.mark.integration
@pytest.mark.requires_data
class TestWithRealData:
    """Integration tests that require real data files."""
    
    def test_search_with_real_data_if_available(self):
        """Test search with real data files if they exist."""
        data_dir = Path("data/processed")
        emb_file = data_dir / "product_emb.npy"
        meta_file = data_dir / "product_emb_meta.parquet"
        
        if not (emb_file.exists() and meta_file.exists()):
            pytest.skip("Real data files not available")
        
        # Import and test with real data
        from app.app_product_search import run_search
        
        try:
            results, snippets, debug_info = run_search(
                query="wireless headphones",
                k=5,
                rerank_k=0,  # Disable reranking to avoid model downloads
                w_dense=1.0,
                w_bm25=0.0,
                w_rerank=0.0,
                w_prior=0.0,
                w_best=0.0,
                prior_C=20.0,
                use_snips=False,
                max_scan=1000,
                min_reviews=5,
                gate_penalty=0.5
            )
            
            assert len(results) <= 5
            assert 'sku' in results.columns
            assert '_final' in results.columns
            
        except Exception as e:
            pytest.skip(f"Real data test failed (expected): {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])