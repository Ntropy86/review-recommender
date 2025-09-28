"""
Unit tests for utility functions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    l2_normalize, minmax_normalize, tokenize_query, build_gate_groups,
    calculate_gate_factor, bayesian_prior, cosine_similarity_search,
    trust_score_from_reviews
)

class TestNormalizationFunctions:
    """Test normalization utility functions."""
    
    def test_l2_normalize_basic(self):
        """Test basic L2 normalization."""
        x = np.array([[3.0, 4.0], [1.0, 0.0]])
        normalized = l2_normalize(x, axis=1)
        
        # Check that each row has unit norm
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-5)
    
    def test_l2_normalize_zero_vector(self):
        """Test L2 normalization with zero vector."""
        x = np.array([[0.0, 0.0], [3.0, 4.0]])
        normalized = l2_normalize(x, axis=1)
        
        # Zero vector should remain zero (due to eps)
        assert normalized[0, 0] == 0.0
        assert normalized[0, 1] == 0.0
        
        # Non-zero vector should be normalized
        expected_norm = np.linalg.norm(normalized[1, :])
        assert abs(expected_norm - 1.0) < 1e-5
    
    def test_minmax_normalize_basic(self):
        """Test basic min-max normalization."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = minmax_normalize(x)
        
        assert normalized[0] == 0.0  # min becomes 0
        assert normalized[-1] == 1.0  # max becomes 1
        assert 0.0 <= np.min(normalized) <= np.max(normalized) <= 1.0
    
    def test_minmax_normalize_constant_array(self):
        """Test min-max normalization with constant array."""
        x = np.array([3.0, 3.0, 3.0, 3.0])
        normalized = minmax_normalize(x)
        
        # All values should become 0 when range is zero
        np.testing.assert_array_equal(normalized, [0.0, 0.0, 0.0, 0.0])
    
    def test_minmax_normalize_empty_array(self):
        """Test min-max normalization with empty array."""
        x = np.array([])
        normalized = minmax_normalize(x)
        
        assert len(normalized) == 0

class TestQueryProcessing:
    """Test query processing functions."""
    
    def test_tokenize_query_basic(self):
        """Test basic query tokenization."""
        query = "best wireless headphones for music"
        tokens = tokenize_query(query)
        
        assert "best" in tokens
        assert "wireless" in tokens
        assert "headphones" in tokens
        assert "music" in tokens
        # Stop words should be removed
        assert "for" not in tokens
    
    def test_tokenize_query_with_punctuation(self):
        """Test tokenization with punctuation."""
        query = "noise-cancelling headphones, really good!"
        tokens = tokenize_query(query)
        
        assert "noise" in tokens
        assert "cancelling" in tokens
        assert "headphones" in tokens
        assert "really" in tokens
        assert "good" in tokens
    
    def test_build_gate_groups_with_colors(self):
        """Test gate group building with color mentions."""
        query = "yellow cat socks"
        groups = build_gate_groups(query)
        
        # Should have yellow color group
        yellow_group = {"yellow", "mustard", "lemon", "gold", "golden"}
        assert yellow_group in groups
        
        # Should have cat synonym group
        cat_group = {"cat", "cats", "kitten", "kittens", "kitty"}
        assert cat_group in groups
        
        # Should have sock synonym group
        sock_group = {"sock", "socks"}
        assert sock_group in groups
    
    def test_build_gate_groups_no_matches(self):
        """Test gate group building with no synonym matches."""
        query = "random unique product"
        groups = build_gate_groups(query)
        
        # Should have individual word groups for longer words
        assert {"random"} in groups
        assert {"unique"} in groups
        assert {"product"} in groups
    
    def test_calculate_gate_factor_full_match(self):
        """Test gate factor calculation with full match."""
        text = "yellow cat socks soft comfortable"
        groups = [
            {"yellow", "mustard", "gold"},
            {"cat", "cats", "kitten"},
            {"sock", "socks"}
        ]
        
        factor, hits, total = calculate_gate_factor(text, groups, penalty=0.5)
        
        assert hits == 3  # All groups match
        assert total == 3
        assert factor == 1.0  # No penalty applied
    
    def test_calculate_gate_factor_partial_match(self):
        """Test gate factor calculation with partial match."""
        text = "yellow comfortable shoes"  # Missing cat and sock mentions
        groups = [
            {"yellow", "mustard", "gold"},
            {"cat", "cats", "kitten"},
            {"sock", "socks"}
        ]
        
        factor, hits, total = calculate_gate_factor(text, groups, penalty=0.5)
        
        assert hits == 1  # Only yellow group matches
        assert total == 3
        assert factor == 0.25  # 0.5^2 penalty for 2 missing groups

class TestStatisticalFunctions:
    """Test statistical utility functions."""
    
    def test_bayesian_prior_basic(self):
        """Test basic Bayesian prior calculation."""
        avg_ratings = np.array([4.0, 3.0, 5.0])
        review_counts = np.array([10, 100, 5])
        
        priors = bayesian_prior(avg_ratings, review_counts, prior_strength=20.0, global_mean=4.0)
        
        # Product with more reviews should be closer to its actual rating
        # Product with fewer reviews should be pulled toward global mean
        assert priors[1] > priors[0]  # 100 reviews vs 10 reviews, both below global mean
        assert priors[2] < 5.0  # High rating with few reviews pulled down
    
    def test_trust_score_from_reviews(self):
        """Test trust score calculation."""
        review_counts = np.array([0, 5, 10, 50, 100])
        trust_scores = trust_score_from_reviews(review_counts, min_reviews=8)
        
        # Trust should increase with review count
        assert trust_scores[0] < trust_scores[1] < trust_scores[2] < trust_scores[3]
        # But should saturate at high counts
        assert trust_scores[4] <= 1.0
        assert trust_scores[3] > 0.8  # Should be high for 50+ reviews

class TestSimilaritySearch:
    """Test similarity search functions."""
    
    def test_cosine_similarity_search_basic(self):
        """Test basic cosine similarity search."""
        # Create simple 2D embeddings
        embeddings = np.array([
            [1.0, 0.0],  # Orthogonal to query
            [0.0, 1.0],  # Parallel to query
            [1.0, 1.0],  # 45 degrees to query
        ], dtype=np.float32)
        
        query_vector = np.array([0.0, 1.0], dtype=np.float32)
        
        indices, similarities = cosine_similarity_search(query_vector, embeddings, top_k=2)
        
        # Should return top 2 most similar
        assert len(indices) == 2
        assert len(similarities) == 2
        
        # [0, 1] should be most similar to [0, 1]
        assert indices[0] == 1
        assert similarities[0] == 1.0
    
    def test_cosine_similarity_search_top_k_larger_than_data(self):
        """Test similarity search when top_k is larger than available data."""
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query_vector = np.array([0.0, 1.0], dtype=np.float32)
        
        indices, similarities = cosine_similarity_search(query_vector, embeddings, top_k=10)
        
        # Should return all available data
        assert len(indices) == 2
        assert len(similarities) == 2

if __name__ == "__main__":
    pytest.main([__file__])