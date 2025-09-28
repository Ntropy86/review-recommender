"""
Unit tests for performance evaluation metrics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.performance_metrics import (
    dcg_at_k, ndcg_at_k, mrr_score, recall_at_k, 
    precision_at_k, IRMetrics
)


class TestPerformanceMetrics:
    """Test cases for IR metrics implementation."""
    
    def test_dcg_at_k(self):
        """Test DCG@k calculation."""
        # Test case from standard IR textbook
        relevance = [3, 2, 3, 0, 1, 2]  # Relevance scores
        
        # DCG@3 = 3/log2(2) + 2/log2(3) + 3/log2(4) = 3 + 1.26 + 1.5 = 5.76 (approx)
        dcg = dcg_at_k(relevance, 3)
        expected = 3.0 + 2.0/np.log2(3) + 3.0/np.log2(4)
        assert abs(dcg - expected) < 0.01
        
        # DCG@0 should return 0
        assert dcg_at_k(relevance, 0) == 0.0
        
        # DCG@1 should just be first relevance score
        assert dcg_at_k(relevance, 1) == 3.0
    
    def test_ndcg_at_k(self):
        """Test nDCG@k calculation."""
        relevance = [3, 2, 3, 0, 1, 2]
        ideal = sorted(relevance, reverse=True)  # [3, 3, 2, 2, 1, 0]
        
        # nDCG should be between 0 and 1
        ndcg = ndcg_at_k(relevance, ideal, 3)
        assert 0.0 <= ndcg <= 1.0
        
        # Perfect ranking should give nDCG = 1
        perfect_ndcg = ndcg_at_k(ideal, ideal, 3)
        assert abs(perfect_ndcg - 1.0) < 0.01
        
        # No relevant items should give nDCG = 0
        no_rel = [0, 0, 0]
        assert ndcg_at_k(no_rel, no_rel, 3) == 0.0
    
    def test_mrr_score(self):
        """Test MRR calculation."""
        # Test case 1: First result is relevant
        ranked_results = [["item1", "item2", "item3"]]
        relevant_items = [{"item1", "item4"}]
        
        mrr = mrr_score(ranked_results, relevant_items)
        assert mrr == 1.0  # First position = 1/1 = 1.0
        
        # Test case 2: Second result is relevant  
        ranked_results = [["item1", "item2", "item3"]]
        relevant_items = [{"item2", "item4"}]
        
        mrr = mrr_score(ranked_results, relevant_items)
        assert mrr == 0.5  # Second position = 1/2 = 0.5
        
        # Test case 3: No relevant results
        ranked_results = [["item1", "item2", "item3"]]
        relevant_items = [{"item4", "item5"}]
        
        mrr = mrr_score(ranked_results, relevant_items)
        assert mrr == 0.0
        
        # Test case 4: Multiple queries
        ranked_results = [
            ["item1", "item2", "item3"],  # Relevant at position 2
            ["item4", "item5", "item6"]   # Relevant at position 1
        ]
        relevant_items = [
            {"item2"},
            {"item4"}
        ]
        
        mrr = mrr_score(ranked_results, relevant_items)
        expected = (0.5 + 1.0) / 2  # (1/2 + 1/1) / 2 = 0.75
        assert abs(mrr - expected) < 0.01
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        # Test case: 2 out of 3 relevant items retrieved in top 5
        ranked_results = [["item1", "item2", "item3", "item4", "item5"]]
        relevant_items = [{"item2", "item4", "item6"}]  # 3 relevant items total
        
        recall = recall_at_k(ranked_results, relevant_items, 5)
        expected = 2.0 / 3.0  # Found 2 out of 3 relevant items
        assert abs(recall - expected) < 0.01
        
        # Test case: Perfect recall
        ranked_results = [["item1", "item2", "item3"]]
        relevant_items = [{"item1", "item2"}]
        
        recall = recall_at_k(ranked_results, relevant_items, 3)
        assert recall == 1.0
        
        # Test case: No relevant items
        ranked_results = [["item1", "item2", "item3"]]
        relevant_items = [set()]  # Empty set
        
        recall = recall_at_k(ranked_results, relevant_items, 3)
        assert recall == 0.0
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        # Test case: 2 out of 5 retrieved items are relevant
        ranked_results = [["item1", "item2", "item3", "item4", "item5"]]
        relevant_items = [{"item2", "item4", "item6"}]
        
        precision = precision_at_k(ranked_results, relevant_items, 5)
        expected = 2.0 / 5.0  # 2 relevant out of 5 retrieved
        assert abs(precision - expected) < 0.01
        
        # Test case: Perfect precision
        ranked_results = [["item1", "item2"]]
        relevant_items = [{"item1", "item2", "item3"}]
        
        precision = precision_at_k(ranked_results, relevant_items, 2)
        assert precision == 1.0
    
    def test_ir_metrics_class(self):
        """Test IRMetrics class functionality."""
        evaluator = IRMetrics()
        
        # Test single query evaluation
        retrieved = ["item1", "item2", "item3", "item4", "item5"]
        relevant = {"item2", "item4", "item6"}
        
        metrics = evaluator.evaluate_query("q1", retrieved, relevant)
        
        # Check that all expected metrics are present
        expected_metrics = ['ndcg@5', 'ndcg@10', 'mrr', 'recall@10', 'recall@20', 'precision@5', 'precision@10']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0
        
        # Test aggregation
        retrieved2 = ["item3", "item4", "item5", "item6", "item7"]
        relevant2 = {"item4", "item6", "item8"}
        
        evaluator.evaluate_query("q2", retrieved2, relevant2)
        
        aggregated = evaluator.aggregate_metrics()
        assert len(aggregated) == len(expected_metrics)
        
        # Test detailed report
        report = evaluator.detailed_report()
        assert len(report) == 2  # Two queries
        assert "q1" in report.index
        assert "q2" in report.index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])