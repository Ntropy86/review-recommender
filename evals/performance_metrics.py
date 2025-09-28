"""
Performance evaluation metrics for information retrieval systems.

This module implements standard IR metrics:
- nDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)  
- Recall@K
- Precision@K
"""

import numpy as np
from typing import List, Dict, Tuple, Set
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at rank k.
    
    Args:
        relevance_scores: List of relevance scores (higher = more relevant)
        k: Cutoff rank
        
    Returns:
        DCG@k score
    """
    if k <= 0:
        return 0.0
    
    # Take only first k items
    rel = np.array(relevance_scores[:k])
    
    # DCG formula: sum(rel_i / log2(i + 1)) for i in [0, k-1]
    ranks = np.arange(1, len(rel) + 1)
    dcg = np.sum(rel / np.log2(ranks + 1))
    
    return float(dcg)


def ndcg_at_k(relevance_scores: List[float], ideal_scores: List[float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank k.
    
    Args:
        relevance_scores: Relevance scores in retrieved order
        ideal_scores: Relevance scores in ideal (perfect) order  
        k: Cutoff rank
        
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def mrr_score(ranked_results: List[List[str]], relevant_items: List[Set[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        ranked_results: List of ranked item lists for each query
        relevant_items: List of relevant item sets for each query
        
    Returns:
        MRR score
    """
    if len(ranked_results) != len(relevant_items):
        raise ValueError("Number of queries must match for results and relevance")
    
    reciprocal_ranks = []
    
    for results, relevant in zip(ranked_results, relevant_items):
        reciprocal_rank = 0.0
        
        for rank, item in enumerate(results, 1):
            if item in relevant:
                reciprocal_rank = 1.0 / rank
                break
        
        reciprocal_ranks.append(reciprocal_rank)
    
    return np.mean(reciprocal_ranks)


def recall_at_k(ranked_results: List[List[str]], relevant_items: List[Set[str]], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        ranked_results: List of ranked item lists for each query
        relevant_items: List of relevant item sets for each query  
        k: Cutoff rank
        
    Returns:
        Recall@K score
    """
    if len(ranked_results) != len(relevant_items):
        raise ValueError("Number of queries must match for results and relevance")
    
    recalls = []
    
    for results, relevant in zip(ranked_results, relevant_items):
        if len(relevant) == 0:
            recalls.append(0.0)
            continue
        
        # Take top-k results
        top_k = results[:k]
        retrieved_relevant = set(top_k) & relevant
        
        recall = len(retrieved_relevant) / len(relevant)
        recalls.append(recall)
    
    return np.mean(recalls)


def precision_at_k(ranked_results: List[List[str]], relevant_items: List[Set[str]], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        ranked_results: List of ranked item lists for each query
        relevant_items: List of relevant item sets for each query
        k: Cutoff rank
        
    Returns:
        Precision@K score
    """
    if len(ranked_results) != len(relevant_items):
        raise ValueError("Number of queries must match for results and relevance")
    
    precisions = []
    
    for results, relevant in zip(ranked_results, relevant_items):
        # Take top-k results
        top_k = results[:k]
        if len(top_k) == 0:
            precisions.append(0.0)
            continue
        
        retrieved_relevant = set(top_k) & relevant
        precision = len(retrieved_relevant) / len(top_k)
        precisions.append(precision)
    
    return np.mean(precisions)


class IRMetrics:
    """Information Retrieval metrics calculator."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_query(self, 
                      query_id: str,
                      retrieved_items: List[str], 
                      relevant_items: Set[str],
                      relevance_scores: Dict[str, float] = None) -> Dict[str, float]:
        """
        Evaluate a single query.
        
        Args:
            query_id: Unique query identifier
            retrieved_items: List of retrieved items in rank order
            relevant_items: Set of relevant items for this query
            relevance_scores: Optional relevance scores for items (0-4 scale)
            
        Returns:
            Dictionary of metric scores
        """
        if relevance_scores is None:
            # Binary relevance: relevant=1, not relevant=0
            relevance_scores = {item: 1.0 for item in relevant_items}
        
        # Calculate relevance scores for retrieved items
        retrieved_relevance = [relevance_scores.get(item, 0.0) for item in retrieved_items]
        
        # Calculate ideal relevance ordering
        all_scores = list(relevance_scores.values())
        ideal_relevance = sorted(all_scores, reverse=True)
        
        metrics = {
            'ndcg@5': ndcg_at_k(retrieved_relevance, ideal_relevance, 5),
            'ndcg@10': ndcg_at_k(retrieved_relevance, ideal_relevance, 10),
            'mrr': 1.0 / (next((i+1 for i, item in enumerate(retrieved_items) if item in relevant_items), float('inf'))),
            'recall@10': recall_at_k([retrieved_items], [relevant_items], 10),
            'recall@20': recall_at_k([retrieved_items], [relevant_items], 20),
            'precision@5': precision_at_k([retrieved_items], [relevant_items], 5),
            'precision@10': precision_at_k([retrieved_items], [relevant_items], 10),
        }
        
        # Handle infinite MRR (no relevant items found)
        if metrics['mrr'] == 0.0:
            metrics['mrr'] = 0.0
        
        self.results[query_id] = metrics
        return metrics
    
    def aggregate_metrics(self) -> Dict[str, float]:
        """
        Aggregate metrics across all evaluated queries.
        
        Returns:
            Dictionary of mean metric scores
        """
        if not self.results:
            return {}
        
        # Collect all metric values
        all_metrics = {}
        for metric_name in self.results[list(self.results.keys())[0]].keys():
            values = [result[metric_name] for result in self.results.values()]
            all_metrics[metric_name] = np.mean(values)
        
        return all_metrics
    
    def detailed_report(self) -> pd.DataFrame:
        """
        Generate detailed evaluation report.
        
        Returns:
            DataFrame with per-query metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame.from_dict(self.results, orient='index')


def evaluate_ranking_methods(search_function, 
                           test_queries: List[Dict],
                           method_configs: Dict[str, Dict]) -> pd.DataFrame:
    """
    Evaluate different ranking methods on test queries.
    
    Args:
        search_function: Function that takes (query, **config) and returns ranked results
        test_queries: List of test queries with ground truth
        method_configs: Dictionary of method configurations
        
    Returns:
        DataFrame with method comparison results
    """
    results = {}
    
    for method_name, config in method_configs.items():
        logger.info(f"Evaluating method: {method_name}")
        
        evaluator = IRMetrics()
        
        for query_data in test_queries:
            query = query_data['query']
            relevant_items = set(query_data['relevant_items'])
            query_id = query_data.get('id', query)
            
            try:
                # Run search with this method configuration
                search_results, _, _ = search_function(query, **config)
                retrieved_items = search_results['sku'].tolist()
                
                # Evaluate this query
                evaluator.evaluate_query(query_id, retrieved_items, relevant_items)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}' with method '{method_name}': {e}")
                continue
        
        # Aggregate results for this method
        method_results = evaluator.aggregate_metrics()
        results[method_name] = method_results
    
    # Convert to DataFrame for easy comparison
    return pd.DataFrame(results).T


if __name__ == "__main__":
    # Example usage
    print("IR Metrics Test")
    
    # Test data
    retrieved = ["item1", "item2", "item3", "item4", "item5"]
    relevant = {"item2", "item4", "item6"}
    
    evaluator = IRMetrics()
    metrics = evaluator.evaluate_query("test_query", retrieved, relevant)
    
    print("Metrics:", metrics)