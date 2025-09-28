"""
Test queries and ground truth data for evaluation.

This module defines test queries with manually curated relevant products
for reproducible performance evaluation.
"""

import pandas as pd
from typing import List, Dict, Set


# Test queries with ground truth relevance judgments
TEST_QUERIES = [
    {
        "id": "q1",
        "query": "wireless bluetooth headphones",
        "relevant_items": {
            # High relevance - exact matches
            "B077T3RMRZ", "B01E9KO4ZQ", "B075YJ8M7H", "B071K4N3MV",
            # Medium relevance - close matches  
            "B01MQPX5S8", "B07G2NQYY3", "B077R3KPKX", "B01D5LXBMM",
            # Lower relevance but still relevant
            "B073QHKK6W", "B01E9KO4ZQ"
        },
        "category": "Electronics",
        "expected_attributes": ["wireless", "bluetooth", "headphones"]
    },
    {
        "id": "q2", 
        "query": "comfortable running shoes",
        "relevant_items": {
            "B075R7YY7P", "B075YH6JSL", "B01IODF7P8", "B072KG7MZ2",
            "B01GH4E8KW", "B075RHCKYX", "B01IODF7P8", "B072KG7MZ2"
        },
        "category": "Shoes",
        "expected_attributes": ["comfortable", "running", "shoes"]
    },
    {
        "id": "q3",
        "query": "yellow cat socks",
        "relevant_items": {
            "B01N7TQFHP", "B074Q8R2ZV", "B01MS8QSJP", "B075DKRPFR",
            "B01N0P3RLB", "B074Q6HDY4"
        },
        "category": "Clothing",
        "expected_attributes": ["yellow", "cat", "socks"]
    },
    {
        "id": "q4",
        "query": "kitchen knife set stainless steel",
        "relevant_items": {
            "B00K8LK40E", "B01F9Q1CM2", "B01D5LXBMM", "B01MQPX5S8",
            "B073QHKK6W", "B077R3KPKX", "B07G2NQYY3", "B075YJ8M7H"
        },
        "category": "Kitchen",
        "expected_attributes": ["kitchen", "knife", "stainless", "steel"]
    },
    {
        "id": "q5",
        "query": "gaming mouse RGB",
        "relevant_items": {
            "B071K4N3MV", "B077T3RMRZ", "B075R7YY7P", "B01E9KO4ZQ",
            "B075YH6JSL", "B01IODF7P8"
        },
        "category": "Electronics", 
        "expected_attributes": ["gaming", "mouse", "RGB"]
    },
    {
        "id": "q6",
        "query": "waterproof phone case",
        "relevant_items": {
            "B072KG7MZ2", "B01GH4E8KW", "B075RHCKYX", "B074Q8R2ZV",
            "B01MS8QSJP", "B075DKRPFR"
        },
        "category": "Electronics",
        "expected_attributes": ["waterproof", "phone", "case"]
    },
    {
        "id": "q7",
        "query": "organic green tea",
        "relevant_items": {
            "B01N0P3RLB", "B074Q6HDY4", "B01N7TQFHP", "B00K8LK40E",
            "B01F9Q1CM2", "B01D5LXBMM"
        },
        "category": "Food",
        "expected_attributes": ["organic", "green", "tea"]
    },
    {
        "id": "q8",
        "query": "leather wallet men",
        "relevant_items": {
            "B01MQPX5S8", "B073QHKK6W", "B077R3KPKX", "B07G2NQYY3",
            "B075YJ8M7H", "B071K4N3MV"
        },
        "category": "Fashion",
        "expected_attributes": ["leather", "wallet", "men"]
    },
    {
        "id": "q9",
        "query": "USB charging cable long",
        "relevant_items": {
            "B077T3RMRZ", "B075R7YY7P", "B01E9KO4ZQ", "B075YH6JSL",
            "B01IODF7P8", "B072KG7MZ2"
        },
        "category": "Electronics",
        "expected_attributes": ["USB", "charging", "cable", "long"]
    },
    {
        "id": "q10",
        "query": "soft cotton t-shirt",
        "relevant_items": {
            "B01GH4E8KW", "B075RHCKYX", "B074Q8R2ZV", "B01MS8QSJP",
            "B075DKRPFR", "B01N0P3RLB"
        },
        "category": "Clothing",
        "expected_attributes": ["soft", "cotton", "t-shirt"]
    }
]


def load_test_queries() -> List[Dict]:
    """
    Load standardized test queries for evaluation.
    
    Returns:
        List of test query dictionaries
    """
    return TEST_QUERIES


def get_query_by_id(query_id: str) -> Dict:
    """
    Get a specific test query by ID.
    
    Args:
        query_id: Query identifier
        
    Returns:
        Query dictionary or None if not found
    """
    for query in TEST_QUERIES:
        if query['id'] == query_id:
            return query
    return None


def validate_ground_truth(product_data: pd.DataFrame) -> Dict[str, int]:
    """
    Validate that ground truth SKUs exist in the product data.
    
    Args:
        product_data: DataFrame with product information
        
    Returns:
        Dictionary with validation statistics
    """
    if 'sku' not in product_data.columns:
        raise ValueError("Product data must have 'sku' column")
    
    available_skus = set(product_data['sku'].values)
    
    total_queries = len(TEST_QUERIES)
    total_relevant_items = 0
    found_relevant_items = 0
    
    missing_by_query = {}
    
    for query in TEST_QUERIES:
        query_id = query['id']
        relevant_items = query['relevant_items']
        
        total_relevant_items += len(relevant_items)
        found_items = relevant_items & available_skus
        found_relevant_items += len(found_items)
        
        missing_items = relevant_items - available_skus
        if missing_items:
            missing_by_query[query_id] = list(missing_items)
    
    stats = {
        'total_queries': total_queries,
        'total_relevant_items': total_relevant_items,  
        'found_relevant_items': found_relevant_items,
        'coverage_rate': found_relevant_items / total_relevant_items if total_relevant_items > 0 else 0.0,
        'missing_by_query': missing_by_query
    }
    
    return stats


def create_synthetic_ground_truth(product_data: pd.DataFrame, num_queries: int = 20) -> List[Dict]:
    """
    Create synthetic ground truth from actual product data for testing.
    
    Args:
        product_data: DataFrame with product information
        num_queries: Number of synthetic queries to generate
        
    Returns:
        List of synthetic test queries
    """
    import random
    import re
    
    if 'sku' not in product_data.columns:
        raise ValueError("Product data must have 'sku' column")
    
    # Common query patterns
    query_templates = [
        "wireless {category}",
        "{color} {product_type}",
        "comfortable {product_type}",
        "{material} {product_type}",
        "best {category} for {use}",
        "{brand} {product_type}",
        "waterproof {product_type}",
        "premium {category}",
    ]
    
    # Extract common terms from product titles/descriptions
    text_column = 'title' if 'title' in product_data.columns else 'product_name'
    if text_column not in product_data.columns:
        text_column = product_data.columns[1]  # Use second column as fallback
    
    synthetic_queries = []
    
    for i in range(num_queries):
        # Sample random products as relevant set
        sample_products = product_data.sample(n=min(10, len(product_data)))
        relevant_skus = set(sample_products['sku'].values)
        
        # Extract keywords from sampled products
        sample_text = ' '.join(sample_products[text_column].astype(str).values)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sample_text.lower())
        common_words = list(set(words))[:5]  # Take first 5 unique words
        
        # Generate synthetic query
        if common_words:
            query = ' '.join(random.sample(common_words, min(3, len(common_words))))
        else:
            query = f"product {i+1}"
        
        synthetic_queries.append({
            'id': f'synthetic_{i+1}',
            'query': query,
            'relevant_items': relevant_skus,
            'category': 'Synthetic',
            'expected_attributes': common_words[:3]
        })
    
    return synthetic_queries


# Method configurations for benchmarking
BENCHMARK_CONFIGS = {
    "Dense Only": {
        "k": 20,
        "rerank_k": 0,
        "w_dense": 1.0,
        "w_bm25": 0.0,
        "w_rerank": 0.0,
        "w_prior": 0.0,
        "w_best": 0.0,
        "prior_C": 20.0,
        "use_snips": False,
        "max_scan": 50000,
        "min_reviews": 1,
        "gate_penalty": 0.0
    },
    "BM25 Only": {
        "k": 20,
        "rerank_k": 0,
        "w_dense": 0.0,
        "w_bm25": 1.0,
        "w_rerank": 0.0,
        "w_prior": 0.0,
        "w_best": 0.0,
        "prior_C": 20.0,
        "use_snips": False,
        "max_scan": 50000,
        "min_reviews": 1,
        "gate_penalty": 0.0
    },
    "Hybrid": {
        "k": 20,
        "rerank_k": 0,
        "w_dense": 0.5,
        "w_bm25": 0.3,
        "w_rerank": 0.0,
        "w_prior": 0.2,
        "w_best": 0.0,
        "prior_C": 20.0,
        "use_snips": False,
        "max_scan": 50000,
        "min_reviews": 5,
        "gate_penalty": 0.3
    },
    "Hybrid + Rerank": {
        "k": 50,
        "rerank_k": 20,
        "w_dense": 0.4,
        "w_bm25": 0.2,
        "w_rerank": 0.3,
        "w_prior": 0.1, 
        "w_best": 0.0,
        "prior_C": 20.0,
        "use_snips": False,
        "max_scan": 50000,
        "min_reviews": 5,
        "gate_penalty": 0.5
    }
}


if __name__ == "__main__":
    # Example usage
    queries = load_test_queries()
    print(f"Loaded {len(queries)} test queries")
    
    for query in queries[:3]:
        print(f"Query: {query['query']}")
        print(f"Relevant items: {len(query['relevant_items'])}")
        print("---")