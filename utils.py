"""
Utility functions for the search application.
Extracted for better testability and reusability.
"""
import re
import math
import numpy as np
from typing import List, Dict, Set

# Constants
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOP_WORDS = {"a","an","the","and","or","of","for","to","in","on","with","is","are","it","this","that"}

# Synonym and color mappings
SYNONYMS = {
    "sock": {"sock","socks"},
    "headphone": {"headphone","headphones","earphone","earphones","earbud","earbuds","headset"},
    "keyboard": {"keyboard","keyboards"},
    "wireless": {"wireless","bluetooth"},
    "noise": {"noise cancelling","noise-canceling","noise canceling","anc"},
    "cat": {"cat","cats","kitten","kittens","kitty"},
    "dog": {"dog","dogs","puppy","puppies"},
    "design": {"design","pattern","print","graphic","artwork","motif","theme"},
}

COLORS = {
    "yellow":{"yellow","mustard","lemon","gold","golden"},
    "red":{"red","scarlet","crimson","maroon"},
    "blue":{"blue","navy","cobalt","azure"},
    "green":{"green","emerald","olive"},
    "black":{"black"},
    "white":{"white","ivory"},
    "pink":{"pink","rose"},
    "purple":{"purple","violet","lavender"},
    "orange":{"orange","amber"},
    "brown":{"brown","tan","beige","khaki"},
    "gray":{"gray","grey","charcoal","slate"},
}

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vectors along specified axis."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1] range."""
    if x.size == 0:
        return x.astype(np.float32)
    
    lo, hi = float(np.min(x)), float(np.max(x))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    
    return ((x - lo) / (hi - lo + 1e-12)).astype(np.float32)

def tokenize_query(query: str) -> List[str]:
    """Tokenize query string, removing stop words."""
    tokens = TOKEN_RE.findall(query.lower())
    return [t for t in tokens if t not in STOP_WORDS]

def build_gate_groups(query: str) -> List[Set[str]]:
    """Build gating groups from query for attribute matching."""
    query_lower = query.lower()
    groups: List[Set[str]] = []
    
    # Add colors explicitly mentioned
    for color_name, color_synonyms in COLORS.items():
        if any(word in query_lower for word in color_synonyms):
            groups.append(color_synonyms)
    
    # Add category/key nouns
    tokens = tokenize_query(query)
    for token in tokens:
        if token in SYNONYMS:
            groups.append(SYNONYMS[token])
        elif len(token) >= 4:  # Only consider longer tokens
            groups.append({token})
    
    # Remove duplicate groups
    unique_groups = []
    for group in groups:
        if group not in unique_groups:
            unique_groups.append(group)
    
    return unique_groups[:6]  # Limit to 6 groups

def calculate_gate_factor(text: str, groups: List[Set[str]], penalty: float = 0.5) -> tuple[float, int, int]:
    """Calculate gating factor based on how many attribute groups match the text."""
    text_lower = text.lower()
    hits = 0
    factor = 1.0
    
    for group in groups:
        has_match = any(synonym in text_lower for synonym in group)
        if has_match:
            hits += 1
        else:
            factor *= penalty
    
    return factor, hits, len(groups)

def bayesian_prior(avg_ratings: np.ndarray, review_counts: np.ndarray, 
                  prior_strength: float = 20.0, global_mean: float = None) -> np.ndarray:
    """Calculate Bayesian prior ratings with review count consideration."""
    if global_mean is None:
        global_mean = float(np.nanmean(avg_ratings))
    
    return ((avg_ratings * review_counts) + (global_mean * prior_strength)) / (review_counts + prior_strength + 1e-9)

def cosine_similarity_search(query_vector: np.ndarray, embeddings_matrix: np.ndarray, 
                           top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform cosine similarity search and return top-k results."""
    similarities = embeddings_matrix @ query_vector
    
    if top_k >= len(similarities):
        top_k = len(similarities)
    
    # Get top-k indices
    top_indices = np.argpartition(-similarities, top_k-1)[:top_k]
    # Sort top-k by similarity
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    
    return top_indices, similarities[top_indices]

def trust_score_from_reviews(review_counts: np.ndarray, min_reviews: int = 8, 
                           saturation: int = 50) -> np.ndarray:
    """Calculate trust score based on number of reviews."""
    # Linear ramp up to min_reviews, then logarithmic saturation
    ramp = np.clip(review_counts / max(min_reviews, 1), 0, 1)
    saturation_score = np.minimum(1.0, np.log1p(review_counts) / np.log1p(max(saturation, 1)))
    
    return (0.6 * ramp + 0.4 * saturation_score).astype(np.float32)