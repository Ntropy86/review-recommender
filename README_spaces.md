---
title: Review Search Copilot
emoji: ✨
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.49.1
app_file: app.py
pinned: false

license: mit
---

# Review Search Copilot

A production-ready AI-powered product search engine that queries millions of product reviews in natural language. Combining semantic embeddings with BM25 sparse retrieval for enterprise-grade product discovery.

## Features

- **Natural Language Search**: Query in plain English like "comfortable yellow cat socks for winter"
- **Hybrid Retrieval**: Combines semantic embeddings + keyword matching for optimal results
- **AI-Powered Ranking**: Cross-encoder reranking with Bayesian priors for quality scoring
- **Review Intelligence**: Leverages millions of product reviews for better recommendations
- **Interactive UI**: Professional Streamlit interface with real-time parameter tuning

## How it Works

1. **Data Processing**: Product reviews are aggregated and encoded using sentence transformers
2. **Hybrid Search**: Dense retrieval (embeddings) + sparse retrieval (BM25) for comprehensive search
3. **Advanced Ranking**: Cross-encoder reranking, Bayesian priors, and trust scoring
4. **Review Snippets**: Best matching review excerpts for each product with relevance scores

## Tech Stack

- **Frontend**: Streamlit
- **ML**: Sentence Transformers, Cross-Encoders, NumPy
- **Data**: Hugging Face Datasets, Pandas, PyArrow
- **Search**: BM25, Cosine Similarity, Cross-Encoder Reranking

## Dataset

This app uses the [Amazon Product Reviews Compiled](https://huggingface.co/datasets/ntropy86/AmazonReviewsCompiled) dataset containing:
- Product embeddings (BGE-small-en-v1.5)
- Review embeddings and metadata
- BM25 search indices
- Aggregated product information

## Usage

1. Enter your search query in natural language
2. Adjust search parameters (weights, reranking pool, etc.)
3. Browse results with relevance scores and review snippets
4. Use the "How it Works" tab to understand the search pipeline

## Performance

- **nDCG@10**: 0.867 (with reranking)
- **MRR@10**: 0.824
- **Recall@20**: 0.695

## Author

Built by [Nitigya Kargeti](https://github.com/ntropy86) - Data Science & ML Engineer