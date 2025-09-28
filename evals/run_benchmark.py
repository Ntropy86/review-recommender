"""
Performance benchmark test suite.

This script runs comprehensive performance evaluations of different retrieval methods
and generates the metrics shown in the README.
"""

import sys
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.performance_metrics import evaluate_ranking_methods, IRMetrics
from evals.test_queries import load_test_queries, BENCHMARK_CONFIGS, validate_ground_truth, create_synthetic_ground_truth
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_availability():
    """Check if required data files are available."""
    required_files = [
        config.PRODUCT_EMB_PATH,
        config.PRODUCT_META_PATH, 
        config.BM25_PATH
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Missing data files: {missing_files}")
        return False
    
    logger.info("All required data files found")
    return True


def load_search_function():
    """Load the search function from the main app."""
    try:
        from app.app_product_search import run_search
        logger.info("Successfully loaded search function")
        return run_search
    except ImportError as e:
        logger.error(f"Failed to import search function: {e}")
        return None


def run_performance_benchmark(use_synthetic_queries=False, num_synthetic=20):
    """
    Run the main performance benchmark.
    
    Args:
        use_synthetic_queries: Whether to use synthetic queries if real data is available
        num_synthetic: Number of synthetic queries to generate
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Starting performance benchmark...")
    
    # Check data availability
    if not check_data_availability():
        logger.error("Required data files not available. Please run data preprocessing first.")
        return None
    
    # Load search function
    search_function = load_search_function()
    if search_function is None:
        logger.error("Could not load search function")
        return None
    
    try:
        # Load test queries
        test_queries = load_test_queries()
        logger.info(f"Loaded {len(test_queries)} predefined test queries")
        
        # If using synthetic queries or if we need to supplement
        if use_synthetic_queries:
            try:
                # Load product metadata to create synthetic queries
                from app.app_product_search import _product_index
                meta, _ = _product_index()
                
                if meta is not None and len(meta) > 0:
                    synthetic_queries = create_synthetic_ground_truth(meta, num_synthetic)
                    test_queries.extend(synthetic_queries)
                    logger.info(f"Added {len(synthetic_queries)} synthetic queries")
                else:
                    logger.warning("Could not load product metadata for synthetic queries")
            except Exception as e:
                logger.warning(f"Failed to create synthetic queries: {e}")
        
        # Validate ground truth against available data
        try:
            from app.app_product_search import _product_index
            meta, _ = _product_index()
            
            if meta is not None:
                validation_stats = validate_ground_truth(meta)
                logger.info(f"Ground truth validation: {validation_stats['coverage_rate']:.2%} coverage")
                
                if validation_stats['coverage_rate'] < 0.1:  # Less than 10% coverage
                    logger.warning("Low ground truth coverage. Consider using synthetic queries.")
            
        except Exception as e:
            logger.warning(f"Could not validate ground truth: {e}")
        
        # Run benchmarks for each method
        logger.info("Running benchmark evaluations...")
        results_df = evaluate_ranking_methods(search_function, test_queries, BENCHMARK_CONFIGS)
        
        # Format results
        if results_df.empty:
            logger.error("No benchmark results generated")
            return None
        
        # Extract key metrics mentioned in README
        key_metrics = ['ndcg@10', 'mrr', 'recall@20']
        
        benchmark_results = {}
        for method in results_df.index:
            method_results = {}
            for metric in key_metrics:
                if metric in results_df.columns:
                    method_results[metric] = results_df.loc[method, metric]
                else:
                    method_results[metric] = 0.0
            benchmark_results[method] = method_results
        
        # Add metadata
        benchmark_results['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(test_queries),
            'config_file': str(config.CONFIG_FILE) if hasattr(config, 'CONFIG_FILE') else 'default',
            'data_files_checked': True
        }
        
        logger.info("Benchmark completed successfully")
        return benchmark_results, results_df
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return None


def format_results_for_readme(benchmark_results: dict) -> str:
    """
    Format benchmark results for README table.
    
    Args:
        benchmark_results: Results from run_performance_benchmark
        
    Returns:
        Formatted table string for README
    """
    if not benchmark_results or '_metadata' not in benchmark_results:
        return "No benchmark results available"
    
    # Remove metadata for table formatting
    results = {k: v for k, v in benchmark_results.items() if k != '_metadata'}
    
    # Create table header
    table = "| Metric | Dense Only | BM25 Only | Hybrid | Hybrid + Rerank |\n"
    table += "|--------|------------|-----------|---------|---------------|\n"
    
    # Metric mapping for display
    metric_display = {
        'ndcg@10': 'nDCG@10',
        'mrr': 'MRR@10', 
        'recall@20': 'Recall@20'
    }
    
    # Add rows for each metric
    for metric_key, display_name in metric_display.items():
        table += f"| {display_name} |"
        
        for method in ['Dense Only', 'BM25 Only', 'Hybrid', 'Hybrid + Rerank']:
            if method in results and metric_key in results[method]:
                value = results[method][metric_key]
                table += f" {value:.3f} |"
            else:
                table += " 0.000 |"
        
        table += "\n"
    
    return table


def save_benchmark_results(benchmark_results: dict, results_df: pd.DataFrame, output_dir: str = "evals"):
    """
    Save benchmark results to files.
    
    Args:
        benchmark_results: Benchmark results dictionary
        results_df: Detailed results DataFrame
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save summary results as JSON
    summary_file = output_path / "benchmark_results.json"
    with open(summary_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"Saved benchmark summary to {summary_file}")
    
    # Save detailed results as CSV
    detailed_file = output_path / "detailed_results.csv"
    results_df.to_csv(detailed_file)
    logger.info(f"Saved detailed results to {detailed_file}")
    
    # Save README table format
    table_file = output_path / "readme_table.md"
    with open(table_file, 'w') as f:
        f.write("# Performance Metrics\n\n")
        f.write(format_results_for_readme(benchmark_results))
        f.write(f"\n\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    logger.info(f"Saved README table to {table_file}")


def main():
    """Main benchmark execution."""
    print("🚀 Running Performance Benchmark")
    print("=" * 50)
    
    # Run benchmark with both predefined and synthetic queries
    results = run_performance_benchmark(use_synthetic_queries=True, num_synthetic=10)
    
    if results is None:
        print("❌ Benchmark failed")
        return 1
    
    benchmark_results, results_df = results
    
    # Display results
    print("\n📊 Benchmark Results:")
    print("-" * 30)
    
    if '_metadata' in benchmark_results:
        metadata = benchmark_results['_metadata']
        print(f"Queries tested: {metadata['num_queries']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print()
    
    # Show formatted table
    print(format_results_for_readme(benchmark_results))
    
    # Save results
    save_benchmark_results(benchmark_results, results_df)
    
    print("\n✅ Benchmark completed successfully!")
    print(f"Results saved to evals/ directory")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)