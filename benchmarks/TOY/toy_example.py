#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-25
# File Name: toy_example.py
# Description: Example usage of ToyBench with LLMAT
"""
import numpy as np
from benchmarks.TOY.toy_benchmarks import ToyBench

def example_usage():
    """Example of how to use ToyBench with the same interface as MATBench"""

    # Example 1: Basic usage with Levy function
    print("="*60)
    print("Example 1: Levy function with random embeddings")
    print("="*60)

    levy_bench = ToyBench(
        data_name="levy",
        run_subset_only=False,  # Use full dataset
        feature_type="random_embedding",  # Use random embeddings
        finetuning=False,
        iupac=False,
        prompt_type=None,
        randseed=666,
        clustering_type="kmeans",
        feature_reduction=None,
        dims=10,  # Dimension of the Levy function
        n_samples=1000  # Number of samples to generate
    )

    print(f"\nDataset info:")
    print(f"  - Number of samples: {len(levy_bench.dataset)}")
    print(f"  - Number of clusters: {levy_bench.n_clusters}")
    print(f"  - Ground truth optimum: {levy_bench.ground_truth_opt:.4f}")
    print(f"  - Maximization problem: {levy_bench.maximization}")

    # Test query
    test_idx = 0
    test_feature = levy_bench.dataset.iloc[test_idx]["features"]
    result = levy_bench(test_feature)
    print(f"\nTest query (index {test_idx}): {result:.4f}")

    # Example 2: Ackley function with subset
    print("\n" + "="*60)
    print("Example 2: Ackley function with subset")
    print("="*60)

    ackley_bench = ToyBench(
        data_name="ackley",
        run_subset_only=True,  # Use subset only
        feature_type="random_embedding",
        finetuning=False,
        iupac=False,
        prompt_type=None,
        randseed=777,
        clustering_type="kmeans",
        feature_reduction=None,
        dims=5,  # 5D Ackley function
        n_samples=200  # Smaller dataset for subset
    )

    print(f"\nDataset info:")
    print(f"  - Number of samples: {len(ackley_bench.dataset)}")
    print(f"  - Number of clusters: {ackley_bench.n_clusters}")
    print(f"  - Ground truth optimum: {ackley_bench.ground_truth_opt:.4f}")

    # Show cluster distribution
    cluster_col = ackley_bench._get_cluster_col()
    cluster_counts = ackley_bench.dataset[cluster_col].value_counts()
    print(f"\nCluster distribution:")
    for cluster_id in sorted(cluster_counts.index):
        print(f"  - Cluster {cluster_id}: {cluster_counts[cluster_id]} samples")

    # Example 3: Usage pattern similar to optimization loop
    print("\n" + "="*60)
    print("Example 3: Simulated optimization loop")
    print("="*60)

    # Create benchmark
    bench = ToyBench(
        data_name="levy",
        run_subset_only=True,
        feature_type="random_embedding",
        finetuning=False,
        iupac=False,
        prompt_type=None,
        randseed=42,
        clustering_type="kmeans",
        feature_reduction=None,
        dims=10,
        n_samples=200
    )

    # Simulate optimization by randomly querying points
    np.random.seed(42)
    n_queries = 10
    print(f"\nSimulating {n_queries} random queries:")

    best_value = float('-inf')
    best_idx = None

    for i in range(n_queries):
        # Random query
        idx = np.random.randint(0, len(bench.dataset))
        feature = bench.dataset.iloc[idx]["features"]

        # Query the function
        value = bench(feature)

        # Track best
        if value > best_value:
            best_value = value
            best_idx = idx

        print(f"  Query {i+1}: idx={idx:3d}, value={value:8.4f}, "
              f"best_so_far={best_value:8.4f}")

    print(f"\nBest found: {best_value:.4f} (index {best_idx})")
    print(f"Ground truth optimum: {bench.ground_truth_opt:.4f}")
    print(f"Gap from optimum: {abs(best_value - bench.ground_truth_opt):.4f}")


if __name__ == "__main__":
    example_usage()
