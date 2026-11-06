#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-25
# File Name: test_compatibility.py
# Description: Test compatibility between ToyBench and MATBench interfaces
"""
import numpy as np
from benchmarks.TOY.toy_benchmarks import ToyBench


def test_basic_interface():
    """Test that ToyBench has the same basic interface as MATBench"""
    print("="*70)
    print("Test 1: Basic Interface Compatibility")
    print("="*70)

    bench = ToyBench(
        data_name="levy",
        run_subset_only=True,
        feature_type="random_embedding",
        finetuning=False,
        iupac=False,
        prompt_type=None,
        randseed=666,
        clustering_type="kmeans",
        feature_reduction=None,
        dims=5,
        n_samples=100
    )

    # Check required attributes
    required_attrs = [
        'dataset', 'target_col', 'target_col_transformed',
        'ground_truth_opt', 'ground_truth_max',
        'ground_truth_opt_id', 'ground_truth_opt_cluster',
        'ground_truth_opt_clusters', 'n_clusters',
        'maximization', 'data_name', 'randseed'
    ]

    print("\nChecking required attributes:")
    for attr in required_attrs:
        has_attr = hasattr(bench, attr)
        status = "✓" if has_attr else "✗"
        print(f"  {status} {attr}: {has_attr}")
        if has_attr:
            value = getattr(bench, attr)
            if not isinstance(value, (pd.DataFrame, np.ndarray)):
                print(f"      Value: {value}")

    # Check required methods
    required_methods = ['__call__', 'complete_call', 'save_dataset']

    print("\nChecking required methods:")
    for method in required_methods:
        has_method = hasattr(bench, method) and callable(getattr(bench, method))
        status = "✓" if has_method else "✗"
        print(f"  {status} {method}: {has_method}")

    print("\n✅ All interface requirements met!\n")
    return bench


def test_dataset_structure():
    """Test that the dataset has the correct structure"""
    print("="*70)
    print("Test 2: Dataset Structure")
    print("="*70)

    bench = ToyBench(
        data_name="ackley",
        run_subset_only=True,
        randseed=42,
        dims=5,
        n_samples=100
    )

    # Check required columns
    required_columns = [
        'Entry Number', 'features', 'targets',
        'cluster', 'targets_transformed'
    ]

    print("\nChecking dataset columns:")
    for col in required_columns:
        has_col = col in bench.dataset.columns
        status = "✓" if has_col else "✗"
        print(f"  {status} {col}: {has_col}")

    print(f"\nDataset shape: {bench.dataset.shape}")
    print(f"Columns: {list(bench.dataset.columns)}")
    print(f"\nSample data:")
    print(bench.dataset.head(3))

    print("\n✅ Dataset structure is correct!\n")
    return bench


def test_clustering():
    """Test that clustering works correctly"""
    print("="*70)
    print("Test 3: Clustering")
    print("="*70)

    bench = ToyBench(
        data_name="levy",
        run_subset_only=True,
        clustering_type="kmeans",
        randseed=123,
        dims=10,
        n_samples=200
    )

    cluster_col = bench._get_cluster_col()
    clusters = bench.dataset[cluster_col].unique()

    print(f"\nClustering type: {bench.clustering_type}")
    print(f"Number of clusters: {bench.n_clusters}")
    print(f"Cluster column: {cluster_col}")
    print(f"Unique clusters: {sorted(clusters)}")

    # Check cluster distribution
    cluster_counts = bench.dataset[cluster_col].value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        pct = 100 * count / len(bench.dataset)
        print(f"  Cluster {cluster_id}: {count:3d} samples ({pct:5.1f}%)")

    # Verify clusters are continuous from 0 to n_clusters-1
    expected_clusters = set(range(bench.n_clusters))
    actual_clusters = set(clusters)
    assert expected_clusters == actual_clusters, \
        f"Clusters should be {expected_clusters}, but got {actual_clusters}"

    print("\n✅ Clustering is working correctly!\n")
    return bench


def test_function_evaluation():
    """Test that function evaluation works"""
    print("="*70)
    print("Test 4: Function Evaluation")
    print("="*70)

    bench = ToyBench(
        data_name="ackley",
        run_subset_only=True,
        randseed=999,
        dims=5,
        n_samples=50
    )

    print(f"\nGround truth optimum: {bench.ground_truth_opt:.6f}")
    print(f"Ground truth max (transformed): {bench.ground_truth_max:.6f}")
    print(f"Optimal cluster: {bench.ground_truth_opt_cluster}")

    # Test single evaluation
    print("\n--- Single Evaluation ---")
    test_idx = 0
    feature = bench.dataset.iloc[test_idx]["features"]
    expected_value = bench.dataset.iloc[test_idx][bench.target_col]

    # Test __call__
    result1 = bench(feature)
    print(f"Using __call__: {result1:.6f}")
    print(f"Expected value: {expected_value:.6f}")
    assert np.isclose(result1, expected_value), \
        f"Mismatch: {result1} != {expected_value}"

    # Test complete_call
    result2, info = bench.complete_call(feature)
    print(f"Using complete_call: {result2:.6f}")
    print(f"Timing info: init={info['time_init']:.6f}, final={info['time_final']:.6f}")
    assert np.isclose(result2, expected_value), \
        f"Mismatch: {result2} != {expected_value}"

    # Test multiple evaluations
    print("\n--- Multiple Evaluations ---")
    n_tests = 5
    for i in range(n_tests):
        idx = np.random.randint(0, len(bench.dataset))
        feature = bench.dataset.iloc[idx]["features"]
        expected = bench.dataset.iloc[idx][bench.target_col]
        result = bench(feature)
        match = "✓" if np.isclose(result, expected) else "✗"
        print(f"  {match} Query {i+1} (idx={idx:2d}): result={result:8.4f}, expected={expected:8.4f}")

    print("\n✅ Function evaluation is working correctly!\n")
    return bench


def test_optimization_workflow():
    """Test a simple optimization workflow"""
    print("="*70)
    print("Test 5: Optimization Workflow Simulation")
    print("="*70)

    bench = ToyBench(
        data_name="levy",
        run_subset_only=True,
        randseed=42,
        dims=10,
        n_samples=200
    )

    print(f"\nStarting optimization simulation...")
    print(f"Dataset size: {len(bench.dataset)}")
    print(f"Ground truth optimum: {bench.ground_truth_opt:.4f}")

    # Simulate random search
    n_iterations = 20
    best_value = float('-inf')
    best_idx = None

    np.random.seed(42)
    queried_indices = set()

    print(f"\nRunning {n_iterations} iterations of random search:")
    for i in range(n_iterations):
        # Random sampling (ensuring no duplicates)
        while True:
            idx = np.random.randint(0, len(bench.dataset))
            if idx not in queried_indices:
                queried_indices.add(idx)
                break

        # Query
        feature = bench.dataset.iloc[idx]["features"]
        value = bench(feature)

        # Track best
        if value > best_value:
            best_value = value
            best_idx = idx
            improved = "🌟"
        else:
            improved = "  "

        if i < 10 or improved == "🌟":  # Print first 10 and improvements
            print(f"  {improved} Iter {i+1:2d}: idx={idx:3d}, "
                  f"value={value:8.4f}, best={best_value:8.4f}")

    # Calculate gap to optimum
    gap = abs(best_value - bench.ground_truth_opt)
    pct_gap = 100 * gap / abs(bench.ground_truth_opt) if bench.ground_truth_opt != 0 else 0

    print(f"\n--- Results ---")
    print(f"Best found: {best_value:.4f} (index {best_idx})")
    print(f"Ground truth: {bench.ground_truth_opt:.4f}")
    print(f"Gap: {gap:.4f} ({pct_gap:.2f}%)")
    print(f"Unique queries: {len(queried_indices)}")

    print("\n✅ Optimization workflow completed successfully!\n")
    return bench


def test_both_functions():
    """Test both Levy and Ackley functions"""
    print("="*70)
    print("Test 6: Both Functions (Levy & Ackley)")
    print("="*70)

    for func_name in ["levy", "ackley"]:
        print(f"\n--- Testing {func_name.upper()} ---")

        bench = ToyBench(
            data_name=func_name,
            run_subset_only=True,
            randseed=123,
            dims=5,
            n_samples=100
        )

        print(f"Dataset size: {len(bench.dataset)}")
        print(f"Dimensions: {bench.dims}")
        print(f"Maximization: {bench.maximization}")
        print(f"Ground truth optimum: {bench.ground_truth_opt:.6f}")
        print(f"Value range: [{bench.dataset[bench.target_col].min():.6f}, "
              f"{bench.dataset[bench.target_col].max():.6f}]")

        # Test a few queries
        n_queries = 3
        print(f"\nTesting {n_queries} random queries:")
        for i in range(n_queries):
            idx = np.random.randint(0, len(bench.dataset))
            feature = bench.dataset.iloc[idx]["features"]
            value = bench(feature)
            print(f"  Query {i+1}: value = {value:.6f}")

    print("\n✅ Both functions working correctly!\n")


def run_all_tests():
    """Run all compatibility tests"""
    print("\n" + "="*70)
    print("TOYBENCH COMPATIBILITY TEST SUITE")
    print("="*70 + "\n")

    try:
        test_basic_interface()
        test_dataset_structure()
        test_clustering()
        test_function_evaluation()
        test_optimization_workflow()
        test_both_functions()

        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nToyBench is fully compatible with MATBench interface.")
        print("You can now use it as a drop-in replacement for testing.\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    import pandas as pd
    run_all_tests()
