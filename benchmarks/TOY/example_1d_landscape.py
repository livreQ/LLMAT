#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-25
# File Name: example_1d_landscape.py
# Description: Example usage of 1D landscape plotting for toy benchmarks
"""
from benchmarks.TOY import ToyBench
from benchmarks.TOY.visualize_toy import plot_1d_function_landscape
import matplotlib.pyplot as plt


def example_levy_1d():
    """Example: Visualize Levy function in 1D"""
    print("=" * 60)
    print("Example: Levy Function 1D Landscape")
    print("=" * 60)

    # Create benchmark
    bench = ToyBench(
        data_name="levy",
        dims=1,
        n_samples=500,
        randseed=42,
        clustering_type="kmeans",
        run_subset_only=False
    )

    print(f"\nDataset size: {len(bench.dataset)}")
    print(f"Ground truth optimum: {bench.ground_truth_opt:.4f}")
    print(f"Number of clusters: {bench.n_clusters}")

    # Plot 1D landscape
    print("\nGenerating 1D landscape visualization...")
    plot_1d_function_landscape(
        bench,
        n_grid=200,
        save_path="benchmarks/TOY/visualizations/example_levy_1d.png"
    )

    print("\n✓ Visualization saved!")


def example_ackley_1d():
    """Example: Visualize Ackley function in 1D"""
    print("\n" + "=" * 60)
    print("Example: Ackley Function 1D Landscape")
    print("=" * 60)

    # Create benchmark
    bench = ToyBench(
        data_name="ackley",
        dims=1,
        n_samples=500,
        randseed=42,
        clustering_type="kmeans",
        run_subset_only=False
    )

    print(f"\nDataset size: {len(bench.dataset)}")
    print(f"Ground truth optimum: {bench.ground_truth_opt:.4f}")
    print(f"Number of clusters: {bench.n_clusters}")

    # Plot 1D landscape
    print("\nGenerating 1D landscape visualization...")
    plot_1d_function_landscape(
        bench,
        n_grid=200,
        save_path="benchmarks/TOY/visualizations/example_ackley_1d.png"
    )

    print("\n✓ Visualization saved!")


def compare_functions():
    """Example: Compare Levy and Ackley side by side"""
    print("\n" + "=" * 60)
    print("Comparison: Levy vs Ackley (1D)")
    print("=" * 60)

    # Create both benchmarks with same settings
    levy = ToyBench("levy", dims=1, n_samples=300, randseed=42, clustering_type="kmeans")
    ackley = ToyBench("ackley", dims=1, n_samples=300, randseed=42, clustering_type="kmeans")

    print(f"\nLevy optimum: {levy.ground_truth_opt:.4f}")
    print(f"Ackley optimum: {ackley.ground_truth_opt:.4f}")

    print("\n1D landscape plots show:")
    print("  - Top left: Function curve with sampled points colored by cluster")
    print("  - Top right: Clean function view with optimum location")
    print("  - Bottom left: Distribution of samples along x-axis")
    print("  - Bottom right: Value distribution by cluster (violin plot)")


if __name__ == "__main__":
    # Run examples
    example_levy_1d()
    example_ackley_1d()
    compare_functions()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - benchmarks/TOY/visualizations/example_levy_1d_landscape.png")
    print("  - benchmarks/TOY/visualizations/example_ackley_1d_landscape.png")
