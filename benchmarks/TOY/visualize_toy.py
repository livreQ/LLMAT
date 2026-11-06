#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-25
# File Name: visualize_toy.py
# Description: Visualization script for toy benchmarks
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from benchmarks.TOY.toy_benchmarks import ToyBench
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_cluster_distribution(bench, save_path=None):
    """Plot distribution of target values across clusters"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    cluster_col = bench._get_cluster_col()
    clusters = sorted(bench.dataset[cluster_col].unique())

    # Plot 1: KDE plot by cluster
    ax = axes[0, 0]
    sns.kdeplot(data=bench.dataset, x=bench.target_col, hue=cluster_col, fill=True, common_norm=False, alpha=0.3, linewidth=2, ax=ax, palette='tab10')

    # Add vertical line at optimum
    max_value = bench.dataset.loc[bench.ground_truth_opt_id][bench.target_col]
    ax.axvline(x=max_value, color='red', linestyle='--', linewidth=2, label='Optimum')

    ax.set_xlabel(f'{bench.target_col} ({"min" if not bench.maximization else "max"})', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of {bench.target_col} by Cluster\n{bench.data_name.upper()} ({bench.dims}D, {len(bench.dataset)} samples)', fontsize=14, fontweight='bold')
    ax.legend(title='Cluster', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot by cluster
    ax = axes[0, 1]
    box_data = [bench.dataset[bench.dataset[cluster_col] == c][bench.target_col].values for c in clusters]
    bp = ax.boxplot(box_data, labels=clusters, patch_artist=True, widths=0.6)

    # Color boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Highlight optimal cluster
    optimal_cluster = bench.ground_truth_opt_cluster
    if optimal_cluster in clusters:
        bp['boxes'][list(clusters).index(optimal_cluster)].set_edgecolor('red')
        bp['boxes'][list(clusters).index(optimal_cluster)].set_linewidth(3)

    ax.axhline(y=max_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimum')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel(f'{bench.target_col}', fontsize=12)
    ax.set_title('Box Plot by Cluster', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Histogram of all values
    ax = axes[1, 0]
    ax.hist(bench.dataset[bench.target_col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=max_value, color='red', linestyle='--', linewidth=2, label=f'Optimum: {max_value:.2f}')
    ax.set_xlabel(f'{bench.target_col}', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Cluster statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    # Compute statistics
    stats_data = []
    stats_data.append(['Cluster', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Has Opt'])

    for c in clusters:
        cluster_data = bench.dataset[bench.dataset[cluster_col] == c][bench.target_col]
        count = len(cluster_data)
        mean = cluster_data.mean()
        std = cluster_data.std()
        min_val = cluster_data.min()
        max_val = cluster_data.max()
        has_opt = '✓' if c in bench.ground_truth_opt_clusters else ''

        stats_data.append([f'{c}', f'{count}', f'{mean:.2f}', f'{std:.2f}', f'{min_val:.2f}', f'{max_val:.2f}', has_opt])

    # Add overall statistics
    stats_data.append(['─' * 8] * 7)
    stats_data.append([
        'Overall', f'{len(bench.dataset)}', f'{bench.dataset[bench.target_col].mean():.2f}', f'{bench.dataset[bench.target_col].std():.2f}',
        f'{bench.dataset[bench.target_col].min():.2f}', f'{bench.dataset[bench.target_col].max():.2f}', '✓'
    ])

    table = ax.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight optimal cluster row
    if optimal_cluster in clusters:
        opt_row = list(clusters).index(optimal_cluster) + 1
        for i in range(7):
            table[(opt_row, i)].set_facecolor('#FFE5E5')

    ax.set_title('Cluster Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.show()
    return fig


def plot_1d_function_landscape(bench, n_grid=200, save_path=None):
    """Plot 1D landscape for 1D functions"""
    if bench.dims != 1:
        print(f"Warning: Function has {bench.dims} dimensions, not 1. Skipping 1D landscape plot.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Get function bounds
    if bench.data_name == "levy":
        from benchmarks.TOY.toy_benchmarks import Levy
        func = Levy(dims=1)
    elif bench.data_name == "ackley":
        from benchmarks.TOY.toy_benchmarks import Ackley
        func = Ackley(dims=1)
    else:
        print(f"Unknown function: {bench.data_name}")
        return None

    # Create grid
    x_vals = np.linspace(func.lb[0], func.ub[0], n_grid)
    y_vals = np.array([func(np.array([x])) for x in x_vals])

    # Plot 1: Function landscape with sampled points
    ax = axes[0, 0]
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Function', alpha=0.8)

    # Plot sampled points colored by cluster
    cluster_col = bench._get_cluster_col()
    clusters = sorted(bench.dataset[cluster_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    for c, color in zip(clusters, colors):
        cluster_data = bench.dataset[bench.dataset[cluster_col] == c]
        x_coords = [x[0] for x in cluster_data['x']]
        y_coords = cluster_data[bench.target_col].values
        ax.scatter(x_coords, y_coords, c=[color], s=50, alpha=0.7, edgecolors='black', linewidth=0.5, label=f'Cluster {c}', zorder=5)

    # Mark optimum
    opt_x = bench.dataset.iloc[bench.ground_truth_opt_id]['x'][0]
    opt_y = bench.ground_truth_opt
    ax.scatter(opt_x, opt_y, c='red', s=400, marker='*', edgecolors='white', linewidth=2, label='Optimum', zorder=10)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Function Value', fontsize=12)
    ax.set_title(f'{bench.data_name.upper()} Function Landscape with Sampled Points\n'
                 f'({len(bench.dataset)} samples, {len(clusters)} clusters)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: Function landscape only (zoomed to show details)
    ax = axes[0, 1]
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.8)
    ax.axvline(x=opt_x, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Optimum')
    ax.axhline(y=opt_y, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Function Value', fontsize=12)
    ax.set_title(f'{bench.data_name.upper()} Function (Clean View)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Sampled points distribution along x-axis
    ax = axes[1, 0]

    # Histogram of sampled x values
    all_x = [x[0] for x in bench.dataset['x']]
    ax.hist(all_x, bins=30, alpha=0.4, color='gray', edgecolor='black', label='All samples')

    # Overlay cluster-specific histograms
    for c, color in zip(clusters, colors):
        cluster_data = bench.dataset[bench.dataset[cluster_col] == c]
        x_coords = [x[0] for x in cluster_data['x']]
        ax.hist(x_coords, bins=30, alpha=0.5, color=color, edgecolor='black', label=f'Cluster {c}')

    ax.axvline(x=opt_x, color='red', linestyle='--', linewidth=2, label='Optimum')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Sampled Points', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Function value distribution by cluster
    ax = axes[1, 1]

    # Violin plot
    cluster_data_list = []
    cluster_labels = []
    for c in clusters:
        cluster_vals = bench.dataset[bench.dataset[cluster_col] == c][bench.target_col].values
        cluster_data_list.append(cluster_vals)
        cluster_labels.append(f'C{c}')

    parts = ax.violinplot(cluster_data_list, positions=range(len(clusters)), showmeans=True, showmedians=True, widths=0.7)

    # Color the violin plots
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.axhline(y=opt_y, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimum')
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(cluster_labels)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Function Value', fontsize=12)
    ax.set_title('Value Distribution by Cluster', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        landscape_path = f"{base}_landscape{ext}"
        os.makedirs(os.path.dirname(landscape_path) if os.path.dirname(landscape_path) else '.', exist_ok=True)
        plt.savefig(landscape_path, dpi=300, bbox_inches='tight')
        print(f"Saved 1D landscape figure to: {landscape_path}")

    plt.show()
    return fig


def plot_2d_function_landscape(bench, n_grid=50, save_path=None):
    """Plot 2D landscape for 2D functions"""
    if bench.dims != 2:
        print(f"Warning: Function has {bench.dims} dimensions, not 2. Skipping landscape plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get function bounds
    if bench.data_name == "levy":
        from benchmarks.TOY.toy_benchmarks import Levy
        func = Levy(dims=2)
    elif bench.data_name == "ackley":
        from benchmarks.TOY.toy_benchmarks import Ackley
        func = Ackley(dims=2)
    else:
        print(f"Unknown function: {bench.data_name}")
        return None

    # Create grid
    x1 = np.linspace(func.lb[0], func.ub[0], n_grid)
    x2 = np.linspace(func.lb[1], func.ub[1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate function on grid
    Z = np.zeros_like(X1)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = func(np.array([X1[i, j], X2[i, j]]))

    # Plot 1: Contour plot
    ax = axes[0]
    contour = ax.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Function Value')

    # Plot sampled points
    cluster_col = bench._get_cluster_col()
    clusters = sorted(bench.dataset[cluster_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    for c, color in zip(clusters, colors):
        cluster_data = bench.dataset[bench.dataset[cluster_col] == c]
        x_coords = [x[0] for x in cluster_data['x']]
        y_coords = [x[1] for x in cluster_data['x']]
        ax.scatter(x_coords, y_coords, c=[color], s=30, alpha=0.6, edgecolors='black', linewidth=0.5, label=f'Cluster {c}')

    # Mark optimum
    opt_x = bench.dataset.iloc[bench.ground_truth_opt_id]['x']
    ax.scatter(opt_x[0], opt_x[1], c='red', s=300, marker='*', edgecolors='white', linewidth=2, label='Optimum', zorder=10)

    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(f'{bench.data_name.upper()} Function Landscape with Sampled Points', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: 3D surface
    ax = plt.subplot(122, projection='3d')
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7, edgecolor='none', antialiased=True)

    # Plot sampled points
    for c, color in zip(clusters, colors):
        cluster_data = bench.dataset[bench.dataset[cluster_col] == c]
        x_coords = [x[0] for x in cluster_data['x']]
        y_coords = [x[1] for x in cluster_data['x']]
        z_coords = cluster_data[bench.target_col].values
        ax.scatter(x_coords, y_coords, z_coords, c=[color], s=20, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Mark optimum
    ax.scatter(opt_x[0], opt_x[1], bench.ground_truth_opt, c='red', s=200, marker='*', edgecolors='white', linewidth=2, zorder=10)

    ax.set_xlabel('x1', fontsize=10)
    ax.set_ylabel('x2', fontsize=10)
    ax.set_zlabel('Function Value', fontsize=10)
    ax.set_title(f'{bench.data_name.upper()} Function Surface', fontsize=14, fontweight='bold')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        landscape_path = f"{base}_landscape{ext}"
        os.makedirs(os.path.dirname(landscape_path) if os.path.dirname(landscape_path) else '.', exist_ok=True)
        plt.savefig(landscape_path, dpi=300, bbox_inches='tight')
        print(f"Saved landscape figure to: {landscape_path}")

    plt.show()
    return fig


def plot_cluster_comparison(bench1, bench2, save_path=None):
    """Compare cluster distributions between two benchmarks"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, bench in enumerate([bench1, bench2]):
        ax = axes[idx]
        cluster_col = bench._get_cluster_col()

        sns.kdeplot(data=bench.dataset, x=bench.target_col, hue=cluster_col, fill=True, common_norm=False, alpha=0.3, linewidth=2, ax=ax, palette='tab10')

        # Add vertical line at optimum
        max_value = bench.dataset.loc[bench.ground_truth_opt_id][bench.target_col]
        ax.axvline(x=max_value, color='red', linestyle='--', linewidth=2)

        ax.set_xlabel(f'{bench.target_col}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{bench.data_name.upper()} ({bench.dims}D, {len(bench.dataset)} samples)\n'
                     f'Optimum: {max_value:.2f}', fontsize=14, fontweight='bold')
        ax.legend(title='Cluster', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison figure to: {save_path}")

    plt.show()
    return fig


def main():
    """Main visualization routine"""
    print("=" * 70)
    print("TOY BENCHMARK VISUALIZATION")
    print("=" * 70)

    # Create output directory
    output_dir = "benchmarks/TOY/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize Levy function (1D)
    print("\n1. Visualizing Levy function (1D)...")
    levy_bench = ToyBench(data_name="levy", dims=1, n_samples=1000, randseed=42, clustering_type="kmeans", run_subset_only=False)

    save_path = f"{output_dir}/levy_1d_clusters.png"
    plot_cluster_distribution(levy_bench, save_path=save_path)

    print("\n1b. Visualizing Levy 1D landscape...")
    plot_1d_function_landscape(levy_bench, n_grid=200, save_path=save_path)

    # Visualize Ackley function (1D)
    print("\n2. Visualizing Ackley function (1D)...")
    ackley_bench = ToyBench(data_name="ackley", dims=1, n_samples=1000, randseed=42, clustering_type="kmeans", run_subset_only=False)

    save_path = f"{output_dir}/ackley_1d_clusters.png"
    plot_cluster_distribution(ackley_bench, save_path=save_path)

    print("\n2b. Visualizing Ackley 1D landscape...")
    plot_1d_function_landscape(ackley_bench, n_grid=200, save_path=save_path)

    # Compare the two functions
    print("\n3. Comparing Levy and Ackley...")
    save_path = f"{output_dir}/levy_vs_ackley_comparison.png"
    plot_cluster_comparison(levy_bench, ackley_bench, save_path=save_path)

    # Visualize 2D landscape
    print("\n4. Visualizing 2D function landscape...")
    levy_2d = ToyBench(data_name="levy", dims=2, n_samples=1000, randseed=42, clustering_type="kmeans", run_subset_only=False)

    save_path = f"{output_dir}/levy_2d_clusters.png"
    plot_cluster_distribution(levy_2d, save_path=save_path)
    plot_2d_function_landscape(levy_2d, n_grid=100, save_path=save_path)

    # Visualize Ackley 2D
    print("\n5. Visualizing Ackley 2D landscape...")
    ackley_2d = ToyBench(data_name="ackley", dims=2, n_samples=1000, randseed=42, clustering_type="kmeans", run_subset_only=False)

    save_path = f"{output_dir}/ackley_2d_clusters.png"
    plot_cluster_distribution(ackley_2d, save_path=save_path)
    plot_2d_function_landscape(ackley_2d, n_grid=100, save_path=save_path)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll figures saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - levy_1d_clusters.png")
    print(f"  - levy_1d_clusters_landscape.png")
    print(f"  - ackley_1d_clusters.png")
    print(f"  - ackley_1d_clusters_landscape.png")
    print(f"  - levy_vs_ackley_comparison.png")
    print(f"  - levy_2d_clusters.png")
    print(f"  - levy_2d_clusters_landscape.png")
    print(f"  - ackley_2d_clusters.png")
    print(f"  - ackley_2d_clusters_landscape.png")


if __name__ == "__main__":
    main()
