#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Visualization utilities for comparing baselines (GP, Laplace) with LLMAT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_baseline_results(baseline_dir):
    """Load baseline results from CSV"""
    csv_path = os.path.join(baseline_dir, "all_baseline_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise ValueError(f"Results file not found: {csv_path}")


def load_llmat_results(llmat_dir):
    """Load LLMAT results from CSV"""
    csv_path = os.path.join(llmat_dir, "all_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Add method column
        df['method'] = 'llmat'
        return df
    else:
        raise ValueError(f"Results file not found: {csv_path}")


def load_all_results(baseline_dir, llmat_dir):
    """Load and combine all results"""
    baseline_df = load_baseline_results(baseline_dir)
    llmat_df = load_llmat_results(llmat_dir)

    # Rename baseline_model to method for consistency
    baseline_df = baseline_df.rename(columns={'baseline_model': 'method'})

    # Combine
    all_df = pd.concat([baseline_df, llmat_df], ignore_index=True)
    return all_df


def plot_method_comparison(results_df, metric='final_gap', save_path=None):
    """
    Compare different methods (GP, Laplace, LLMAT) across noise levels

    Note: Baselines (GP, Laplace) are only run on clean data (noise_scale=0.0)
          They appear as horizontal lines across all noise levels for comparison

    Args:
        results_df: DataFrame with columns [method, function, noise_scale, metric]
        metric: Metric to plot ('final_gap' or 'final_regret')
        save_path: Path to save figure
    """
    functions = results_df['function'].unique()
    noise_types = results_df['noise_type'].unique()

    n_functions = len(functions)
    n_noise_types = len(noise_types)

    if n_functions == 0:
        print("No functions to plot")
        return None

    # Create subplots
    fig, axes = plt.subplots(n_noise_types, n_functions,
                            figsize=(7*n_functions, 5*n_noise_types))

    if n_functions == 1 and n_noise_types == 1:
        axes = np.array([[axes]])
    elif n_functions == 1:
        axes = axes.reshape(-1, 1)
    elif n_noise_types == 1:
        axes = axes.reshape(1, -1)

    # Color palette for methods
    methods = sorted(results_df['method'].unique())
    colors = sns.color_palette("Set2", len(methods))
    method_colors = dict(zip(methods, colors))

    for noise_idx, noise_type in enumerate(noise_types):
        for func_idx, function in enumerate(functions):
            ax = axes[noise_idx, func_idx] if n_noise_types > 1 or n_functions > 1 else axes[0, 0]

            # Get all noise scales for LLMAT (to set x-axis range)
            llmat_data = results_df[
                (results_df['method'] == 'llmat') &
                (results_df['function'] == function) &
                (results_df['noise_type'] == noise_type)
            ]
            noise_scales_range = sorted(llmat_data['noise_scale'].unique()) if len(llmat_data) > 0 else [0.0]

            for method in methods:
                # Filter data
                data = results_df[
                    (results_df['method'] == method) &
                    (results_df['function'] == function) &
                    (results_df['noise_type'] == noise_type)
                ]

                if len(data) == 0:
                    continue

                # Group by noise scale
                grouped = data.groupby('noise_scale')[metric].agg(['mean', 'std', 'count'])

                # For baselines (only at noise_scale=0.0), plot as horizontal reference line
                if method in ['gp', 'laplace'] and len(grouped) == 1 and 0.0 in grouped.index:
                    baseline_mean = grouped.loc[0.0, 'mean']
                    baseline_std = grouped.loc[0.0, 'std']

                    # Plot horizontal line across all noise scales
                    ax.axhline(y=baseline_mean, color=method_colors[method],
                              linestyle='--', linewidth=2, label=f'{method.upper()} (clean data)',
                              alpha=0.7)
                    # Add shaded region for std
                    ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
                              color=method_colors[method], alpha=0.1)
                else:
                    # Plot LLMAT normally with error bars
                    ax.errorbar(
                        grouped.index,
                        grouped['mean'],
                        yerr=grouped['std'],
                        label=method.upper(),
                        marker='o',
                        markersize=6,
                        linewidth=2,
                        capsize=5,
                        color=method_colors[method]
                    )

            # Formatting
            ax.set_xlabel('Noise Scale', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{function.upper()} - {noise_type}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


def plot_trajectory_comparison(baseline_dir, llmat_dir, function='levy',
                               noise_scale=0.0, noise_type='gaussian', save_path=None):
    """
    Compare GAP and regret trajectories across methods

    Args:
        baseline_dir: Directory with baseline results
        llmat_dir: Directory with LLMAT results
        function: Function name
        noise_scale: Noise scale to compare
        noise_type: Noise type
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    methods = ['gp', 'laplace', 'llmat']
    colors = {'gp': 'blue', 'laplace': 'green', 'llmat': 'red'}

    for method in methods:
        # Load trace files
        if method == 'llmat':
            pattern = f"{llmat_dir}/{function}/noise-{noise_type}-{noise_scale}/llmat/**/trace_y_his_*.npy"
        else:
            pattern = f"{baseline_dir}/{function}/noise-{noise_type}-{noise_scale}/{method}/trace_y_his_*.npy"

        trace_files = list(Path().glob(pattern))

        if not trace_files:
            print(f"No traces found for {method} at pattern: {pattern}")
            continue

        # Load all traces and compute cumulative best
        all_gaps = []
        all_regrets = []

        # Determine clean optimum (0 for Levy)
        clean_opt = 0.0  # TODO: Load from benchmark

        for trace_file in trace_files:
            trace = np.load(trace_file)
            cumulative_best = np.minimum.accumulate(trace)

            # Compute GAP and regret
            initial_best = cumulative_best[0]
            gap = []
            regret = []

            for best_y in cumulative_best:
                # GAP
                if clean_opt != initial_best:
                    g = (initial_best - best_y) / (initial_best - clean_opt)
                else:
                    g = 1.0
                gap.append(np.nan_to_num(g, nan=1.0))

                # Regret
                r = abs(best_y - clean_opt)
                regret.append(r)

            all_gaps.append(np.array(gap))
            all_regrets.append(np.array(regret))

        if not all_gaps:
            continue

        # Pad to same length
        max_len = max(len(g) for g in all_gaps)
        gaps_padded = []
        regrets_padded = []

        for gap, regret in zip(all_gaps, all_regrets):
            if len(gap) < max_len:
                gap_padded = np.concatenate([gap, [gap[-1]]*(max_len - len(gap))])
                regret_padded = np.concatenate([regret, [regret[-1]]*(max_len - len(regret))])
            else:
                gap_padded = gap
                regret_padded = regret
            gaps_padded.append(gap_padded)
            regrets_padded.append(regret_padded)

        # Compute mean and std
        gap_mean = np.mean(gaps_padded, axis=0)
        gap_std = np.std(gaps_padded, axis=0)
        regret_mean = np.mean(regrets_padded, axis=0)
        regret_std = np.std(regrets_padded, axis=0)

        iterations = np.arange(len(gap_mean))

        # Plot GAP
        ax = axes[0]
        ax.plot(iterations, gap_mean, color=colors[method],
               label=method.upper(), linewidth=2)
        ax.fill_between(iterations, gap_mean - gap_std, gap_mean + gap_std,
                       color=colors[method], alpha=0.2)

        # Plot Regret
        ax = axes[1]
        ax.plot(iterations, regret_mean, color=colors[method],
               label=method.upper(), linewidth=2)
        ax.fill_between(iterations, regret_mean - regret_std, regret_mean + regret_std,
                       color=colors[method], alpha=0.2)

    # Format GAP plot
    axes[0].set_xlabel('BO Iteration', fontsize=12)
    axes[0].set_ylabel('GAP (Normalized Progress)', fontsize=12)
    axes[0].set_title(f'{function.upper()} - GAP Trajectory (noise={noise_scale})',
                     fontsize=14, fontweight='bold')
    axes[0].axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Optimal')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Format Regret plot
    axes[1].set_xlabel('BO Iteration', fontsize=12)
    axes[1].set_ylabel('Regret (Distance to Optimum)', fontsize=12)
    axes[1].set_title(f'{function.upper()} - Regret Trajectory (noise={noise_scale})',
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory comparison to {save_path}")

    return fig


def print_summary_table(results_df):
    """Print summary statistics table"""
    print("\n" + "="*80)
    print("Summary Statistics: Mean ± Std across all seeds")
    print("="*80)

    methods = sorted(results_df['method'].unique())
    functions = sorted(results_df['function'].unique())
    noise_scales = sorted(results_df['noise_scale'].unique())

    for function in functions:
        print(f"\n{function.upper()}:")
        print("-" * 80)
        print(f"{'Method':<15} {'Noise':<8} {'GAP':<20} {'Regret':<20}")
        print("-" * 80)

        for method in methods:
            for noise_scale in noise_scales:
                data = results_df[
                    (results_df['method'] == method) &
                    (results_df['function'] == function) &
                    (results_df['noise_scale'] == noise_scale)
                ]

                if len(data) > 0:
                    gap_mean = data['final_gap'].mean()
                    gap_std = data['final_gap'].std()
                    regret_mean = data['final_regret'].mean()
                    regret_std = data['final_regret'].std()

                    print(f"{method.upper():<15} {noise_scale:<8.2f} "
                          f"{gap_mean:.4f} ± {gap_std:.4f}     "
                          f"{regret_mean:.4f} ± {regret_std:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize baseline comparison')
    parser.add_argument('--baseline_dir', type=str, default='results/baselines_toy',
                       help='Directory with baseline results')
    parser.add_argument('--llmat_dir', type=str, default='results/ablation',
                       help='Directory with LLMAT results')
    parser.add_argument('--output_dir', type=str, default='results/comparisons',
                       help='Directory to save comparison plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    print("Loading results...")
    results_df = load_all_results(args.baseline_dir, args.llmat_dir)

    # Print summary
    print_summary_table(results_df)

    # Plot method comparison for GAP
    print("\nGenerating GAP comparison plot...")
    plot_method_comparison(
        results_df,
        metric='final_gap',
        save_path=os.path.join(args.output_dir, 'method_comparison_gap.png')
    )

    # Plot method comparison for Regret
    print("Generating Regret comparison plot...")
    plot_method_comparison(
        results_df,
        metric='final_regret',
        save_path=os.path.join(args.output_dir, 'method_comparison_regret.png')
    )

    # Plot trajectory comparison for noise_scale=0.0
    print("Generating trajectory comparison...")
    plot_trajectory_comparison(
        args.baseline_dir,
        args.llmat_dir,
        function='levy',
        noise_scale=0.0,
        save_path=os.path.join(args.output_dir, 'trajectory_comparison_levy_noise0.png')
    )

    print(f"\nAll plots saved to {args.output_dir}")
