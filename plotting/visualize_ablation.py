#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.edu
# Created Time : 2025-11-27
# File Name: visualize_ablation.py
# Description: Visualization utilities for noise ablation study results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle


def load_ablation_results(results_dir):
    """Load all ablation results from directory"""

    # Load aggregated CSV if available
    csv_path = os.path.join(results_dir, "all_results.csv")
    if os.path.exists(csv_path):
        results_df = pd.read_csv(csv_path)
        print(f"Loaded {len(results_df)} results from {csv_path}")
        return results_df

    # Otherwise, load individual pickle files
    pkl_files = list(Path(results_dir).glob("*.pkl"))
    if not pkl_files:
        raise ValueError(f"No results found in {results_dir}")

    all_results = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)

            # Compute true gap if clean_opt is available
            clean_opt = result.get('clean_opt', result.get('ground_truth_opt'))
            final_best_y = result['final_best_y']
            true_gap = (final_best_y - clean_opt) if final_best_y is not None and clean_opt is not None else result.get('final_gap')

            all_results.append({
                'function': result['function'],
                'noise_type': result['noise_type'],
                'noise_scale': result['noise_scale'],
                'seed': result['seed'],
                'dims': result.get('dims', 10),
                'final_best_y': final_best_y,
                'ground_truth_opt': result['ground_truth_opt'],
                'clean_opt': clean_opt,
                'final_gap': true_gap,  # Gap to true clean optimum
            })

    results_df = pd.DataFrame(all_results)
    print(f"Loaded {len(results_df)} results from {len(pkl_files)} pickle files")
    return results_df


def plot_noise_robustness(results_df, save_path=None, metric='final_gap'):
    """
    Plot noise robustness: performance vs noise scale

    Args:
        results_df: DataFrame with columns [function, noise_type, noise_scale, seed, metric]
        save_path: Path to save figure (if None, display only)
        metric: Metric to plot ('final_gap' or 'final_best_y')
    """

    functions = results_df['function'].unique()
    noise_types = results_df['noise_type'].unique()

    n_functions = len(functions)
    if n_functions == 0:
        print("No functions to plot")
        return None

    # Adjust figure size based on number of functions
    fig, axes = plt.subplots(1, n_functions, figsize=(7*n_functions, 5))
    if n_functions == 1:
        axes = [axes]

    for idx, function in enumerate(functions):
        ax = axes[idx]

        for noise_type in noise_types:
            # Filter data
            data = results_df[
                (results_df['function'] == function) &
                (results_df['noise_type'] == noise_type)
            ]

            # Group by noise scale
            grouped = data.groupby('noise_scale')[metric].agg(['mean', 'std'])

            # Plot with error bars
            ax.errorbar(
                grouped.index,
                grouped['mean'],
                yerr=grouped['std'],
                marker='o',
                label=noise_type,
                capsize=5,
                linewidth=2,
                markersize=8
            )

        ax.set_xlabel('Noise Scale (relative to target std)', fontsize=12)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_title(f'{function.upper()} Function', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def plot_noise_type_comparison(results_df, save_path=None):
    """
    Compare different noise types using box plots

    Args:
        results_df: DataFrame with ablation results
        save_path: Path to save figure
    """

    functions = results_df['function'].unique()
    n_functions = len(functions)

    fig, axes = plt.subplots(1, n_functions, figsize=(7*n_functions, 5))
    if n_functions == 1:
        axes = [axes]

    for idx, function in enumerate(functions):
        ax = axes[idx]

        # Filter data for this function
        data = results_df[results_df['function'] == function]

        # Create box plot
        sns.boxplot(
            data=data,
            x='noise_scale',
            y='final_gap',
            hue='noise_type',
            ax=ax,
            palette='Set2'
        )

        ax.set_xlabel('Noise Scale', fontsize=12)
        ax.set_ylabel('Final Gap to Optimum', fontsize=12)
        ax.set_title(f'{function.upper()} Function', fontsize=14, fontweight='bold')
        ax.legend(title='Noise Type', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def plot_convergence_curves(results_dir, save_path=None):
    """
    Plot convergence curves for different noise levels

    Args:
        results_dir: Directory containing pickle files with trace_y
        save_path: Path to save figure
    """

    # Load pickle files to get trace_y
    pkl_files = list(Path(results_dir).glob("*.pkl"))
    if not pkl_files:
        raise ValueError(f"No pickle files found in {results_dir}")

    # Organize results by configuration
    results_by_config = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)

        key = (result['function'], result['noise_type'], result['noise_scale'])
        if key not in results_by_config:
            results_by_config[key] = []

        results_by_config[key].append(result['trace_y'])

    # Get unique functions and noise types
    functions = sorted(set(k[0] for k in results_by_config.keys()))
    noise_types = sorted(set(k[1] for k in results_by_config.keys()))

    # Create subplots
    fig, axes = plt.subplots(len(noise_types), len(functions),
                            figsize=(7*len(functions), 5*len(noise_types)))

    if len(noise_types) == 1 and len(functions) == 1:
        axes = np.array([[axes]])
    elif len(noise_types) == 1:
        axes = axes.reshape(1, -1)
    elif len(functions) == 1:
        axes = axes.reshape(-1, 1)

    # Plot convergence curves
    for i, noise_type in enumerate(noise_types):
        for j, function in enumerate(functions):
            ax = axes[i, j]

            # Get all noise scales for this function and noise type
            configs = [(f, nt, ns) for f, nt, ns in results_by_config.keys()
                      if f == function and nt == noise_type]
            noise_scales = sorted(set(ns for _, _, ns in configs))

            # Color map for noise scales
            colors = plt.cm.viridis(np.linspace(0, 1, len(noise_scales)))

            for ns_idx, noise_scale in enumerate(noise_scales):
                key = (function, noise_type, noise_scale)
                if key not in results_by_config:
                    continue

                traces = results_by_config[key]

                # Convert traces to numpy arrays
                max_len = max(len(trace) for trace in traces)
                traces_padded = []
                for trace in traces:
                    if len(trace) < max_len:
                        # Pad with last value
                        padded = np.concatenate([trace, [trace[-1]]*(max_len - len(trace))])
                    else:
                        padded = trace
                    traces_padded.append(padded)

                traces_array = np.array(traces_padded)

                # Plot mean and std
                mean_trace = np.mean(traces_array, axis=0)
                std_trace = np.std(traces_array, axis=0)

                iterations = np.arange(len(mean_trace))

                ax.plot(iterations, mean_trace, color=colors[ns_idx],
                       label=f'noise={noise_scale}', linewidth=2)
                ax.fill_between(iterations,
                               mean_trace - std_trace,
                               mean_trace + std_trace,
                               color=colors[ns_idx], alpha=0.2)

            ax.set_xlabel('BO Iteration', fontsize=11)
            ax.set_ylabel('Best Objective Value', fontsize=11)
            ax.set_title(f'{function.upper()} - {noise_type}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def plot_gap_and_regret_trajectories(results_dir, save_path=None):
    """
    Plot GAP and regret trajectories for different noise levels

    Args:
        results_dir: Directory containing pickle files with trace_gap and trace_regret
        save_path: Path to save figure
    """

    # Load pickle files
    pkl_files = list(Path(results_dir).glob("*.pkl"))
    if not pkl_files:
        raise ValueError(f"No pickle files found in {results_dir}")

    # Organize results by configuration
    results_by_config = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)

        key = (result['function'], result['noise_type'], result['noise_scale'])
        if key not in results_by_config:
            results_by_config[key] = {'gap': [], 'regret': []}

        results_by_config[key]['gap'].append(result.get('trace_gap', []))
        results_by_config[key]['regret'].append(result.get('trace_regret', []))

    # Get unique functions and noise types
    functions = sorted(set(k[0] for k in results_by_config.keys()))
    noise_types = sorted(set(k[1] for k in results_by_config.keys()))

    # Create subplots: 2 rows (GAP, Regret) x n_functions columns
    fig, axes = plt.subplots(2, len(functions), figsize=(7*len(functions), 10))
    if len(functions) == 1:
        axes = axes.reshape(-1, 1)

    for j, function in enumerate(functions):
        for i, metric in enumerate(['gap', 'regret']):
            ax = axes[i, j]

            # Get all noise scales for this function
            configs = [(f, nt, ns) for f, nt, ns in results_by_config.keys()
                      if f == function]
            noise_scales = sorted(set(ns for _, _, ns in configs))

            # Color map
            colors = plt.cm.viridis(np.linspace(0, 1, len(noise_scales)))

            for ns_idx, noise_scale in enumerate(noise_scales):
                # Combine all noise types for this scale (or filter by type if needed)
                all_traces = []
                for noise_type in noise_types:
                    key = (function, noise_type, noise_scale)
                    if key in results_by_config:
                        all_traces.extend(results_by_config[key][metric])

                # Filter out empty traces
                all_traces = [trace for trace in all_traces if len(trace) > 0]

                if not all_traces:
                    continue

                # Pad traces to same length
                max_len = max(len(trace) for trace in all_traces)
                if max_len == 0:
                    continue

                traces_padded = []
                for trace in all_traces:
                    if len(trace) < max_len:
                        padded = np.concatenate([trace, [trace[-1]]*(max_len - len(trace))])
                    else:
                        padded = trace
                    traces_padded.append(padded)

                traces_array = np.array(traces_padded)
                mean_trace = np.mean(traces_array, axis=0)
                std_trace = np.std(traces_array, axis=0)

                iterations = np.arange(len(mean_trace))

                ax.plot(iterations, mean_trace, color=colors[ns_idx],
                       label=f'noise={noise_scale}', linewidth=2)
                ax.fill_between(iterations,
                               mean_trace - std_trace,
                               mean_trace + std_trace,
                               color=colors[ns_idx], alpha=0.2)

            ax.set_xlabel('BO Iteration', fontsize=11)
            ylabel = 'GAP (Normalized Progress)' if metric == 'gap' else 'Regret (Distance to Optimum)'
            ax.set_ylabel(ylabel, fontsize=11)
            title = f'{function.upper()} - {metric.upper()}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

            # Add reference line for GAP=1 (optimal)
            if metric == 'gap':
                ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Optimal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def create_summary_table(results_df, save_path=None):
    """
    Create a summary table with mean ± std for each configuration

    Args:
        results_df: DataFrame with ablation results
        save_path: Path to save table (CSV or LaTeX)
    """

    # Group by configuration
    summary = results_df.groupby(['function', 'noise_type', 'noise_scale']).agg({
        'final_gap': ['mean', 'std', 'min', 'max'],
        'final_best_y': ['mean', 'std']
    }).round(4)

    # Create formatted string columns
    summary_formatted = results_df.groupby(['function', 'noise_type', 'noise_scale']).apply(
        lambda x: pd.Series({
            'Final Gap': f"{x['final_gap'].mean():.4f} ± {x['final_gap'].std():.4f}",
            'Best Y': f"{x['final_best_y'].mean():.4f} ± {x['final_best_y'].std():.4f}",
            'N Runs': len(x)
        })
    ).reset_index()

    print("\nSummary Table:")
    print(summary_formatted.to_string(index=False))

    if save_path:
        if save_path.endswith('.tex'):
            summary_formatted.to_latex(save_path, index=False)
        else:
            summary_formatted.to_csv(save_path, index=False)
        print(f"\nSaved summary table to {save_path}")

    return summary_formatted


def visualize_all(results_dir, output_dir=None, config_path=None):
    """
    Generate all visualizations for ablation study

    Args:
        results_dir: Directory containing ablation results
        output_dir: Directory to save visualizations (default: results_dir/figures)
        config_path: Path to config file to filter functions (optional)
    """

    if output_dir is None:
        output_dir = os.path.join(results_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    results_df = load_ablation_results(results_dir)

    # Filter by config if provided
    if config_path is None:
        config_path = os.path.join(results_dir, "config.yaml")

    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Filter to only include functions in config
        if 'ablation' in config and 'functions' in config['ablation']:
            allowed_functions = config['ablation']['functions']
            results_df = results_df[results_df['function'].isin(allowed_functions)]
            print(f"Filtered to functions: {allowed_functions}")

    if len(results_df) == 0:
        print("No results to visualize after filtering!")
        return

    print(f"\nGenerating visualizations...")

    # 1. Noise robustness plot
    print("  - Noise robustness (final gap)")
    plot_noise_robustness(
        results_df,
        save_path=os.path.join(output_dir, "noise_robustness_gap.png"),
        metric='final_gap'
    )

    print("  - Noise robustness (final best y)")
    plot_noise_robustness(
        results_df,
        save_path=os.path.join(output_dir, "noise_robustness_best_y.png"),
        metric='final_best_y'
    )

    # 2. Noise type comparison
    print("  - Noise type comparison")
    plot_noise_type_comparison(
        results_df,
        save_path=os.path.join(output_dir, "noise_type_comparison.png")
    )

    # 3. Convergence curves
    print("  - Convergence curves")
    plot_convergence_curves(
        results_dir,
        save_path=os.path.join(output_dir, "convergence_curves.png")
    )

    # 4. GAP and Regret trajectories
    print("  - GAP and Regret trajectories")
    plot_gap_and_regret_trajectories(
        results_dir,
        save_path=os.path.join(output_dir, "gap_regret_trajectories.png")
    )

    # 5. Summary table
    print("  - Summary table")
    create_summary_table(
        results_df,
        save_path=os.path.join(output_dir, "summary_table.csv")
    )

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize noise ablation study results')
    parser.add_argument('results_dir', type=str,
                       help='Directory containing ablation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualizations (default: results_dir/figures)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file for filtering functions (default: results_dir/config.yaml)')

    args = parser.parse_args()

    visualize_all(args.results_dir, args.output_dir, args.config)
