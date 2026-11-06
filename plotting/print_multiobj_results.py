#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Print multi-objective results for LAPEFT and LLMAT comparison.

Usage:
    python print_multiobj_results.py --dataset redox-mer --feature_model t5-base-chem-average --clustering kmeans
    python print_multiobj_results.py --dataset laser --feature_model t5-base-chem-average
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_results(base_path, method, dataset, feature_model, clustering=None):
    """
    Load results for a specific method, dataset, and feature model.

    Args:
        base_path: Base results directory
        method: 'multiobj_lapeft' or 'multiobj_llmat'
        dataset: Dataset name (e.g., 'redox-mer', 'laser')
        feature_model: Feature model name
        clustering: Clustering type for llmat (None for lapeft)

    Returns:
        dict: Dictionary with result arrays
    """
    path = Path(base_path) / dataset / feature_model / method
    if clustering and method == "multiobj_llmat":
        path = path / clustering

    if not path.exists():
        print(f"Path {path} does not exist")
        return None

    results = {}

    # Find all .npy files for this method
    npy_files = list(path.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {path}")
        return None

    # Group files by suffix (different seeds/configs)
    file_groups = {}
    for npy_file in npy_files:
        stem = npy_file.stem
        # Extract suffix (everything after the metric name)
        if "_trace_hypervolume_" in stem:
            suffix = stem.replace("trace_hypervolume_", "")
        elif "_trace_scalarized_best_y_" in stem:
            suffix = stem.replace("trace_scalarized_best_y_", "")
        elif "_trace_best_y_" in stem:
            suffix = stem.replace("trace_best_y_", "")
        elif "_trace_timing_" in stem:
            suffix = stem.replace("trace_timing_", "")
        elif "_timing_train_" in stem:
            suffix = stem.replace("timing_train_", "")
        elif "_timing_preds_" in stem:
            suffix = stem.replace("timing_preds_", "")
        else:
            continue

        if suffix not in file_groups:
            file_groups[suffix] = {}
        file_groups[suffix][stem.split('_')[0]] = npy_file

    # Load results for each group
    all_results = []
    for suffix, files in file_groups.items():
        result = {
            'suffix': suffix,
            'trace_hypervolume': None,
            'trace_scalarized_best_y': None,
            'trace_best_y': None,
            'trace_timing': None,
            'timing_train': None,
            'timing_preds': None
        }

        for metric, file_path in files.items():
            if metric == 'trace':
                if 'hypervolume' in str(file_path):
                    result['trace_hypervolume'] = np.load(file_path)
                elif 'scalarized' in str(file_path):
                    result['trace_scalarized_best_y'] = np.load(file_path)
                elif 'best' in str(file_path) and 'y' in str(file_path):
                    result['trace_best_y'] = np.load(file_path)
                elif 'timing' in str(file_path):
                    result['trace_timing'] = np.load(file_path)
            elif metric == 'timing':
                if 'train' in str(file_path):
                    result['timing_train'] = np.load(file_path)
                elif 'preds' in str(file_path):
                    result['timing_preds'] = np.load(file_path)

        # Try alternative loading for missing files
        if result['trace_hypervolume'] is None:
            hv_file = path / f"trace_hypervolume_{suffix}.npy"
            if hv_file.exists():
                result['trace_hypervolume'] = np.load(hv_file)

        if result['trace_scalarized_best_y'] is None:
            scalar_file = path / f"trace_scalarized_best_y_{suffix}.npy"
            if scalar_file.exists():
                result['trace_scalarized_best_y'] = np.load(scalar_file)

        if result['trace_best_y'] is None:
            best_file = path / f"trace_best_y_{suffix}.npy"
            if best_file.exists():
                result['trace_best_y'] = np.load(best_file)

        if result['trace_timing'] is None:
            timing_file = path / f"trace_timing_{suffix}.npy"
            if timing_file.exists():
                result['trace_timing'] = np.load(timing_file)

        all_results.append(result)

    return all_results


def plot_hypervolume_curves(results, method_name, dataset, feature_model, save_path=None):
    """
    Plot hypervolume curves for a single method.
    """
    if not results:
        print(f"No results to plot for {method_name}")
        return

    plt.figure(figsize=(10, 6))

    # Collect all hypervolume trajectories
    all_hv = []
    max_length = 0

    for result in results:
        if result['trace_hypervolume'] is not None:
            hv = result['trace_hypervolume']
            all_hv.append(hv)
            max_length = max(max_length, len(hv))

    if not all_hv:
        print(f"No hypervolume data to plot for {method_name}")
        return

    # Pad shorter trajectories with their final value
    padded_hv = []
    for hv in all_hv:
        if len(hv) < max_length:
            # Pad with the final value
            padded = np.pad(hv, (0, max_length - len(hv)), 'edge')
            padded_hv.append(padded)
        else:
            padded_hv.append(hv)

    hv_array = np.array(padded_hv)
    iterations = np.arange(max_length)

    # Plot individual runs (light color)
    for i, hv in enumerate(padded_hv):
        plt.plot(iterations, hv, alpha=0.3, color='blue', linewidth=1)

    # Plot mean curve with confidence interval
    mean_hv = np.mean(hv_array, axis=0)
    std_hv = np.std(hv_array, axis=0)

    plt.plot(iterations, mean_hv, color='blue', linewidth=3, label=f'{method_name} (Mean)')
    plt.fill_between(iterations, mean_hv - std_hv, mean_hv + std_hv,
                    alpha=0.3, color='blue', label='±1 Std Dev')

    plt.xlabel('BO Iteration')
    plt.ylabel('Hypervolume')
    plt.title(f'Hypervolume Progression - {method_name}\n{dataset}/{feature_model}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_comparison_curves(lapeft_results, llmat_results, dataset, feature_model, save_path=None):
    """
    Plot comparison of hypervolume curves between LAPEFT and LLMAT.
    """
    if not lapeft_results or not llmat_results:
        print("Need results from both methods for comparison")
        return

    plt.figure(figsize=(12, 8))

    methods_data = [
        ('LAPEFT', lapeft_results, 'red'),
        ('LLMAT', llmat_results, 'blue')
    ]

    for method_name, results, color in methods_data:
        # Collect all hypervolume trajectories
        all_hv = []
        max_length = 0

        for result in results:
            if result['trace_hypervolume'] is not None:
                hv = result['trace_hypervolume']
                all_hv.append(hv)
                max_length = max(max_length, len(hv))

        if not all_hv:
            continue

        # Pad shorter trajectories with their final value
        padded_hv = []
        for hv in all_hv:
            if len(hv) < max_length:
                padded = np.pad(hv, (0, max_length - len(hv)), 'edge')
                padded_hv.append(padded)
            else:
                padded_hv.append(hv)

        hv_array = np.array(padded_hv)
        iterations = np.arange(max_length)

        # Plot individual runs (very light)
        for i, hv in enumerate(padded_hv):
            plt.plot(iterations, hv, alpha=0.1, color=color, linewidth=0.5)

        # Plot mean curve with confidence interval
        mean_hv = np.mean(hv_array, axis=0)
        std_hv = np.std(hv_array, axis=0)

        plt.plot(iterations, mean_hv, color=color, linewidth=3,
                label=f'{method_name} (Mean ± Std)')
        plt.fill_between(iterations, mean_hv - std_hv, mean_hv + std_hv,
                        alpha=0.2, color=color)

    plt.xlabel('BO Iteration', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.title(f'Hypervolume Comparison: LAPEFT vs LLMAT\n{dataset}/{feature_model}',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add final values as text
    if lapeft_results and llmat_results:
        lapeft_final = [r['trace_hypervolume'][-1] for r in lapeft_results if r['trace_hypervolume'] is not None]
        llmat_final = [r['trace_hypervolume'][-1] for r in llmat_results if r['trace_hypervolume'] is not None]

        if lapeft_final and llmat_final:
            lapeft_mean = np.mean(lapeft_final)
            llmat_mean = np.mean(llmat_final)

            plt.axhline(y=lapeft_mean, color='red', linestyle='--', alpha=0.7,
                       label=f'LAPEFT Final: {lapeft_mean:.3f}')
            plt.axhline(y=llmat_mean, color='blue', linestyle='--', alpha=0.7,
                       label=f'LLMAT Final: {llmat_mean:.3f}')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def print_summary_stats(results, method_name, dataset, feature_model):
    """
    Print summary statistics for results.
    """
    if not results:
        print(f"No results found for {method_name} on {dataset}/{feature_model}")
        return

    print(f"\n{'='*60}")
    print(f"Results for {method_name.upper()} - {dataset}/{feature_model}")
    print(f"{'='*60}")

    # Extract final hypervolumes and scalarized best values
    final_hypervolumes = []
    final_scalarized = []
    all_hypervolumes = []
    all_scalarized = []

    for result in results:
        if result['trace_hypervolume'] is not None:
            final_hypervolumes.append(result['trace_hypervolume'][-1])
            all_hypervolumes.append(result['trace_hypervolume'])
        if result['trace_scalarized_best_y'] is not None:
            final_scalarized.append(result['trace_scalarized_best_y'][-1])
            all_scalarized.append(result['trace_scalarized_best_y'])

    if final_hypervolumes:
        print("Final Hypervolume:")
        print(f"  Mean: {np.mean(final_hypervolumes):.4f}")
        print(f"  Std:  {np.std(final_hypervolumes):.4f}")
        print(f"  Min:  {np.min(final_hypervolumes):.4f}")
        print(f"  Max:  {np.max(final_hypervolumes):.4f}")
        print(f"  Runs: {len(final_hypervolumes)}")

    if final_scalarized:
        print("\nFinal Scalarized Best:")
        print(f"  Mean: {np.mean(final_scalarized):.4f}")
        print(f"  Std:  {np.std(final_scalarized):.4f}")
        print(f"  Min:  {np.min(final_scalarized):.4f}")
        print(f"  Max:  {np.max(final_scalarized):.4f}")
        print(f"  Runs: {len(final_scalarized)}")

    # Show hypervolume progression
    if all_hypervolumes:
        print("\nHypervolume Progression (first 10 iterations):")
        hv_array = np.array(all_hypervolumes)
        for i in range(min(10, len(hv_array[0]))):
            mean_hv = np.mean(hv_array[:, i])
            std_hv = np.std(hv_array[:, i])
            print("4d")


def compare_methods(lapeft_results, llmat_results, dataset, feature_model):
    """
    Compare LAPEFT and LLMAT results.
    """
    if not lapeft_results or not llmat_results:
        print(f"Cannot compare - missing results for {dataset}/{feature_model}")
        return

    print(f"\n{'='*80}")
    print(f"COMPARISON: LAPEFT vs LLMAT - {dataset}/{feature_model}")
    print(f"{'='*80}")

    # Extract final values
    lapeft_hv = [r['trace_hypervolume'][-1] for r in lapeft_results if r['trace_hypervolume'] is not None]
    llmat_hv = [r['trace_hypervolume'][-1] for r in llmat_results if r['trace_hypervolume'] is not None]

    lapeft_scalar = [r['trace_scalarized_best_y'][-1] for r in lapeft_results if r['trace_scalarized_best_y'] is not None]
    llmat_scalar = [r['trace_scalarized_best_y'][-1] for r in llmat_results if r['trace_scalarized_best_y'] is not None]

    if lapeft_hv and llmat_hv:
        lapeft_mean_hv = np.mean(lapeft_hv)
        llmat_mean_hv = np.mean(llmat_hv)

        print("Final Hypervolume:")
        print(".4f")
        print(".4f")
        if llmat_mean_hv > lapeft_mean_hv:
            improvement = (llmat_mean_hv - lapeft_mean_hv) / lapeft_mean_hv * 100
            print(".1f")
        else:
            improvement = (lapeft_mean_hv - llmat_mean_hv) / llmat_mean_hv * 100
            print(".1f")

    if lapeft_scalar and llmat_scalar:
        lapeft_mean_scalar = np.mean(lapeft_scalar)
        llmat_mean_scalar = np.mean(llmat_scalar)

        print("\nFinal Scalarized Best:")
        print(".4f")
        print(".4f")


def main():
    parser = argparse.ArgumentParser(
        description='Print multi-objective results for LAPEFT and LLMAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python print_multiobj_results.py --dataset redox-mer --feature_model t5-base-chem-average --clustering kmeans
    python print_multiobj_results.py --dataset laser --feature_model t5-base-chem-average --plot --save_plots
    python print_multiobj_results.py --results_dir /custom/path --dataset redox-mer --feature_model t5-base-chem-average
        """)
    parser.add_argument('--results_dir', type=str, default='results/multiobj',
                        help='Results directory containing multiobj results (default: results/multiobj)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['redox-mer', 'laser'],
                        help='Dataset name')
    parser.add_argument('--feature_model', type=str, default='t5-base-chem-average',
                        help='Feature model name')
    parser.add_argument('--clustering', type=str, default='kmeans',
                        choices=['kmeans', 'llms'],
                        help='Clustering type for LLMAT')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--plot_dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')

    args = parser.parse_args()

    # Check if results directory exists
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{args.results_dir}' does not exist")
        print("Make sure you're running this script from the AlphaLLMBO root directory")
        print("The default path is 'results/multiobj' - use --results_dir to specify a different location")
        sys.exit(1)

    # Load results
    print(f"Loading results for {args.dataset}/{args.feature_model}...")

    try:
        lapeft_results = load_results(args.results_dir, 'multiobj_lapeft',
                                    args.dataset, args.feature_model)

        llmat_results = load_results(args.results_dir, 'multiobj_llmat',
                                        args.dataset, args.feature_model, args.clustering)

        # Print individual results
        print_summary_stats(lapeft_results, 'LAPEFT', args.dataset, args.feature_model)
        print_summary_stats(llmat_results, 'LLMAT', args.dataset, args.feature_model)

        # Generate plots if requested
        if args.plot:
            print("\nGenerating plots...")

            # Create plot directory if saving
            if args.save_plots:
                plot_dir = Path(args.plot_dir)
                plot_dir.mkdir(exist_ok=True)

            # Plot individual method curves
            if lapeft_results:
                save_path = plot_dir / f"lapeft_{args.dataset}_{args.feature_model}.png" if args.save_plots else None
                plot_hypervolume_curves(lapeft_results, 'LAPEFT', args.dataset, args.feature_model, save_path)

            if llmat_results:
                clustering_suffix = f"_{args.clustering}" if args.clustering != 'kmeans' else ""
                save_path = plot_dir / f"llmat_{args.dataset}_{args.feature_model}{clustering_suffix}.png" if args.save_plots else None
                plot_hypervolume_curves(llmat_results, 'LLMAT', args.dataset, args.feature_model, save_path)

            # Plot comparison
            if lapeft_results and llmat_results:
                save_path = plot_dir / f"comparison_{args.dataset}_{args.feature_model}_{args.clustering}.png" if args.save_plots else None
                plot_comparison_curves(lapeft_results, llmat_results, args.dataset, args.feature_model, save_path)

        # Compare methods
        compare_methods(lapeft_results, llmat_results, args.dataset, args.feature_model)

    except Exception as e:
        print(f"Error loading results: {e}")
        print("Make sure the results files exist in the correct directory structure:")
        print("  results/multiobj/{dataset}/{feature_model}/{method}/")
        print("And that the environment has required dependencies (numpy, pandas, matplotlib, seaborn)")
        sys.exit(1)


if __name__ == "__main__":
    main()
