#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Plot GAP and regret trajectories from levy-10d results
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
from pathlib import Path
import re

def load_levy_results(base_dir='results/levy-10d'):
    """Load all levy results from different noise configurations

    Note: We use trace_y_his (history of selected points) and compute cumulative
    minimum since trace_best_y may not be correctly tracking the best value.
    """

    results = {}

    # Find all noise configuration directories
    for noise_dir in Path(base_dir).iterdir():
        if not noise_dir.is_dir():
            continue

        # Parse noise type and scale from directory name
        dir_name = noise_dir.name
        if 'noise' in dir_name:
            # e.g., x_original_noise-gaussian-0.1
            match = re.search(r'noise-(\w+)-([\d.]+)', dir_name)
            if match:
                noise_type = match.group(1)
                noise_scale = float(match.group(2))
            else:
                continue
        elif dir_name == 'x_original':
            noise_type = 'gaussian'
            noise_scale = 0.0
        else:
            continue

        # Find trace_y_his files (history of selected points)
        trace_files = list(noise_dir.glob('**/trace_y_his_*.npy'))

        key = (noise_type, noise_scale)
        if key not in results:
            results[key] = []

        for trace_file in trace_files:
            try:
                trace = np.load(trace_file)
                # Compute cumulative minimum (best so far for minimization)
                cumulative_best = np.minimum.accumulate(trace)
                results[key].append(cumulative_best)
                print(f"Loaded: {trace_file.name} -> {noise_type} {noise_scale}")
            except Exception as e:
                print(f"Error loading {trace_file}: {e}")

    return results


def compute_gap_and_regret(trace_best_y, clean_opt):
    """
    Compute GAP and regret from trace_best_y

    Args:
        trace_best_y: Array of best y values over iterations
        clean_opt: True clean optimum (for levy 10D, it's around 0)

    Note:
        For minimization problems (like Levy), best_y decreases over time.
        GAP should still go from 0 (no progress) to 1 (optimal).
    """
    initial_best = trace_best_y[0]

    # GAP: normalized progress
    gap = []
    regret = []

    for best_y in trace_best_y:
        # GAP - handle both minimization and maximization
        if clean_opt != initial_best:
            # For minimization (clean_opt < initial_best): as best_y decreases, gap increases
            # For maximization (clean_opt > initial_best): as best_y increases, gap increases
            g = (initial_best - best_y) / (initial_best - clean_opt)
        else:
            g = 1.0
        # Clip GAP to [0, 1] range (noise can make observed y better than true optimum)
        gap.append(np.clip(np.nan_to_num(g, nan=1.0), 0.0, 1.0))

        # Regret: distance to optimum
        r = abs(best_y - clean_opt)
        regret.append(r)

    return np.array(gap), np.array(regret)


def plot_trajectories(results, clean_opt=0.0, save_path='levy_gap_regret.png'):
    """Plot GAP and regret trajectories"""

    # Get noise scales
    noise_scales = sorted(set(scale for _, scale in results.keys()))
    noise_types = sorted(set(ntype for ntype, _ in results.keys()))

    print(f"Noise types: {noise_types}")
    print(f"Noise scales: {noise_scales}")

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_scales)))

    for metric_idx, metric in enumerate(['gap', 'regret']):
        ax = axes[metric_idx]

        for scale_idx, noise_scale in enumerate(noise_scales):
            # Collect all traces for this noise scale
            all_gaps = []
            all_regrets = []

            for noise_type in noise_types:
                key = (noise_type, noise_scale)
                if key not in results:
                    continue

                for trace in results[key]:
                    gap, regret = compute_gap_and_regret(trace, clean_opt)
                    all_gaps.append(gap)
                    all_regrets.append(regret)

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

            # Compute mean and SEM (standard error of the mean)
            if metric == 'gap':
                data = np.array(gaps_padded)
            else:
                data = np.array(regrets_padded)

            mean_trace = np.mean(data, axis=0)
            sem_trace = st.sem(data, axis=0)  # Changed from std to sem

            iterations = np.arange(len(mean_trace))

            # Plot
            ax.plot(iterations, mean_trace, color=colors[scale_idx],
                   label=f'noise={noise_scale}', linewidth=2)
            ax.fill_between(iterations,
                           mean_trace - sem_trace,
                           mean_trace + sem_trace,
                           color=colors[scale_idx], alpha=0.2)

        # Labels
        ax.set_xlabel('BO Iteration', fontsize=12)
        if metric == 'gap':
            ax.set_ylabel('GAP (Normalized Progress)', fontsize=12)
            ax.set_title('LEVY 10D - GAP Trajectory', fontsize=14, fontweight='bold')
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Optimal')
        else:
            ax.set_ylabel('Regret (Distance to Optimum)', fontsize=12)
            ax.set_title('LEVY 10D - Regret Trajectory', fontsize=14, fontweight='bold')

        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot levy-10d GAP and regret trajectories')
    parser.add_argument('--base_dir', type=str, default='results/levy-10d',
                       help='Base directory containing levy results')
    parser.add_argument('--clean_opt', type=float, default=0.0,
                       help='True clean optimum for levy function (default: 0.0)')
    parser.add_argument('--output', type=str, default='levy_gap_regret.png',
                       help='Output path for figure')

    args = parser.parse_args()

    print(f"Loading results from: {args.base_dir}")
    results = load_levy_results(args.base_dir)

    if not results:
        print("No results found!")
    else:
        print(f"\nFound {sum(len(v) for v in results.values())} total traces")
        print(f"Configurations: {len(results)}")

        plot_trajectories(results, clean_opt=args.clean_opt, save_path=args.output)
