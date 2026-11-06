#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: plot_noise_ablation_levy.py
# Description: Plot ablation study of noise scales for levy-10d benchmark
#              Compares GAP and Regret metrics across different noise levels
"""

import os
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from copy import deepcopy

# Configuration
seeds = [42, 123, 456, 789, 1024]  # Seeds used in levy-10d experiments
base_path = "results/levy-10d"

# Noise configurations to compare
noise_configs = {
    "No Noise": {
        "dir": "x_original",
        "label": "No Noise (Original)",
        "linestyle": "-",
        "color": None  # Will use default palette
    },
    "Gaussian-0.05": {
        "dir": "x_original_noise-gaussian-0.05",
        "label": "Gaussian σ=0.05",
        "linestyle": "-",
        "color": None
    },
    "Gaussian-0.1": {
        "dir": "x_original_noise-gaussian-0.1",
        "label": "Gaussian σ=0.1",
        "linestyle": "-",
        "color": None
    },
    "Gaussian-0.2": {
        "dir": "x_original_noise-gaussian-0.2",
        "label": "Gaussian σ=0.2",
        "linestyle": "-",
        "color": None
    },
    "Gaussian-0.5": {
        "dir": "x_original_noise-gaussian-0.5",
        "label": "Gaussian σ=0.5",
        "linestyle": "--",
        "color": None
    },
    "Gaussian-1.0": {
        "dir": "x_original_noise-gaussian-1.0",
        "label": "Gaussian σ=1.0",
        "linestyle": "--",
        "color": None
    },
    "Uniform-1.0": {
        "dir": "x_original_noise-uniform-1.0",
        "label": "Uniform scale=1.0",
        "linestyle": "-.",
        "color": None
    }
}

# Algorithm parameters (fixed for levy-10d experiments)
n_init_data = 10
acqf = "ei"
gamma = 0.5
eta = 0.005
tree_depth = 3

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 16})  # Set default font size for all text elements
plt.rcParams['axes.labelsize'] = 16  # Set font size for axes labels
plt.rcParams['axes.titlesize'] = 16  # Set font size for axes titles
# plt.rcParams['legend.fontsize'] = 16  # Set font size for legend
plt.rcParams['figure.titlesize'] = 16  # Set font size for figure title
plt.rcParams['axes.labelweight'] = 'bold'  # Set axes label weight to bold
plt.rcParams['axes.titleweight'] = 'bold'  # Set axes title weight to bold
plt.rcParams['lines.linewidth'] = 2  # Set line width for plots
plt.rcParams['lines.markersize'] = 8  # Set marker size for plots
plt.rcParams['grid.linestyle'] = '--'  # Set grid line style
plt.rcParams['grid.linewidth'] = 0.5  # Set grid line width
plt.rcParams['axes.grid'] = True  # Enable grid lines


def get_suffix(seed):
    """Get suffix for file naming."""
    return f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"


def load_trace_best_y(noise_dir, seed):
    """Load trace_best_y data for a specific noise level and seed."""
    suffix = get_suffix(seed)
    filepath = f"{base_path}/{noise_dir}/llmat/kmeans/trace_best_y_{suffix}.npy"

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        ret = np.load(filepath)
        return ret
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_gap(trace_by, y_opt=0.0):
    """
    Compute GAP metric from trace_best_y.
    GAP[t] = (y_t - y_0) / (y_opt - y_0)

    For Levy function, the optimal value is 0.0
    """
    trace_gap = deepcopy(trace_by)
    trace_gap[0] = 0  # Set the first element to 0 for GAP calculation

    y_0 = trace_by[1]  # Initial best value after first observation

    for j in range(1, min(101, len(trace_by))):
        y_t = trace_by[j]
        # For minimization: GAP = (y_0 - y_t) / (y_0 - y_opt)
        # Higher GAP means better (closer to optimum)
        denominator = y_0 - y_opt
        if abs(denominator) < 1e-10:
            trace_gap[j] = 1.0
        else:
            gap_val = (y_0 - y_t) / denominator
            # Clip at 1.0: with noise, observed y_t can be better than true optimum
            trace_gap[j] = np.clip(np.nan_to_num(gap_val, nan=0.0), 0.0, 1.0)

    return trace_gap


def compute_regret(trace_by, y_opt=0.0):
    """
    Compute simple regret from trace_best_y.
    Regret[t] = |y_t - y_opt|

    For Levy function, the optimal value is 0.0
    Note: Use absolute value since noise can make observed y better than true optimum
    """
    trace_regret = np.abs(trace_by - y_opt)
    return trace_regret


def plot_gap_comparison(save_path="./results/figs/noise_ablation/"):
    """
    Plot GAP comparison across different noise levels.
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Plotting GAP Comparison for Noise Ablation (Levy-10D)")
    print(f"{'='*60}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn color palette
    colors = sns.color_palette("husl", len(noise_configs))

    # Collect and plot data for each noise configuration
    for idx, (noise_name, noise_config) in enumerate(noise_configs.items()):
        print(f"\nProcessing {noise_name}")

        trace_gaps = []
        noise_dir = noise_config["dir"]

        for seed in seeds:
            trace_by = load_trace_best_y(noise_dir, seed)

            if trace_by is not None:
                trace_gap = compute_gap(trace_by, y_opt=0.0)
                trace_gaps.append(trace_gap)

        if len(trace_gaps) == 0:
            print(f"  Warning: No data found for {noise_name}")
            continue

        # Compute statistics
        trace_y_gaps = np.stack(trace_gaps)
        mean = np.mean(trace_y_gaps, axis=0)[1:101]
        sem = st.sem(trace_y_gaps, axis=0)[1:101]

        T = np.arange(len(mean)) + 1

        # Plot
        color = colors[idx] if noise_config["color"] is None else noise_config["color"]
        ax.plot(T, mean, color=color, linestyle=noise_config["linestyle"],
                label=noise_config["label"], linewidth=2)
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.15, color=color)

        print(f"  {noise_name}: {len(trace_gaps)} runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.ylim(0, 1.05)
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.title(f"Noise Ablation Study (Levy-10D): GAP ($\\uparrow$)", fontsize=18)
    plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(save_path + "levy_10d_noise_ablation_GAP.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(save_path + "levy_10d_noise_ablation_GAP.png", format='png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out - using Agg backend
    plt.close()

    print(f"\nPlots saved to {save_path}")


def plot_regret_comparison(save_path="./results/figs/noise_ablation/"):
    """
    Plot Simple Regret comparison across different noise levels.
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Plotting Regret Comparison for Noise Ablation (Levy-10D)")
    print(f"{'='*60}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn color palette
    colors = sns.color_palette("husl", len(noise_configs))

    # Collect and plot data for each noise configuration
    for idx, (noise_name, noise_config) in enumerate(noise_configs.items()):
        print(f"\nProcessing {noise_name}")

        trace_regrets = []
        noise_dir = noise_config["dir"]

        for seed in seeds:
            trace_by = load_trace_best_y(noise_dir, seed)

            if trace_by is not None:
                trace_regret = compute_regret(trace_by, y_opt=0.0)
                trace_regrets.append(trace_regret)

        if len(trace_regrets) == 0:
            print(f"  Warning: No data found for {noise_name}")
            continue

        # Compute statistics
        trace_y_regrets = np.stack(trace_regrets)
        mean = np.mean(trace_y_regrets, axis=0)[1:101]
        sem = st.sem(trace_y_regrets, axis=0)[1:101]

        T = np.arange(len(mean)) + 1

        # Plot
        color = colors[idx] if noise_config["color"] is None else noise_config["color"]
        ax.plot(T, mean, color=color, linestyle=noise_config["linestyle"],
                label=noise_config["label"], linewidth=2)
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.15, color=color)

        print(f"  {noise_name}: {len(trace_regrets)} runs, final Regret: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Simple Regret', fontsize=16)
    plt.title(f"Noise Ablation Study (Levy-10D): Simple Regret ($\\downarrow$)", fontsize=18)
    plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    plt.yscale('log')  # Use log scale for better visualization of regret
    plt.tight_layout()

    plt.savefig(save_path + "levy_10d_noise_ablation_regret.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(save_path + "levy_10d_noise_ablation_regret.png", format='png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out - using Agg backend
    plt.close()

    print(f"\nPlots saved to {save_path}")


def plot_combined_comparison(save_path="./results/figs/noise_ablation/"):
    """
    Plot both GAP and Regret in subplots side by side.
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Plotting Combined GAP & Regret Comparison (Levy-10D)")
    print(f"{'='*60}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Use seaborn color palette
    colors = sns.color_palette("husl", len(noise_configs))

    # Collect and plot data for each noise configuration
    for idx, (noise_name, noise_config) in enumerate(noise_configs.items()):
        print(f"\nProcessing {noise_name}")

        trace_gaps = []
        trace_regrets = []
        noise_dir = noise_config["dir"]

        for seed in seeds:
            trace_by = load_trace_best_y(noise_dir, seed)

            if trace_by is not None:
                trace_gap = compute_gap(trace_by, y_opt=0.0)
                trace_gaps.append(trace_gap)

                trace_regret = compute_regret(trace_by, y_opt=0.0)
                trace_regrets.append(trace_regret)

        if len(trace_gaps) == 0:
            print(f"  Warning: No data found for {noise_name}")
            continue

        color = colors[idx] if noise_config["color"] is None else noise_config["color"]

        # Plot GAP
        trace_y_gaps = np.stack(trace_gaps)
        mean_gap = np.mean(trace_y_gaps, axis=0)[1:101]
        sem_gap = st.sem(trace_y_gaps, axis=0)[1:101]
        T = np.arange(len(mean_gap)) + 1

        ax1.plot(T, mean_gap, color=color, linestyle=noise_config["linestyle"],
                label=noise_config["label"], linewidth=2)
        ax1.fill_between(T, mean_gap - sem_gap, mean_gap + sem_gap, alpha=0.15, color=color)

        # Plot Regret
        trace_y_regrets = np.stack(trace_regrets)
        mean_regret = np.mean(trace_y_regrets, axis=0)[1:101]
        sem_regret = st.sem(trace_y_regrets, axis=0)[1:101]

        ax2.plot(T, mean_regret, color=color, linestyle=noise_config["linestyle"],
                label=noise_config["label"], linewidth=2)
        ax2.fill_between(T, mean_regret - sem_regret, mean_regret + sem_regret,
                        alpha=0.15, color=color)

        print(f"  {noise_name}: GAP={mean_gap[-1]:.4f}±{sem_gap[-1]:.4f}, "
              f"Regret={mean_regret[-1]:.4f}±{sem_regret[-1]:.4f}")

    # Configure GAP subplot
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('BO Rounds: T', fontsize=16)
    ax1.set_ylabel('GAP', fontsize=16)
    ax1.set_title(f"GAP ($\\uparrow$)", fontsize=18)
    ax1.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # Configure Regret subplot
    ax2.set_xlim(1, 100)
    ax2.set_xlabel('BO Rounds: T', fontsize=16)
    ax2.set_ylabel('Simple Regret', fontsize=16)
    ax2.set_title(f"Simple Regret ($\\downarrow$)", fontsize=18)
    ax2.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', linewidth=0.5)

    plt.suptitle("Noise Ablation Study (Levy-10D)", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(save_path + "levy_10d_noise_ablation_combined.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(save_path + "levy_10d_noise_ablation_combined.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"\nPlots saved to {save_path}")


def print_summary_statistics():
    """
    Print summary statistics for all noise configurations.
    """
    print(f"\n{'='*70}")
    print(f"Summary Statistics: Final Performance (T=100)")
    print(f"{'='*70}")
    print(f"{'Noise Level':<30s} {'GAP':<20s} {'Regret':<20s} {'Runs':<10s}")
    print(f"{'-'*70}")

    for noise_name, noise_config in noise_configs.items():
        noise_dir = noise_config["dir"]

        trace_gaps = []
        trace_regrets = []

        for seed in seeds:
            trace_by = load_trace_best_y(noise_dir, seed)

            if trace_by is not None:
                trace_gap = compute_gap(trace_by, y_opt=0.0)
                trace_gaps.append(trace_gap)

                trace_regret = compute_regret(trace_by, y_opt=0.0)
                trace_regrets.append(trace_regret)

        if len(trace_gaps) > 0:
            trace_y_gaps = np.stack(trace_gaps)
            final_gaps = trace_y_gaps[:, 100]
            mean_gap = np.mean(final_gaps)
            sem_gap = st.sem(final_gaps)

            trace_y_regrets = np.stack(trace_regrets)
            final_regrets = trace_y_regrets[:, 100]
            mean_regret = np.mean(final_regrets)
            sem_regret = st.sem(final_regrets)

            print(f"{noise_config['label']:<30s} "
                  f"{mean_gap:.4f} ± {sem_gap:.4f}   "
                  f"{mean_regret:.4f} ± {sem_regret:.4f}   "
                  f"{len(trace_gaps):>5d}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Create output directory
    output_path = "./results/figs/noise_ablation/"

    # Print summary statistics
    print_summary_statistics()

    # Generate plots
    plot_gap_comparison(save_path=output_path)
    plot_regret_comparison(save_path=output_path)
    plot_combined_comparison(save_path=output_path)

    print("\nAll plots generated successfully!")
