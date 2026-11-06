#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: plot_gamma_ablation_redox.py
# Description: Plot ablation study for gamma parameter (0.35, 0.5, 0.6)
#              Compares GAP and Regret metrics for redox-mer benchmark
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from copy import deepcopy

# Ground truth optimal value for redox-mer dataset
# Note: experiments run as MINIMIZATION, so optimal is the minimum value
GROUND_TRUTH_OPT = 1.54135178  # Minimum Ered value (best for minimization)

# Configuration
seeds = list(range(16))  # Seeds 0-15
data_name = "redox-mer"
f_model = "t5-base-chem"
prompt_type = "just-smiles"
finetuning = False
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_type = "kmeans"

# Fixed hyperparameters
n_init_data = 10
acqf = "ei"
eta = 0.005
tree_depth = 3

# Gamma configurations to compare
gamma_configs = {
    "gamma=0.35": {
        "gamma": 0.35,
        "label": "γ=0.35",
        "linestyle": "-",
        "color": None
    },
    "gamma=0.5": {
        "gamma": 0.5,
        "label": "γ=0.5",
        "linestyle": "--",
        "color": None
    },
    "gamma=0.6": {
        "gamma": 0.6,
        "label": "γ=0.6",
        "linestyle": "-.",
        "color": None
    }
}

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.grid'] = True


def get_suffix(gamma, seed):
    """Get suffix for file naming."""
    return f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"


def load_trace_best_y(gamma, seed):
    """Load trace_best_y data for a specific gamma value and seed."""
    suffix = get_suffix(gamma, seed)

    # Construct path based on benchmark structure
    feature_name = f"{f_model}-{prompt_type}-{feature_reduction}"
    path = f"results/{data_name}/{feature_name}/llmat/{clustering_type}"

    # Exact filename pattern: ensure no extra p_val parameter between tree_depth and seed
    # Format should be: trace_best_y_{n_init}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}.npy
    filepath = f"{path}/trace_best_y_{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}.npy"

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        ret = np.load(filepath)
        return ret
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_gap(trace_by, y_opt):
    """
    Compute GAP metric from trace_best_y.
    For minimization: GAP[t] = (y_0 - y_t) / (y_0 - y_opt)
    Higher GAP means better (closer to optimum)
    """
    trace_gap = deepcopy(trace_by)
    trace_gap[0] = 0  # Set the first element to 0 for GAP calculation

    y_0 = trace_by[1]  # Initial best value

    for j in range(1, min(101, len(trace_by))):
        y_t = trace_by[j]
        # For minimization: positive GAP when y_t < y_0 (improvement)
        denominator = y_0 - y_opt
        if abs(denominator) < 1e-10:
            trace_gap[j] = 1.0
        else:
            gap_val = (y_0 - y_t) / denominator
            trace_gap[j] = np.clip(np.nan_to_num(gap_val, nan=0.0), 0.0, 1.0)

    return trace_gap


def compute_regret(trace_by, y_opt):
    """
    Compute average regret (cumulative average regret) from trace_best_y.
    Average Regret[t] = (1/t) * sum_{i=1}^{t} |y_i - y_opt|
    Lower regret means better (closer to optimum on average)
    """
    simple_regret = np.abs(trace_by - y_opt)

    # Compute cumulative average regret
    avg_regret = np.zeros_like(simple_regret)
    avg_regret[0] = simple_regret[0]

    for t in range(1, len(simple_regret)):
        # Average of regrets from iteration 1 to t
        avg_regret[t] = np.mean(simple_regret[1:t+1])

    return avg_regret


def plot_gap_comparison(save_path="./results/figs/gamma_ablation/"):
    """
    Plot GAP comparison across different gamma values.
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Plotting GAP Comparison for Gamma Ablation (Redox-Mer)")
    print(f"{'='*60}")
    print(f"Ground truth optimal: {GROUND_TRUTH_OPT:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn color palette
    colors = sns.color_palette("husl", len(gamma_configs))

    # Collect and plot data for each gamma configuration
    for idx, (gamma_name, gamma_config) in enumerate(gamma_configs.items()):
        print(f"\nProcessing {gamma_name}")

        trace_gaps = []
        gamma = gamma_config["gamma"]

        for seed in seeds:
            trace_by = load_trace_best_y(gamma, seed)

            if trace_by is not None:
                trace_gap = compute_gap(trace_by, GROUND_TRUTH_OPT)
                trace_gaps.append(trace_gap)

        if len(trace_gaps) == 0:
            print(f"  Warning: No data found for {gamma_name}")
            continue

        # Compute statistics
        trace_y_gaps = np.stack(trace_gaps)
        mean = np.mean(trace_y_gaps, axis=0)[1:101]
        sem = st.sem(trace_y_gaps, axis=0)[1:101]

        T = np.arange(len(mean)) + 1

        # Plot
        color = colors[idx] if gamma_config["color"] is None else gamma_config["color"]
        ax.plot(T, mean, color=color, linestyle=gamma_config["linestyle"],
                label=gamma_config["label"], linewidth=2)
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.15, color=color)

        print(f"  {gamma_name}: {len(trace_gaps)} runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.ylim(0, 1.05)
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.title(f"Gamma Ablation Study (Redox-Mer): GAP ($\\uparrow$)", fontsize=18)
    plt.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(save_path + "redox_mer_gamma_ablation_GAP.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(save_path + "redox_mer_gamma_ablation_GAP.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nGAP plot saved to {save_path}")


def plot_regret_comparison(save_path="./results/figs/gamma_ablation/"):
    """
    Plot Average Regret comparison across different gamma values.
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Plotting Average Regret Comparison for Gamma Ablation (Redox-Mer)")
    print(f"{'='*60}")

    y_opt = GROUND_TRUTH_OPT

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn color palette
    colors = sns.color_palette("husl", len(gamma_configs))

    # Collect and plot data for each gamma configuration
    for idx, (gamma_name, gamma_config) in enumerate(gamma_configs.items()):
        print(f"\nProcessing {gamma_name}")

        trace_regrets = []
        gamma = gamma_config["gamma"]

        for seed in seeds:
            trace_by = load_trace_best_y(gamma, seed)

            if trace_by is not None:
                trace_regret = compute_regret(trace_by, y_opt)
                trace_regrets.append(trace_regret)

        if len(trace_regrets) == 0:
            print(f"  Warning: No data found for {gamma_name}")
            continue

        # Compute statistics
        trace_y_regrets = np.stack(trace_regrets)
        mean = np.mean(trace_y_regrets, axis=0)[1:101]
        sem = st.sem(trace_y_regrets, axis=0)[1:101]

        T = np.arange(len(mean)) + 1

        # Plot
        color = colors[idx] if gamma_config["color"] is None else gamma_config["color"]
        ax.plot(T, mean, color=color, linestyle=gamma_config["linestyle"],
                label=gamma_config["label"], linewidth=2)
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.15, color=color)

        print(f"  {gamma_name}: {len(trace_regrets)} runs, final Average Regret: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Average Regret', fontsize=16)
    plt.title(f"Gamma Ablation Study (Redox-Mer): Average Regret ($\\downarrow$)", fontsize=18)
    plt.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    plt.yscale('log')  # Use log scale for better visualization
    plt.tight_layout()

    plt.savefig(save_path + "redox_mer_gamma_ablation_regret.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(save_path + "redox_mer_gamma_ablation_regret.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAverage Regret plot saved to {save_path}")


def plot_combined_comparison(save_path="./results/figs/gamma_ablation/"):
    """
    Plot both GAP and Regret in subplots side by side.
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Plotting Combined GAP & Regret Comparison (Redox-Mer)")
    print(f"{'='*60}")

    y_opt = GROUND_TRUTH_OPT
    print(f"Ground truth optimal: {y_opt:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Use seaborn color palette
    colors = sns.color_palette("husl", len(gamma_configs))

    # Collect and plot data for each gamma configuration
    for idx, (gamma_name, gamma_config) in enumerate(gamma_configs.items()):
        print(f"\nProcessing {gamma_name}")

        trace_gaps = []
        trace_regrets = []
        gamma = gamma_config["gamma"]

        for seed in seeds:
            trace_by = load_trace_best_y(gamma, seed)

            if trace_by is not None:
                trace_gap = compute_gap(trace_by, y_opt)
                trace_gaps.append(trace_gap)

                trace_regret = compute_regret(trace_by, y_opt)
                trace_regrets.append(trace_regret)

        if len(trace_gaps) == 0:
            print(f"  Warning: No data found for {gamma_name}")
            continue

        color = colors[idx] if gamma_config["color"] is None else gamma_config["color"]

        # Plot GAP
        trace_y_gaps = np.stack(trace_gaps)
        mean_gap = np.mean(trace_y_gaps, axis=0)[1:101]
        sem_gap = st.sem(trace_y_gaps, axis=0)[1:101]
        T = np.arange(len(mean_gap)) + 1

        ax1.plot(T, mean_gap, color=color, linestyle=gamma_config["linestyle"],
                label=gamma_config["label"], linewidth=2)
        ax1.fill_between(T, mean_gap - sem_gap, mean_gap + sem_gap, alpha=0.15, color=color)

        # Plot Regret
        trace_y_regrets = np.stack(trace_regrets)
        mean_regret = np.mean(trace_y_regrets, axis=0)[1:101]
        sem_regret = st.sem(trace_y_regrets, axis=0)[1:101]

        ax2.plot(T, mean_regret, color=color, linestyle=gamma_config["linestyle"],
                label=gamma_config["label"], linewidth=2)
        ax2.fill_between(T, mean_regret - sem_regret, mean_regret + sem_regret,
                        alpha=0.15, color=color)

        print(f"  {gamma_name}: GAP={mean_gap[-1]:.4f}±{sem_gap[-1]:.4f}, "
              f"Avg Regret={mean_regret[-1]:.4f}±{sem_regret[-1]:.4f}")

    # Configure GAP subplot
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('BO Rounds: T', fontsize=16)
    ax1.set_ylabel('GAP', fontsize=16)
    ax1.set_title(f"GAP ($\\uparrow$)", fontsize=18)
    ax1.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # Configure Regret subplot
    ax2.set_xlim(1, 100)
    ax2.set_xlabel('BO Rounds: T', fontsize=16)
    ax2.set_ylabel('Average Regret', fontsize=16)
    ax2.set_title(f"Average Regret ($\\downarrow$)", fontsize=18)
    ax2.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', linewidth=0.5)

    plt.suptitle("Gamma Ablation Study (Redox-Mer)", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(save_path + "redox_mer_gamma_ablation_combined.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(save_path + "redox_mer_gamma_ablation_combined.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCombined plot saved to {save_path}")


def print_summary_statistics():
    """
    Print summary statistics for all gamma configurations.
    """
    y_opt = GROUND_TRUTH_OPT

    print(f"\n{'='*70}")
    print(f"Summary Statistics: Final Performance (T=100)")
    print(f"{'='*70}")
    print(f"{'Gamma':<30s} {'GAP':<20s} {'Avg Regret':<20s} {'Runs':<10s}")
    print(f"{'-'*70}")

    for gamma_name, gamma_config in gamma_configs.items():
        gamma = gamma_config["gamma"]

        trace_gaps = []
        trace_regrets = []

        for seed in seeds:
            trace_by = load_trace_best_y(gamma, seed)

            if trace_by is not None:
                trace_gap = compute_gap(trace_by, y_opt)
                trace_gaps.append(trace_gap)

                trace_regret = compute_regret(trace_by, y_opt)
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

            print(f"{gamma_config['label']:<30s} "
                  f"{mean_gap:.4f} ± {sem_gap:.4f}   "
                  f"{mean_regret:.4f} ± {sem_regret:.4f}   "
                  f"{len(trace_gaps):>5d}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Create output directory
    output_path = "./results/figs/gamma_ablation/"

    # Print summary statistics
    print_summary_statistics()

    # Generate plots
    plot_gap_comparison(save_path=output_path)
    plot_regret_comparison(save_path=output_path)
    plot_combined_comparison(save_path=output_path)

    print("\nAll plots generated successfully!")
