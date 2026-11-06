#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: plot_gap_algorithms_avg.py
# Description: Plot average GAP metric comparison across all datasets
#              for each feature model and different algorithms (LLMAT, random, gp, laplace, lfbo, bore)
"""

import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from copy import deepcopy
from benchmarks.MAT.material_benchmarks import MATBench

# Configuration
seeds = list(set(range(15)))
data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
f_models = ["t5-base-chem", "molformer", "fingerprints", "t5-base", "gpt2-medium", "gpt2-large", "llama-2-7b"]  # Can be customized
prompt_type = "just-smiles"
finetuning = False  # Fixed feature model
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_type = "kmeans"  # Default clustering type for LLMAT

# Algorithm configurations
algorithms = {
    "LLMAT": {
        "type": "ours",
        "label": "LLMAT",
        "linestyle": "-",
        "load_func": "load_ours"
    },
    "GP": {
        "type": "baseline",
        "label": "GP",
        "linestyle": "--",
        "load_func": "load_baselines",
        "baseline_name": "gp"
    },
    "Laplace": {
        "type": "baseline",
        "label": "Laplace",
        "linestyle": "-.",
        "load_func": "load_baselines",
        "baseline_name": "laplace"
    },
    "Random": {
        "type": "baseline",
        "label": "Random",
        "linestyle": ":",
        "load_func": "load_baselines",
        "baseline_name": "random"
    },
    "LFBO": {
        "type": "ours",
        "label": "LFBO",
        "linestyle": (0, (3, 1, 1, 1)),
        "load_func": "load_ours",
        "baseline_name": "lfbo"
    },
    "BORE": {
        "type": "ours",
        "label": "BORE",
        "linestyle": (0, (5, 2, 1, 2)),
        "load_func": "load_ours",
        "baseline_name": "bore"
    }
}

# Set plot style
# sns.set_theme(style="whitegrid")
# plt.rcParams.update({'font.size': 16})
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['figure.titlesize'] = 14
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['lines.markersize'] = 8
# plt.rcParams['grid.linestyle'] = '--'
# plt.rcParams['grid.linewidth'] = 0.5
# plt.rcParams['axes.grid'] = True

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 16})  # Set default font size for all text elements
plt.rcParams['axes.labelsize'] = 16  # Set font size for axes labels
plt.rcParams['axes.titlesize'] = 16  # Set font size for axes titles
# plt.rcParams['legend.fontsize'] = 16  # Set font size for legend
plt.rcParams['figure.titlesize'] = 16  # Set font size for figure title
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'  # Set axes label weight to bold
plt.rcParams['axes.titleweight'] = 'bold'  # Set axes title weight to bold
plt.rcParams['lines.linewidth'] = 2  # Set line width for plots
plt.rcParams['lines.markersize'] = 8  # Set marker size for plots
plt.rcParams['grid.linestyle'] = '--'  # Set grid line style
plt.rcParams['grid.linewidth'] = 0.5  # Set grid line width
plt.rcParams['axes.grid'] = True  # Enable grid lines


def get_ours_suffix(seed, f_model, algo, n_init_data=10, acqf="ei", gamma=0.5, eta=0.01, tree_depth=3):
    """Get suffix for LLMAT results."""
    if algo == "llmat":
        if f_model == "t5-base-chem":
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
        elif f_model == "molformer" or f_model == "fingerprints":
            eta = 0.005
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
        else:
            eta = 0.005
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_0.01_{seed}"
    elif algo == "lfbo":
        eta = 0.005
        tree_depth = 2
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    elif algo == "bore":
        eta = 0.005
        tree_depth = 2
        acqf = "pi"
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    else:
        raise NotImplementedError(f"Algorithm {algo} not recognized for suffix generation.")
    return suffix


def get_baselines_suffix(seed, n_init_data=10, acqf="ts", laplace_type="all_layer", base_line="gp"):
    """Get suffix for baseline results."""
    suffix = f"{n_init_data}_{acqf}_{laplace_type}_{base_line}_{seed}"
    return suffix


def load_ours(mat_bench, f_model, seed, r_type="best_y", algo="llmat"):
    """Load LLMAT results."""
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/{algo}"
    clustering = mat_bench.dataset_name.split("/")[-2]
    if algo == "llmat":
        path = f"{path}/{clustering}"

    suffix = get_ours_suffix(seed, f_model, algo)
    filepath = f"{path}/trace_{r_type}_{suffix}.npy"

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        ret = np.load(filepath)
        return ret
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_baselines(mat_bench, seed, base_line="gp", r_type="best_y"):
    """Load baseline results."""
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/lapeft"

    suffix = get_baselines_suffix(seed, base_line=base_line)
    filepath = f"{path}/trace_{r_type}_{suffix}.npy"

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    try:
        ret = np.load(filepath)
        return ret
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_gap(trace_by, mat_bench):
    """
    Compute GAP metric from trace_best_y.
    GAP[t] = (y_t - y_0) / (y_opt - y_0)
    """
    trace_gap = deepcopy(trace_by)
    trace_gap[0] = 0  # Set the first element to 0 for GAP calculation

    for j in range(1, min(101, len(trace_by))):
        y_0 = trace_by[1]
        y_t = trace_by[j]
        trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

    return trace_gap


def plot_average_gap_per_model(path="./results/figs/algorithms_avg/"):
    """
    Plot average GAP across all datasets for different algorithms,
    with one plot per feature model.
    """
    os.makedirs(path, exist_ok=True)

    for f_model in f_models:
        print(f"\n{'='*60}")
        print(f"Processing Feature Model: {f_model}")
        print(f"{'='*60}")

        fig, ax = plt.subplots(figsize=(10, 6))

        trace_gaps_all_algos = {algo_name: [] for algo_name in algorithms.keys()}

        # Collect data for each algorithm
        for algo_name, algo_config in algorithms.items():
            print(f"\n  Processing {algo_name}")

            for data_name in data_names:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                for seed in seeds:
                    if algo_config["type"] == "ours":
                        trace_by = load_ours(mat_bench, f_model, seed, r_type="best_y", algo=algo_name.lower())
                    else:  # baseline
                        trace_by = load_baselines(mat_bench, seed, base_line=algo_config["baseline_name"], r_type="best_y")

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        trace_gaps_all_algos[algo_name].append(trace_gap)

        # Plot all algorithms for this feature model
        for algo_name, algo_config in algorithms.items():
            if len(trace_gaps_all_algos[algo_name]) == 0:
                print(f"  Warning: No data found for {algo_name}")
                continue

            trace_y_gaps = np.stack(trace_gaps_all_algos[algo_name])
            mean = np.mean(trace_y_gaps, axis=0)[1:101]
            sem = st.sem(trace_y_gaps, axis=0)[1:101]

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

            sns.lineplot(data=df, x='T', y='y', label=algo_config["label"], linestyle=algo_config["linestyle"], ax=ax)
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.2)

            print(f"  {algo_name}: {len(trace_y_gaps)} total runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

        plt.xlim(1, 100)
        plt.ylim(0, 1.05)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')
        plt.xlabel('BO Rounds: T', fontsize=16)
        plt.ylabel('GAP', fontsize=16)
        plt.tight_layout()
        plt.title(f"Algorithm Comparison ({f_model}): Average GAP Across Datasets ($\\uparrow$)")
        plt.savefig(path + f"{f_model}_GAP_algorithms_average.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(path + f"{f_model}_GAP_algorithms_average.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


def plot_average_gap_all_models_combined(path="./results/figs/algorithms_avg/"):
    """
    Plot average GAP across all datasets with all feature models and algorithms
    on a single plot, using different colors for models and line styles for algorithms.
    """
    os.makedirs(path, exist_ok=True)

    # Use seaborn color palette for feature models
    model_colors = sns.color_palette("deep", len(f_models))
    f_model_color_map = dict(zip(f_models, model_colors))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Store legend elements
    algo_legend_elements = []
    algo_legend_labels = []
    model_legend_elements = []
    model_legend_labels = []

    added_algos = set()
    added_models = set()

    for f_model in f_models:
        print(f"\n{'='*60}")
        print(f"Processing Feature Model: {f_model}")
        print(f"{'='*60}")

        color = f_model_color_map[f_model]

        for algo_name, algo_config in algorithms.items():
            print(f"\n  Processing {algo_name}")

            trace_gaps = []

            for data_name in data_names:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                for seed in seeds:
                    if algo_config["type"] == "ours":
                        trace_by = load_ours(mat_bench, f_model, seed, r_type="best_y", algo=algo_name.lower())
                    else:  # baseline
                        if algo_config["baseline_name"] == "random" and f_model != "fingerprints":
                            continue
                        trace_by = load_baselines(mat_bench, seed, base_line=algo_config["baseline_name"], r_type="best_y")

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        trace_gaps.append(trace_gap)

            if len(trace_gaps) == 0:
                print(f"  Warning: No data found for {f_model} - {algo_name}")
                continue

            trace_y_gaps = np.stack(trace_gaps)
            mean = np.mean(trace_y_gaps, axis=0)[1:101]
            sem = st.sem(trace_y_gaps, axis=0)[1:101]

            T = np.arange(len(mean)) + 1

            # Plot with model color and algorithm line style
            ax.plot(T, mean, color=color, linestyle=algo_config["linestyle"], label=f"{f_model}-{algo_config['label']}", linewidth=2)
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.15, color=color)

            # Add to legend elements
            if algo_name not in added_algos:
                algo_legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=algo_config["linestyle"], linewidth=2))
                algo_legend_labels.append(algo_config["label"])
                added_algos.add(algo_name)

            if f_model not in added_models:
                model_legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-', linewidth=3))
                model_legend_labels.append(f_model)
                added_models.add(f_model)

            print(f"  {f_model}-{algo_name}: {len(trace_gaps)} runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.ylim(0, 1.05)
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)

    # Create two separate legends
    algo_legend = ax.legend(algo_legend_elements, algo_legend_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), title='Algorithms', fontsize=12, title_fontsize=14,
                            frameon=True, fancybox=True, shadow=False, framealpha=0.9)

    model_legend = ax.legend(model_legend_elements, model_legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), title='Feature Models', fontsize=12, title_fontsize=14,
                             frameon=True, fancybox=True, shadow=False, framealpha=0.9)

    # Add the first legend back to the axes
    ax.add_artist(algo_legend)

    plt.tight_layout()
    plt.title(f"Algorithm Comparison: Average GAP Across All Datasets ($\\uparrow$)", fontsize=18, pad=20)
    plt.savefig(path + "all_models_GAP_algorithms_average.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(path + "all_models_GAP_algorithms_average.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_average_gap_subplots(path="./results/figs/algorithms_avg/"):
    """
    Plot average GAP with subplots: one subplot per feature model.
    Each subplot shows all algorithms for that model.
    """
    os.makedirs(path, exist_ok=True)

    n_models = len(f_models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten() if n_models > 1 else axes

    for idx, f_model in enumerate(f_models):
        ax = axes_flat[idx]

        print(f"\n{'='*60}")
        print(f"Processing Feature Model: {f_model}")
        print(f"{'='*60}")

        trace_gaps_all_algos = {algo_name: [] for algo_name in algorithms.keys()}

        # Collect data for each algorithm
        for algo_name, algo_config in algorithms.items():
            print(f"\n  Processing {algo_name}")

            for data_name in data_names:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                for seed in seeds:
                    if algo_config["type"] == "ours":
                        trace_by = load_ours(mat_bench, f_model, seed, r_type="best_y", algo=algo_name.lower())
                    else:  # baseline
                        if algo_config["baseline_name"] == "random" and f_model != "fingerprints":
                            continue
                        trace_by = load_baselines(mat_bench, seed, base_line=algo_config["baseline_name"], r_type="best_y")

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        trace_gaps_all_algos[algo_name].append(trace_gap)

        # Plot all algorithms for this feature model
        for algo_name, algo_config in algorithms.items():
            if len(trace_gaps_all_algos[algo_name]) == 0:
                print(f"  Warning: No data found for {algo_name}")
                continue

            trace_y_gaps = np.stack(trace_gaps_all_algos[algo_name])
            mean = np.mean(trace_y_gaps, axis=0)[1:101]
            sem = st.sem(trace_y_gaps, axis=0)[1:101]

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

            sns.lineplot(data=df, x='T', y='y', label=algo_config["label"], linestyle=algo_config["linestyle"], ax=ax)
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.2)

            print(f"  {algo_name}: {len(trace_y_gaps)} runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

        ax.set_xlim(1, 100)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel('GAP', fontsize=14, fontweight='bold')
        ax.set_title(f"{f_model} ($\\uparrow$)", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(f_models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(path + "all_models_GAP_algorithms_subplots.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(path + "all_models_GAP_algorithms_subplots.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_average_gap_over_all(path="./results/figs/algorithms_avg/"):
    """
    Plot average GAP over ALL datasets AND all feature models for different algorithms.
    This aggregates across both datasets and feature models to show algorithm performance.
    """
    os.makedirs(path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Computing Average GAP Over All Datasets and All Feature Models")
    print(f"{'='*60}")

    fig, ax = plt.subplots(figsize=(6, 3))  #figsize=(8, 6)

    trace_gaps_all_algos = {algo_name: [] for algo_name in algorithms.keys()}

    # Collect data for each algorithm across ALL models and datasets
    for algo_name, algo_config in algorithms.items():
        print(f"\n  Processing {algo_name}")

        for f_model in f_models:
            print(f"    Processing feature model: {f_model}")

            for data_name in data_names:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                for seed in seeds:
                    if algo_config["type"] == "ours":
                        trace_by = load_ours(mat_bench, f_model, seed, r_type="best_y", algo=algo_name.lower())
                    else:  # baseline
                        if algo_config["baseline_name"] == "random" and f_model != "fingerprints":
                            continue
                        trace_by = load_baselines(mat_bench, seed, base_line=algo_config["baseline_name"], r_type="best_y")

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        trace_gaps_all_algos[algo_name].append(trace_gap)

    # Plot all algorithms
    for algo_name, algo_config in algorithms.items():
        if len(trace_gaps_all_algos[algo_name]) == 0:
            print(f"  Warning: No data found for {algo_name}")
            continue

        trace_y_gaps = np.stack(trace_gaps_all_algos[algo_name])
        mean = np.mean(trace_y_gaps, axis=0)[1:101]
        sem = st.sem(trace_y_gaps, axis=0)[1:101]

        T = np.arange(len(mean)) + 1
        df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

        sns.lineplot(data=df, x='T', y='y', label=algo_config["label"], linestyle=algo_config["linestyle"], ax=ax)
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.2)

        print(f"  {algo_name}: {len(trace_y_gaps)} total runs across all models and datasets, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.ylim(0, 1.05)
    plt.setp(ax.get_legend().get_texts(), fontsize='12')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.tight_layout()
    plt.title(f"Average GAP Over All Models & Datasets ($\\uparrow$)", fontsize=14)
    plt.savefig(path + "all_models_datasets_GAP_algorithms_average.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(path + "all_models_datasets_GAP_algorithms_average.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics (Final GAP at T=100)")
    print(f"{'='*60}")
    for algo_name in algorithms.keys():
        if len(trace_gaps_all_algos[algo_name]) > 0:
            trace_y_gaps = np.stack(trace_gaps_all_algos[algo_name])
            final_gaps = trace_y_gaps[:, 100]
            mean_final = np.mean(final_gaps)
            sem_final = st.sem(final_gaps)
            print(f"{algo_name:15s}: {mean_final:.4f} ± {sem_final:.4f} (n={len(trace_y_gaps)})")


if __name__ == "__main__":
    # Plot average GAP for each feature model separately
    # plot_average_gap_per_model(path="./results/figs/algorithms_avg/")

    # Plot average GAP with all models combined in one plot
    # plot_average_gap_all_models_combined(path="./results/figs/algorithms_avg/")

    # Plot average GAP with subplots (one per model)
    # plot_average_gap_subplots(path="./results/figs/algorithms_avg/")

    # Plot average GAP over all models and datasets
    plot_average_gap_over_all(path="./results/figs/algorithms_avg/")
