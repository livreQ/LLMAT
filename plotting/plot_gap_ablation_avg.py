#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: plot_gap_ablation_avg.py
# Description: Plot average GAP metric for ablation study across all datasets
#              for each feature model and different algorithms
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
# f_models = ["t5-base-chem", "molformer", "gpt2-medium", "gpt2-large", "llama-2-7b"]  # Can be customized
f_models = ["molformer"]  # Fixed to molformer for demonstration
prompt_type = "just-smiles"
finetuning = False  # Fixed feature model
iupac = False
feature_reduction = "average"
run_subset_only = False

# Fixed parameters for ablation
n_init_data = 10
acqf = "ei"
gamma = 0.5

# Ablation configurations
# Each config specifies: eta, p_val, tree_depth, clustering_type
ablation_configs = {
    "LLMAT Base w/o Meta": {
        "eta": 0,
        "p_val": 0,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "label": "LLMAT Base w/o Meta (η=0)",
        "linestyle": "-"
    },
    "LLMAT Base w/o MCTS": {
        "eta": 0.005,
        "p_val": 0,
        "tree_depth": 2,
        "clustering_type": "kmeans",
        "label": "LLMAT Base w/o MCTS (tree depth=0)",
        "linestyle": "--"
    },
    "LLMAT w/ Kmeans Clustering": {
        "eta": 0.005,
        "p_val": 0.01,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "label": "LLMAT w/ Kmeans Clustering",
        "linestyle": "-."
    },
    "LLMAT w/ LLM Clustering": {
        "eta": 0.005,
        "p_val": 0.01,
        "tree_depth": 3,
        "clustering_type": "llms",
        "label": "LLMAT w/ LLM Clustering",
        "linestyle": ":"
    },
    "LLMAT Base (w/o Clustering)": {
        "eta": 0.005,
        "p_val": 0,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "selection_type": "uct",
        "label": "LLMAT Base (w/o Clustering)",
        "color": None,
        "linestyle": ":"
    },
    "LLMAT w/ Greedy Selection": {
        "eta": 0.005,
        "p_val": 0,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "selection_type": "greedy",
        "label": "LLMAT w/ Greedy Selection (UCT→Greedy)",
        "linestyle": (0, (3, 1, 1, 1))
    }
}

# # Set plot style
# sns.set_theme(style="whitegrid")
# plt.rcParams.update({'font.size': 14})
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['figure.titlesize'] = 16
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['lines.markersize'] = 8
# plt.rcParams['grid.linestyle'] = '--'
# plt.rcParams['grid.linewidth'] = 0.5
# plt.rcParams['axes.grid'] = True

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
plt.rcParams['legend.fontsize'] = 13  # Set font size for legend

fig, ax = plt.subplots(figsize=(8, 3))


def get_ablation_suffix(seed, f_model, eta, p_val, tree_depth, selection_type="uct"):
    """
    Generate suffix for ablation experiment based on the naming rules.
    For fixed feature model (finetuning=False):
        - if p_val > 0: suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{p_val}_{seed}"
        - if p_val == 0: suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    Greedy selection appends "_greedy" to the suffix.
    """
    if p_val > 0:
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{p_val}_{seed}"
    else:
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    if selection_type == "greedy":
        suffix += "_greedy"
    return suffix


def load_ablation_results(mat_bench, f_model, eta, p_val, tree_depth, seed, r_type="best_y",
                          selection_type="uct"):
    """
    Load results for a specific ablation configuration.
    """
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/llmat"
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"{path}/{clustering}"
    if tree_depth == 2:
        # MCTS depth 2 corresponds to no MCTS
        path = f"results/{prefix}/lfbo"

    suffix = get_ablation_suffix(seed, f_model, eta, p_val, tree_depth,
                                 selection_type=selection_type)
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

    for j in range(1, 101):
        y_0 = trace_by[1]
        y_t = trace_by[j]
        trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

    return trace_gap


def plot_average_gap_ablation_per_model(path="./results/figs/ablation_avg/"):
    """
    Plot average GAP across all datasets for different ablation conditions,
    with one plot per feature model.
    """
    os.makedirs(path, exist_ok=True)

    for f_model in f_models:
        print(f"\n{'='*60}")
        print(f"Processing Feature Model: {f_model}")
        print(f"{'='*60}")

        fig, ax = plt.subplots(figsize=(8, 3))

        trace_gaps_all_configs = {config_name: [] for config_name in ablation_configs.keys()}

        # Collect data for each ablation configuration
        for config_name, config in ablation_configs.items():
            eta = config["eta"]
            p_val = config["p_val"]
            tree_depth = config["tree_depth"]
            clustering_type = config["clustering_type"]
            selection_type = config.get("selection_type", "uct")

            print(f"\n  Processing {config_name}: eta={eta}, p_val={p_val}, tree_depth={tree_depth}, clustering={clustering_type}, selection={selection_type}")

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
                    trace_by = load_ablation_results(mat_bench, f_model, eta, p_val, tree_depth, seed,
                                                     r_type="best_y", selection_type=selection_type)

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        trace_gaps_all_configs[config_name].append(trace_gap)

        # Plot all ablation configurations for this feature model
        for config_name, config in ablation_configs.items():
            if len(trace_gaps_all_configs[config_name]) == 0:
                print(f"  Warning: No data found for {config_name}")
                continue

            trace_y_gaps = np.stack(trace_gaps_all_configs[config_name])
            mean = np.mean(trace_y_gaps, axis=0)[1:101]
            sem = st.sem(trace_y_gaps, axis=0)[1:101]

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

            sns.lineplot(data=df, x='T', y='y', label=config["label"], linestyle=config["linestyle"], ax=ax)
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.2)

            print(f"  {config_name}: {len(trace_y_gaps)} total runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

        plt.xlim(1, 100)
        plt.ylim(0, 1.05)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')
        plt.xlabel('BO Rounds: T', fontsize=16)
        plt.ylabel('GAP', fontsize=16)
        plt.tight_layout()
        plt.title(f"Algorithmic Modules Ablation: Average GAP Across Datasets ($\\uparrow$)")
        plt.savefig(path + f"{f_model}_GAP_ablation_average.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(path + f"{f_model}_GAP_ablation_average.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


def plot_gap_auc_ablation(path="./results/figs/ablation_avg/"):
    """
    Compute the AUC of the GAP curve (T=1..100) for every ablation
    configuration, pooled across all datasets, and compare them: prints a
    table ranked by AUC (descending) and saves a bar chart per feature
    model. Unlike the final-GAP@T=100 number printed elsewhere, AUC
    summarises sustained progress toward the optimum across the whole
    trajectory, not just the endpoint — this is what actually distinguishes
    e.g. "LLMAT Base (w/o Clustering)" (UCT selection) from "LLMAT w/
    Greedy Selection" (UCT→Greedy) when their final GAP values are close.
    """
    os.makedirs(path, exist_ok=True)
    summary_rows = []

    for f_model in f_models:
        print(f"\n{'='*60}")
        print(f"GAP AUC — Feature Model: {f_model}")
        print(f"{'='*60}")

        auc_by_config = {config_name: [] for config_name in ablation_configs.keys()}

        for config_name, config in ablation_configs.items():
            eta = config["eta"]
            p_val = config["p_val"]
            tree_depth = config["tree_depth"]
            clustering_type = config["clustering_type"]
            selection_type = config.get("selection_type", "uct")

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
                    trace_by = load_ablation_results(mat_bench, f_model, eta, p_val, tree_depth, seed,
                                                     r_type="best_y", selection_type=selection_type)

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        auc_by_config[config_name].append(np.trapz(trace_gap[1:101]))

        config_stats = []
        for config_name, config in ablation_configs.items():
            aucs = auc_by_config[config_name]
            if not aucs:
                print(f"  Warning: No data found for {config_name}")
                continue
            mean_auc, sem_auc = float(np.mean(aucs)), float(st.sem(aucs))
            config_stats.append((config_name, config["label"], mean_auc, sem_auc, len(aucs)))
            summary_rows.append({
                'f_model': f_model,
                'config': config_name,
                'label': config["label"],
                'auc_mean': mean_auc,
                'auc_sem': sem_auc,
                'n_runs': len(aucs),
            })

        config_stats.sort(key=lambda r: r[2], reverse=True)

        print(f"\n  {'Config':<45s} {'AUC (T=1..100)':>18s} {'n':>5s}")
        print(f"  {'-'*70}")
        for config_name, label, mean_auc, sem_auc, n in config_stats:
            print(f"  {label:<45s} {mean_auc:>10.2f} ± {sem_auc:<5.2f} {n:>5d}")

        if not config_stats:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [c[1] for c in config_stats]
        means = [c[2] for c in config_stats]
        sems = [c[3] for c in config_stats]
        colors = sns.color_palette("tab10", len(config_stats))

        ax.bar(range(len(labels)), means, yerr=sems, color=colors, capsize=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('AUC of GAP Curve ($\\uparrow$)', fontsize=14, fontweight='bold')
        ax.set_title(f"{f_model}: Ablation Comparison by GAP AUC", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(path + f"{f_model}_GAP_AUC_ablation_bar.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(path + f"{f_model}_GAP_AUC_ablation_bar.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(path + "gap_auc_ablation_summary.csv", index=False)
    return summary_df


def plot_average_gap_all_models_combined(path="./results/figs/ablation_avg/"):
    """
    Plot average GAP across all datasets with all feature models and ablation conditions
    on a single plot, using different colors for models and line styles for ablations.
    """
    os.makedirs(path, exist_ok=True)

    # Use seaborn color palette for feature models
    model_colors = sns.color_palette("deep", len(f_models))
    f_model_color_map = dict(zip(f_models, model_colors))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Store legend elements
    config_legend_elements = []
    config_legend_labels = []
    model_legend_elements = []
    model_legend_labels = []

    added_configs = set()
    added_models = set()

    for f_model in f_models:
        print(f"\n{'='*60}")
        print(f"Processing Feature Model: {f_model}")
        print(f"{'='*60}")

        color = f_model_color_map[f_model]

        for config_name, config in ablation_configs.items():
            eta = config["eta"]
            p_val = config["p_val"]
            tree_depth = config["tree_depth"]
            clustering_type = config["clustering_type"]
            selection_type = config.get("selection_type", "uct")

            print(f"\n  Processing {config_name}")

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
                    trace_by = load_ablation_results(mat_bench, f_model, eta, p_val, tree_depth, seed,
                                                     r_type="best_y", selection_type=selection_type)

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        trace_gaps.append(trace_gap)

            if len(trace_gaps) == 0:
                print(f"  Warning: No data found for {f_model} - {config_name}")
                continue

            trace_y_gaps = np.stack(trace_gaps)
            mean = np.mean(trace_y_gaps, axis=0)[1:101]
            sem = st.sem(trace_y_gaps, axis=0)[1:101]

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

            # Plot with model color and config line style
            ax.plot(T, mean, color=color, linestyle=config["linestyle"], label=f"{f_model}-{config['label']}", linewidth=2)
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.15, color=color)

            # Add to legend elements
            if config_name not in added_configs:
                config_legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=config["linestyle"], linewidth=2))
                config_legend_labels.append(config["label"])
                added_configs.add(config_name)

            if f_model not in added_models:
                model_legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='-', linewidth=3))
                model_legend_labels.append(f_model)
                added_models.add(f_model)

            print(f"  {f_model}-{config_name}: {len(trace_gaps)} runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.ylim(0, 1.05)
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)

    # Create two separate legends
    config_legend = ax.legend(config_legend_elements, config_legend_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), title='Ablations', fontsize=12, title_fontsize=14,
                              frameon=True, fancybox=True, shadow=False, framealpha=0.9)

    model_legend = ax.legend(model_legend_elements, model_legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), title='Feature Models', fontsize=12, title_fontsize=14,
                             frameon=True, fancybox=True, shadow=False, framealpha=0.9)

    # Add the first legend back to the axes
    ax.add_artist(config_legend)

    plt.tight_layout()
    plt.title(f"Ablation Study: Average GAP Across All Datasets ($\\uparrow$)", fontsize=18, pad=20)
    plt.savefig(path + "all_models_GAP_ablation_average.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(path + "all_models_GAP_ablation_average.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_average_gap_subplots(path="./results/figs/ablation_avg/"):
    """
    Plot average GAP with subplots: one subplot per feature model.
    Each subplot shows all ablation conditions for that model.
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

        trace_gaps_all_configs = {config_name: [] for config_name in ablation_configs.keys()}

        # Collect data for each ablation configuration
        for config_name, config in ablation_configs.items():
            eta = config["eta"]
            p_val = config["p_val"]
            tree_depth = config["tree_depth"]
            clustering_type = config["clustering_type"]
            selection_type = config.get("selection_type", "uct")

            print(f"\n  Processing {config_name}")

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
                    trace_by = load_ablation_results(mat_bench, f_model, eta, p_val, tree_depth, seed,
                                                     r_type="best_y", selection_type=selection_type)

                    if trace_by is not None:
                        trace_gap = compute_gap(trace_by, mat_bench)
                        trace_gaps_all_configs[config_name].append(trace_gap)

        # Plot all ablation configurations for this feature model
        for config_name, config in ablation_configs.items():
            if len(trace_gaps_all_configs[config_name]) == 0:
                print(f"  Warning: No data found for {config_name}")
                continue

            trace_y_gaps = np.stack(trace_gaps_all_configs[config_name])
            mean = np.mean(trace_y_gaps, axis=0)[1:101]
            sem = st.sem(trace_y_gaps, axis=0)[1:101]

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

            sns.lineplot(data=df, x='T', y='y', label=config["label"], linestyle=config["linestyle"], ax=ax)
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.2)

            print(f"  {config_name}: {len(trace_y_gaps)} runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

        ax.set_xlim(1, 100)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel('GAP', fontsize=14, fontweight='bold')
        ax.set_title(f"{f_model} ($\\uparrow$)", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(f_models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(path + "all_models_GAP_ablation_subplots.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(path + "all_models_GAP_ablation_subplots.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Plot average GAP for each feature model separately
    plot_average_gap_ablation_per_model(path="./results/figs/ablation_avg/")

    # Compare ablation configs (incl. UCT vs Greedy selection) by GAP AUC
    plot_gap_auc_ablation(path="./results/figs/ablation_avg/")

    # Plot average GAP with all models combined in one plot
    # plot_average_gap_all_models_combined(path="./results/figs/ablation_avg/")

    # Plot average GAP with subplots (one per model)
    # plot_average_gap_subplots(path="./results/figs/ablation_avg/")
