#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: plot_gap_ablation.py
# Description: Plot GAP metric for ablation study comparing different configurations
"""

# GAP metric for ablation study
from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import os

seeds = list(set(range(15)))
data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
f_model = "molformer"  # Fixed to t5-base-chem
prompt_type = "just-smiles"
finetuning = False  # Fixed feature model
iupac = False
feature_reduction = "average"
run_subset_only = False

# Ablation conditions
# Full model uses llms clustering, ablations use kmeans clustering
ablation_configs = {
    "w/o Meta": {
        "eta": 0,
        "p_val": 0,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "label": "w/o Meta (η=0)",
        "color": None,
        "linestyle": "-"
    },
    "w/o MCTS": {
        "eta": 0.005,
        "p_val": 0,
        "tree_depth": 2,
        "clustering_type": "kmeans",
        "label": "w/o MCTS (depth=0)",
        "color": None,
        "linestyle": "--"
    },
    "w/ LLM Clustering": {
        "eta": 0.005,
        "p_val": 0.01,
        "tree_depth": 3,
        "clustering_type": "llms",
        "label": "w/ LLM Clustering (p=0.01)",
        "color": None,
        "linestyle": (0, (3, 1, 1, 1, 1, 1))
    },
    "w/ Kmeans Clustering": {
        "eta": 0.005,
        "p_val": 0.01,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "label": "w/ Kmeans Clustering (p=0.01)",
        "color": None,
        "linestyle": "-."
    },
    "Base Model (Meta + MCTS)": {
        "eta": 0.005,
        "p_val": 0,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "selection_type": "uct",
        "label": "Base Model (Meta + MCTS)",
        "color": None,
        "linestyle": ":"
    },
    "w/ Greedy Selection": {
        "eta": 0.005,
        "p_val": 0,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "selection_type": "greedy",
        "label": "w/ Greedy Selection (UCT→Greedy)",
        "color": None,
        "linestyle": (0, (3, 1, 1, 1))
    },
    "Decoupled (KMeans + Indep. LFBO)": {
        "eta": 0.005,
        "p_val": 0,
        "tree_depth": 3,
        "clustering_type": "kmeans",
        "selection_type": "uct",
        "algo": "decoupled_mcts_lfbo",   # stored under results/.../decoupled_mcts_lfbo/
        "suffix_tag": "_decoupled",  # appended after seed in filename
        "label": "Decoupled (KMeans + Indep. LFBO)",
        "color": None,
        "linestyle": (0, (1, 1))
    }
}

# Fixed parameters
n_init_data = 10
acqf = "ei"
gamma = 0.5

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


def get_ablation_suffix(data_name, seed, eta, p_val, tree_depth,
                        selection_type="uct", suffix_tag=""):
    """
    Generate suffix for ablation experiment based on the naming rules.
    For fixed feature model (finetuning=False):
        - if p_val > 0: suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{p_val}_{seed}"
        - if p_val == 0: suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    Greedy selection appends "_greedy"; suffix_tag is appended last (e.g. "_decoupled").
    """
    if p_val > 0:
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{p_val}_{seed}"
    else:
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    if selection_type == "greedy":
        suffix += "_greedy"
    suffix += suffix_tag
    return suffix


def load_ablation_results(mat_bench, eta, p_val, tree_depth, seed, r_type="best_y",
                          selection_type="uct", algo="llmat", suffix_tag=""):
    """
    Load results for a specific ablation configuration.
    `algo` controls the subdirectory (llmat or decoupled_mcts_lfbo).
    `suffix_tag` is appended to the filename (e.g. "_decoupled").
    """
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"results/{prefix}/{algo}/{clustering}"
    data_name = mat_bench.dataset_name.split("/")[0]

    suffix = get_ablation_suffix(data_name, seed, eta, p_val, tree_depth,
                                 selection_type=selection_type, suffix_tag=suffix_tag)

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


def plot_average_gap_ablation(path="./results/figs/ablation/"):
    """
    Plot average GAP across all datasets for different ablation conditions.
    """
    os.makedirs(path, exist_ok=True)

    trace_gaps_all_configs = {config_name: [] for config_name in ablation_configs.keys()}

    # Collect data for each ablation configuration
    for config_name, config in ablation_configs.items():
        eta = config["eta"]
        p_val = config["p_val"]
        tree_depth = config["tree_depth"]
        clustering_type = config["clustering_type"]

        selection_type = config.get("selection_type", "uct")
        algo = config.get("algo", "llmat")
        suffix_tag = config.get("suffix_tag", "")
        print(f"\nProcessing {config_name}: eta={eta}, p_val={p_val}, tree_depth={tree_depth}, "
              f"clustering={clustering_type}, selection={selection_type}, algo={algo}")

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
                trace_by = load_ablation_results(mat_bench, eta, p_val, tree_depth, seed,
                                                 r_type="best_y", selection_type=selection_type,
                                                 algo=algo, suffix_tag=suffix_tag)

                if trace_by is not None:
                    trace_gap = compute_gap(trace_by, mat_bench)
                    trace_gaps_all_configs[config_name].append(trace_gap)

    # Plot all ablation configurations
    fig, ax = plt.subplots(figsize=(6, 3))

    for config_name, config in ablation_configs.items():
        if len(trace_gaps_all_configs[config_name]) == 0:
            print(f"Warning: No data found for {config_name}")
            continue

        trace_y_gaps = np.stack(trace_gaps_all_configs[config_name])
        mean = np.mean(trace_y_gaps, axis=0)[1:101]  # Over randseeds
        sem = st.sem(trace_y_gaps, axis=0)[1:101]  # Over randseeds

        T = np.arange(len(mean)) + 1
        df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

        # Plot line with configuration-specific style
        line = sns.lineplot(data=df, x='T', y='y', label=config["label"], linestyle=config["linestyle"], ax=ax)
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.2)

        print(f"{config_name}: {len(trace_y_gaps)} total runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

    plt.xlim(1, 100)
    plt.ylim(0, 1.05)
    plt.setp(ax.get_legend().get_texts(), fontsize='14')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.tight_layout()
    plt.title(f"Ablation Study: Average GAP Across Datasets ($\\uparrow$)")
    plt.savefig(path + f"{f_model}_GAP_ablation_average.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(path + f"{f_model}_GAP_ablation_average.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_gap_ablation_per_dataset(path="./results/figs/ablation/"):
    """
    Plot GAP for each dataset separately for different ablation conditions.
    """
    os.makedirs(path, exist_ok=True)

    for data_name in data_names:
        fig, ax = plt.subplots(figsize=(6, 3))

        print(f"\n{'='*60}")
        print(f"Processing dataset: {data_name}")
        print(f"{'='*60}")

        for config_name, config in ablation_configs.items():
            eta = config["eta"]
            p_val = config["p_val"]
            tree_depth = config["tree_depth"]
            clustering_type = config["clustering_type"]
            selection_type = config.get("selection_type", "uct")
            algo = config.get("algo", "llmat")
            suffix_tag = config.get("suffix_tag", "")

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

            trace_gaps = []
            for seed in seeds:
                trace_by = load_ablation_results(mat_bench, eta, p_val, tree_depth, seed,
                                                 r_type="best_y", selection_type=selection_type,
                                                 algo=algo, suffix_tag=suffix_tag)

                if trace_by is not None:
                    trace_gap = compute_gap(trace_by, mat_bench)
                    trace_gaps.append(trace_gap)

            if len(trace_gaps) == 0:
                print(f"  {config_name}: No data found")
                continue

            trace_y_gaps = np.stack(trace_gaps)
            mean = np.mean(trace_y_gaps, axis=0)[1:101]
            sem = st.sem(trace_y_gaps, axis=0)[1:101]

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

            sns.lineplot(data=df, x='T', y='y', label=config["label"], linestyle=config["linestyle"], ax=ax)
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.2)

            print(f"  {config_name}: {len(trace_gaps)} runs, final GAP: {mean[-1]:.4f} ± {sem[-1]:.4f}")

        plt.xlim(1, 100)
        plt.ylim(0, 1.05)
        plt.setp(ax.get_legend().get_texts(), fontsize='14')
        plt.xlabel('BO Rounds: T', fontsize=16)
        plt.ylabel('GAP', fontsize=16)
        plt.tight_layout()

        # Format data name for title
        display_name = data_name
        if data_name == "pce":
            display_name = "Photovoltaics"
        elif data_name == "redox-mer":
            display_name = "Redox-Mer"
        display_name = display_name.capitalize()

        plt.title(f"Ablation Study: {display_name} ($\\uparrow$)")
        plt.savefig(path + f"{data_name}_{f_model}_GAP_ablation.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(path + f"{data_name}_{f_model}_GAP_ablation.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Plot average GAP across all datasets
    plot_average_gap_ablation(path="./results/figs/ablation/")

    # Plot GAP for each dataset separately
    # plot_gap_ablation_per_dataset(path="./results/figs/ablation/")
