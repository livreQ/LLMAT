#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Aug 19 15:07:17 2025
# File Name: plot_molformer.py
# Description:
"""

#GAP metric
from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from plotting.load_results import load_ours, load_baselines

seeds = list(set(range(15)))
data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
# "solvation",
f_models = ["t5-base-chem"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
prompt_type = "just-smiles"
# finetuning = False
finetuning = [True, False]
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]

our_algo = "llmat"
base_lines = ["laplace", "gp"]

trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
trace_gaps_alldata_ours = {"finetune": [], "fixed": []}

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
plt.figure(figsize=(6, 3))


def plot_average_gap(path="./results/figs/"):

    for fi in finetuning:
        if fi:
            label_mkr = "finetune"
        else:
            label_mkr = "fixed"
        for f_model in f_models:
            for data_name in data_names:
                for clustering_type in clustering_types:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=fi,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    trace_bys = []
                    trace_gaps = []
                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi)
                        trace_gap = deepcopy(trace_by)
                        trace_gap[0] = 0  # Set the first element to 0 for GAP calculation
                        if trace_by is not None:
                            for j in range(1, 101):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_bys.append(trace_by)
                            trace_gaps.append(trace_gap)
                            trace_gaps_alldata_ours[label_mkr].append(trace_gap)

                    for baseline in base_lines:
                        trace_bys = []
                        trace_gaps = []
                        for seed in seeds:
                            # trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed)
                            trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline, finetune=fi)
                            trace_gap = deepcopy(trace_by)
                            trace_gap[0] = 0  # Set the first element to 0 for GAP calculation
                            if trace_by is not None:
                                for j in range(1, 101):
                                    y_0 = trace_by[1]
                                    y_t = trace_by[j]
                                    trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                                trace_bys.append(trace_by)
                                trace_gaps.append(trace_gap)
                                trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)

        trace_y_gaps_all_ours = np.stack(trace_gaps_alldata_ours[label_mkr])

        mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
        sem = st.sem(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds

        T = np.arange(len(mean)) + 1
        df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
        if fi:
            ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}", linestyle='--')  # 'sd' shows standard deviation as shaded region
        else:
            ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}")  # 'sd' shows standard deviation as shaded region
        ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)
        for baseline in base_lines:
            if baseline == "gp" and label_mkr == "finetune":
                continue
            trace_y_gaps_all = np.stack(trace_gaps_alldata_baseline[baseline + "_" + label_mkr])
            mean = np.mean(trace_y_gaps_all, axis=0)[1:101]  # Over randseeds
            sem = st.sem(trace_y_gaps_all, axis=0)[1:101]  # Over randseeds
            T = np.arange(len(mean)) + 1
            print(len(T), len(mean), len(sem))
            df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
            if fi:
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}", linestyle='--')
            else:
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}")
            ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)
    if data_name == "pce":
        data_name = "Photovoltaics"
        data_name = data_name.capitalize()

    plt.xlim(1, max(T))
    plt.ylim(0, 1.05)
    plt.setp(ax1.get_legend().get_texts(), fontsize=12)
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"Average GAP Across Datasets ($\\uparrow$)")
    plt.savefig(path + f_model + "_GAP_finetune_average.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_gap(path, f_models, data_names):

    for data_name in data_names:
        for f_model in f_models:
            for clustering_type in clustering_types:
                for fi in finetuning:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=fi,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )
                    if fi:
                        label_mkr = "finetune"
                    else:
                        label_mkr = "fixed"

                    trace_bys = []
                    trace_gaps = []
                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi)
                        trace_gap = deepcopy(trace_by)
                        trace_gap[0] = 0
                        if trace_by is not None:
                            for j in range(1, 101):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_bys.append(trace_by)
                            trace_gaps.append(trace_gap)

                    trace_y_his = np.stack(trace_bys)
                    trace_y_gaps = np.stack(trace_gaps)

                    mean = np.mean(trace_y_gaps, axis=0)[1:101]  # Over randseeds
                    sem = st.sem(trace_y_gaps, axis=0)[1:101]  # Over randseeds

                    T = np.arange(len(mean)) + 1
                    df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                    if fi:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}", linestyle='--')
                    else:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}")  # 'sd' shows standard deviation as shaded region
                    ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

                    for baseline in base_lines:
                        if baseline == "gp" and label_mkr == "finetune":
                            continue
                        trace_bys = []
                        trace_gaps = []
                        for seed in seeds:
                            # trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed)
                            trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline, finetune=fi)
                            trace_gap = deepcopy(trace_by)
                            trace_gap[0] = 0
                            if trace_by is not None:
                                for j in range(1, 101):
                                    y_0 = trace_by[1]
                                    y_t = trace_by[j]
                                    trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                                trace_bys.append(trace_by)
                                trace_gaps.append(trace_gap)

                        trace_y_his = np.stack(trace_bys)
                        trace_y_gaps = np.stack(trace_gaps)
                        mean = np.mean(trace_y_gaps, axis=0)[1:101]  # Over randseeds
                        sem = st.sem(trace_y_gaps, axis=0)[1:101]  # Over randseeds
                        T = np.arange(len(mean)) + 1
                        print(len(T), len(mean), len(sem))
                        df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                        if fi:
                            ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}", linestyle='--')
                        else:
                            ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}")
                        ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)
            plt.xlim(1, max(T))
            plt.ylim(0, 1.05)
            plt.setp(ax1.get_legend().get_texts(), fontsize='16')
            plt.xlabel('BO Rounds: T', fontsize=16)
            plt.ylabel('GAP ', fontsize=16)
            plt.tight_layout()
            if data_name == "pce":
                data_name = "Photovoltaics"
            data_name = data_name.capitalize()
            # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
            plt.title(f"{data_name}($\\uparrow$)")
            plt.savefig(path + data_name + "_" + f_model + "_GAP_finetune.pdf", format='pdf', bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
    f_models = ["fingerprints"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]

    # plot_gap(path="./results/figs/0907/GAPs/", f_models=f_models, data_names=data_names)

    plot_average_gap(path="./results/figs/0907/GAPs/")
