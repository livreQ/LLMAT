#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Aug 19 23:22:28 2025
# File Name: plot_cluster_time_ablation.py
# Description:
"""

from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from plotting.load_results import load_baselines
import os

seeds = list(set(range(15)))

data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
# "solvation",
f_models = ["molformer"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
prompt_type = "just-smiles"
finetuning = [True]
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]

our_algo = "llmat"
base_lines = ["laplace", "gp"]
p_vals = [0, 0.01, 0.05]

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


def get_ours_suffix(data_name, f_model, seed, n_init_data=10, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.005, tree_depth=3, finetune=True, p_val=0.05):

    if p_val > 0:
        if finetune:
            lora_lr = "3e-5"
            lora_reg = 0
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{p_val}_{seed}"
        else:
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{p_val}_{seed}"
    else:
        if finetune:
            lora_lr = "3e-5"
            lora_reg = 0
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
        else:
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    return suffix


def load_ours(mat_bench, f_model, r_type, seed, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.01, tree_depth=3, finetune=True, p_val=0):

    # print(mat_bench.dataset_name)
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/llmat"
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"{path}/{clustering}"
    data_name = mat_bench.dataset_name.split("/")[0]
    suffix = get_ours_suffix(data_name, f_model, seed, n_init_data=10, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.005, finetune=finetune, p_val=p_val)
    if not os.path.exists(f"{path}/trace_best_y_{suffix}.npy"):
        print(f"{path}/trace_best_y_{suffix}.npy")
        return None
    if r_type == "best_y":
        ret = np.load(f"{path}/trace_best_y_{suffix}.npy")
    elif r_type == "trace_time":
        ret = np.load(f"{path}/trace_timing_{suffix}.npy")
    elif r_type == "train_time":
        ret = np.load(f"{path}/timing_train_{suffix}.npy")
    elif r_type == "pred_time":
        ret = np.load(f"{path}/timing_preds_{suffix}.npy")
    elif r_type == "trace_acqvals":
        ret = np.load(f"{path}/trace_acqvals_{suffix}.npy")
    elif r_type == "trace_y_his":
        ret = np.load(f"{path}/trace_y_his_{suffix}.npy")
    else:
        ret = None
    return ret


def plot_regret_pval_ablation(path, f_models, data_names):
    """
    Plot the regret for different p_vals.
    """

    trace_regrets_alldata_ours = {"finetune_0": [], "fixed_0": [], "finetune_0.01": [], "fixed_0.01": [], "finetune_0.05": [], "fixed_0.05": []}

    def update_min_max(data):
        all_values = np.concatenate([np.ravel(arr) for arr in data])
        global_min = all_values.min()
        global_max = all_values.max()
        return global_min, global_max

    def min_max_scale(data, global_min, global_max, a=0, b=1):

        all_values = np.concatenate([np.ravel(arr) for arr in data])
        # global_min = all_values.min()
        # global_max = all_values.max()

        scaled_data = []
        for arr in data:
            arr = np.array(arr)
            arr_scaled = (arr - global_min) / (global_max - global_min) * (b - a) + a
            scaled_data.append(arr_scaled)
        return scaled_data

    for data_name in data_names:
        global_min = 10000000
        global_max = 0
        trace_regrets_ours = {}  #"finetune_" + str(pval):[], "fixed_" + str(pval): []}
        for pval in p_vals:
            for fi in finetuning:
                if fi:
                    label_mkr = "finetune"
                else:
                    label_mkr = "fixed"

                trace_regrets_ours[label_mkr + "_" + str(pval)] = []

                for f_model in f_models:
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

                        print(mat_bench.dataset_name)
                        trace_bys = []
                        trace_regrets = []
                        for seed in seeds:
                            trace_by = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, finetune=fi, p_val=pval)
                            trace_regret = deepcopy(trace_by)
                            if trace_by is not None:
                                for j in range(0, 101):
                                    if mat_bench.maximization:
                                        trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                    else:
                                        trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                                trace_bys.append(trace_by)
                                trace_regrets.append(trace_regret)
                                trace_regrets_ours[label_mkr + "_" + str(pval)].append(trace_regret)
                        cur_min, cur_max = update_min_max(trace_regrets_ours[label_mkr + "_" + str(pval)])
                        if cur_min < global_min:
                            global_min = cur_min
                        if cur_max > global_max:
                            global_max = cur_max
        print(trace_regrets_ours.keys())
        for pval in p_vals:
            for fi in finetuning:
                if fi:
                    label_mkr = "finetune"
                else:
                    label_mkr = "fixed"
                trace_regrets_ours[label_mkr + "_" + str(pval)] = min_max_scale(trace_regrets_ours[label_mkr + "_" + str(pval)], global_min, global_max)
                trace_regrets_alldata_ours[label_mkr + "_" + str(pval)] += trace_regrets_ours[label_mkr + "_" + str(pval)]

    # print(trace_regrets_alldata_ours.keys())
    for pval in p_vals:
        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"
            # print(trace_regrets_alldata_ours[label_mkr + "_" + str(pval)])
            trace_y_regrets = np.stack(trace_regrets_alldata_ours[label_mkr + "_" + str(pval)])
            # trace_y_regrets = min_max_scale(trace_y_regrets)

            mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
            sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
            if fi:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{pval}", linestyle='--')  # 'sd' shows standard deviation as shaded region
            else:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{pval}")
            ax1.fill_between(T, mean - sem, mean + sem, alpha=0.1)

    plt.yscale('log')
    plt.xlim(1, max(T))
    # plt.ylim(0.2, 0.55)
    # from matplotlib.ticker import FormatStrFormatter
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Regret', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"Average Regret w.r.t Different pvals ($\downarrow$)")
    plt.savefig(path + f_model + "_regret_finetune_average_pval_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_gap_pval_ablation(path, f_models, data_names):
    """
    Plot the gap for different tree depths.
    """

    trace_gaps_alldata_ours = {"finetune_0": [], "fixed_0": [], "finetune_0.01": [], "fixed_0.01": [], "finetune_0.05": [], "fixed_0.05": []}
    for pval in p_vals:
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
                            trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi, p_val=pval)
                            trace_gap = deepcopy(trace_by)
                            if trace_by is not None:
                                for j in range(1, 100):
                                    y_0 = trace_by[1]
                                    y_t = trace_by[j]
                                    trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                                trace_bys.append(trace_by)
                                trace_gaps.append(trace_gap)
                                trace_gaps_alldata_ours[label_mkr + "_" + str(pval)].append(trace_gap)

            trace_y_gaps_all_ours = np.stack(trace_gaps_alldata_ours[label_mkr + "_" + str(pval)])

            mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
            sem = st.sem(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
            if fi:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{pval}", linestyle='--')  # 'sd' shows standard deviation as shaded region
            else:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{pval}")
            ax1.fill_between(T, mean - sem, mean + sem, alpha=0.1)

    # plt.xscale('log')
    plt.xlim(1, max(T) - 1)
    plt.ylim(0, 1)
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"GAP for Different Cluster ($\\uparrow$)")
    plt.savefig(path + f_model + "_GAP_finetune_average_pval_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    f_models = ["molformer"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]

    plot_regret_pval_ablation(path="./results/figs/0907/pval_ablation/", f_models=f_models, data_names=data_names)
    plot_gap_pval_ablation(path="./results/figs/0907/pval_ablation/", f_models=f_models, data_names=data_names)
