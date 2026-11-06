#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Aug 19 15:08:10 2025
# File Name: plot_regret.py
# Description:
"""

###### plot regrets #########

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
finetuning = [True, False]
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]

our_algo = "llmat"

trace_regrets_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
trace_regrets_alldata_ours = {"finetune": [], "fixed": []}
# def min_max_scale(data, a=0, b=1):
#     data = np.array(data)
#     return (data - data.min()) / (data.max() - data.min()) * (b - a) + a

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


def plot_average_regret(path, f_models, data_names):
    """
    Plot the average regret for each dataset and feature model.
    """
    base_lines = ["laplace", "gp"]
    plt.figure(figsize=(6, 3))
    for data_name in data_names:
        global_min = 10000000
        global_max = 0
        trace_regrets_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
        trace_regrets_ours = {"finetune": [], "fixed": []}
        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"

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
                        trace_by = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, finetune=fi)
                        trace_regret = deepcopy(trace_by)
                        if trace_by is not None:
                            for j in range(0, 101):
                                if mat_bench.maximization:
                                    trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                else:
                                    trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                            trace_bys.append(trace_by)
                            trace_regrets.append(trace_regret)
                            trace_regrets_ours[label_mkr].append(trace_regret)
                    cur_min, cur_max = update_min_max(trace_regrets_ours[label_mkr])
                    if cur_min < global_min:
                        global_min = cur_min
                    if cur_max > global_max:
                        global_max = cur_max
                    for baseline in base_lines:
                        trace_bys = []
                        trace_regrets = []
                        for seed in seeds:
                            trace_by = load_baselines(mat_bench, r_type="trace_y_his", seed=seed, base_line=baseline, finetune=fi)
                            trace_regret = deepcopy(trace_by)
                            if trace_by is not None:
                                for j in range(0, 101):
                                    if mat_bench.maximization:
                                        trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                    else:
                                        trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                                trace_bys.append(trace_by)
                                trace_regrets.append(trace_regret)
                                trace_regrets_baseline[baseline + "_" + label_mkr].append(trace_regret)
                        cur_min, cur_max = update_min_max(trace_regrets_baseline[baseline + "_" + label_mkr])
                        if cur_min < global_min:
                            global_min = cur_min
                        if cur_max > global_max:
                            global_max = cur_max

        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"
            trace_regrets_ours[label_mkr] = min_max_scale(trace_regrets_ours[label_mkr], global_min, global_max)
            for baseline in base_lines:
                if baseline == "gp" and label_mkr == "finetune":
                    continue
                trace_regrets_baseline[baseline + "_" + label_mkr] = min_max_scale(trace_regrets_baseline[baseline + "_" + label_mkr], global_min, global_max)
                trace_regrets_alldata_baseline[baseline + "_" + label_mkr] += trace_regrets_baseline[baseline + "_" + label_mkr]

            trace_regrets_alldata_ours[label_mkr] += trace_regrets_ours[label_mkr]

    for fi in finetuning:
        if fi:
            label_mkr = "finetune"
        else:
            label_mkr = "fixed"
        trace_y_regrets = np.stack(trace_regrets_alldata_ours[label_mkr])
        # trace_y_regrets = min_max_scale(trace_y_regrets)

        mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
        sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds

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
            trace_y_regrets = np.stack(trace_regrets_alldata_baseline[baseline + "_" + label_mkr])
            # trace_y_regrets = min_max_scale(trace_y_regrets)
            mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
            sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds
            T = np.arange(len(mean)) + 1
            print(len(T), len(mean), len(sem))
            df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
            if fi:
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}", linestyle='--')
            else:
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}")
            ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

    plt.xlim(1, max(T))
    plt.setp(ax1.get_legend().get_texts(), fontsize='12')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Regret', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"Average Regret Across Datasets ($\downarrow$)")
    plt.savefig(path + f_model + "_regret_finetune_average.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_regret(path, f_models, data_names, base_lines=["laplace", "gp"]):

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
                    print(mat_bench.dataset_name)
                    trace_bys = []
                    trace_regrets = []
                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, finetune=fi)
                        trace_regret = deepcopy(trace_by)
                        if trace_by is not None:
                            for j in range(0, 101):
                                if mat_bench.maximization:
                                    trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                else:
                                    trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                            trace_bys.append(trace_by)
                            trace_regrets.append(trace_regret)

                    trace_y_his = np.stack(trace_bys)
                    trace_y_regrets = np.stack(trace_regrets)

                    mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
                    sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds

                    T = np.arange(len(mean)) + 1
                    df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                    if fi:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}", linestyle='--')  # 'sd' shows standard deviation as shaded region
                    else:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}")  # 'sd' shows standard deviation as shaded region
                    # ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{f_model}-{label_mkr}")  # 'sd' shows standard deviation as shaded region
                    ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

                    for baseline in base_lines:
                        if baseline == "gp" and label_mkr == "finetune":
                            continue
                        trace_bys = []
                        trace_regrets = []
                        for seed in seeds:
                            trace_by = load_baselines(mat_bench, r_type="trace_y_his", seed=seed, base_line=baseline, finetune=fi)
                            trace_regret = deepcopy(trace_by)
                            if trace_by is not None:
                                for j in range(0, 101):
                                    if mat_bench.maximization:
                                        trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                    else:
                                        trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                                trace_bys.append(trace_by)
                                trace_regrets.append(trace_regret)

                        trace_y_his = np.stack(trace_bys)
                        trace_y_regrets = np.stack(trace_regrets)
                        mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
                        sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds
                        T = np.arange(len(mean)) + 1
                        print(len(T), len(mean), len(sem))
                        df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                        # ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{f_model}-{label_mkr}")
                        if fi:
                            ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}", linestyle='--')
                        else:
                            ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}")
                        ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)
                if data_name == "pce":
                    data_name = "Photovoltaics"
                data_name = data_name.capitalize()
                plt.xlim(1, max(T))
                plt.setp(ax1.get_legend().get_texts(), fontsize='16')
                plt.xlabel('BO Rounds: T', fontsize=16)
                plt.ylabel('Regret', fontsize=16)
                plt.tight_layout()
                # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
                plt.title(f"{data_name}($\downarrow$)")
                plt.savefig(path + data_name + "_" + f_model + "_regret_finetune.pdf", format='pdf', bbox_inches='tight')
                plt.show()


if __name__ == "__main__":
    # f_models = ["molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
    f_models = ["t5-base-chem", "molformer"]
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
    for f_model in f_models:
        print(f"Plotting average regret for {f_model}")
        plot_average_regret(path="./results/figs/0907/Regrets/", f_models=[f_model], data_names=data_names)

    # for f_model in f_models:
    #     print(f"Plotting regret for {f_model}")
    #     plot_regret(path="./results/figs/0907/Regrets/", f_models=[f_model], data_names=data_names)
