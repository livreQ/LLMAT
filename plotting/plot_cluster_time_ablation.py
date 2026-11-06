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
plt.rcParams['lines.linewidth'] = 1  # Set line width for plots
plt.rcParams['lines.markersize'] = 8  # Set marker size for plots
plt.rcParams['grid.linestyle'] = '--'  # Set grid line style
plt.rcParams['grid.linewidth'] = 0.1  # Set grid line width
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


def plot_regret_pval_ablation(path, f_models, data_name, clustering_types=["kmeans"]):
    """
    Plot the regret for different p_vals.
    """

    # trace_regrets_alldata_ours = {"kmeans_0": [], "llms_0": [], "kmeans_0.01": [], "llms_0.01": [], "kmeans_0.05": [], "llms_0.05": []}

    trace_regrets_ours = {}  #"finetune_" + str(pval):[], "fixed_" + str(pval): []}
    for pval in p_vals:
        for fi in finetuning:
            for f_model in f_models:
                for clustering_type in clustering_types:
                    label_mkr = clustering_type
                    # if clustering_type == "kmeans" and pval == 0:
                    #     continue
                    trace_regrets_ours[label_mkr + "_" + str(pval)] = []
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

                    # trace_y_regrets = np.stack(trace_regrets_ours[label_mkr + "_" + str(pval)])
                    trace_y_regrets = np.stack(trace_regrets)
                    mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
                    sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds
                    print(label_mkr + "_" + str(pval))
                    T = np.arange(len(mean)) + 1
                    df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                    if pval == 0:
                        mk = 'o'
                        mke = 10
                    elif pval == 0.01:
                        mk = 's'
                        mke = 20
                    elif pval == 0.05:
                        mk = 'D'
                        mke = 30
                    else:
                        mk = 'x'
                    if clustering_type == "kmeans":
                        ax1 = sns.lineplot(data=df, x='T', y='y', marker=mk, markevery=mke, label=f"LLMAT-{label_mkr}-{pval}")  # 'sd' shows standard deviation as shaded region
                    else:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{pval}", marker=mk, markevery=mke, linestyle='--')
                    ax1.fill_between(T, mean - sem, mean + sem, alpha=0.1)

    plt.yscale('log')
    plt.xlim(1, max(T))
    # plt.ylim(0, 1.1)
    if data_name == "pce":
        data_name = "photovoltaics"
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Regret', fontsize=16)
    plt.tight_layout()
    plt.title(f"Regret on {data_name.capitalize()} w.r.t different pvals  ($\downarrow$)")
    plt.savefig(path + f_model + "_" + data_name + "_regret_finetune_pval_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_gap_pval_ablation(path, f_models, data_name, clustering_types=["kmeans"]):
    """
    Plot the gap for different tree depths.
    """

    trace_gaps_ours = {}

    for pval in p_vals:
        for fi in finetuning:
            for f_model in f_models:
                for clustering_type in clustering_types:
                    label_mkr = clustering_type
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

                    trace_gaps_ours[label_mkr + "_" + str(pval)] = []

                    trace_bys = []

                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi, p_val=pval)
                        trace_gap = deepcopy(trace_by)
                        if trace_by is not None:
                            for j in range(1, 100):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_bys.append(trace_by)
                            trace_gaps_ours[label_mkr + "_" + str(pval)].append(trace_gap)

                    trace_y_gaps_all_ours = np.stack(trace_gaps_ours[label_mkr + "_" + str(pval)])

                    mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
                    sem = st.sem(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
                    if pval == 0:
                        mk = 'o'
                        mke = 10
                    elif pval == 0.01:
                        mk = 's'
                        mke = 20
                    elif pval == 0.05:
                        mk = 'D'
                        mke = 30
                    else:
                        mk = 'x'
                    T = np.arange(len(mean)) + 1
                    df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                    if clustering_type == "kmeans":
                        ax1 = sns.lineplot(data=df, x='T', y='y', marker=mk, markevery=mke, label=f"LLMAT-{label_mkr}-{pval}")  # 'sd' shows standard deviation as shaded region
                    else:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{pval}", marker=mk, markevery=mke, linestyle='--')
                    ax1.fill_between(T, mean - sem, mean + sem, alpha=0.1)

    # plt.xscale('log')
    if data_name == "pce":
        data_name = "photovoltaics"
    plt.xlim(1, max(T) - 1)
    plt.ylim(0, 1.02)
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"GAP on {data_name.capitalize()} w.r.t different pvals ($\\uparrow$)")
    plt.savefig(path + f_model + "_" + data_name + "_GAP_finetune_pval_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# Add value labels on top of bars
def add_value_labels_separate(ax, data, metric):
    # Calculate means for labeling
    means = data.groupby(['p_value', 'method', 'std'])['value'].mean().reset_index()

    # Get bar positions
    for i, (p_val, group_data) in enumerate(means.groupby('p_value')):
        methods = group_data['method'].values
        values = group_data['value'].values
        std_vals = group_data['std'].values

        for j, (method, val, std_val) in enumerate(zip(methods, values, std_vals)):
            # Calculate x position (approximate)
            x_pos = i + (j - 0.5) * 0.4

            # Format label based on metric
            if metric == 'Time (s)':
                label = f'{val:.0f}s'
            else:
                label = f'{val:.3f}'

            # ax.text(x_pos, val + std_val + (ax.get_ylim()[1] * 0.02), label, ha='center', va='bottom', fontweight='bold', fontsize=9)


def add_manual_error_bars(ax, df, x_col, y_col, hue_col, error_col):

    x_values = df[x_col].unique()
    hue_values = df[hue_col].unique()

    x_positions = {val: i for i, val in enumerate(x_values)}
    hue_positions = {val: i for i, val in enumerate(hue_values)}

    n_hue = len(hue_values)
    width = 0.8 / n_hue

    for _, row in df.iterrows():
        x_pos = x_positions[row[x_col]]
        hue_pos = hue_positions[row[hue_col]]

        actual_x = x_pos + (hue_pos - (n_hue - 1) / 2) * width

        y_val = row[y_col]
        error_val = row[error_col]

        ax.errorbar(actual_x, y_val, yerr=error_val, fmt='none', color='black', capsize=3, capthick=1, elinewidth=1)


def plot_prediction_time(path, f_models, data_name, clustering_types=["kmeans"]):
    """
      plot prediction time trace over BO rounds for different p_vals for each dataset.
    """
    for pval in p_vals:
        for fi in finetuning:
            for f_model in f_models:
                for clustering_type in clustering_types:
                    label_mkr = clustering_type
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
                    print(mat_bench.dataset_name)
                    if clustering_type == "kmeans" and pval == 0:
                        continue
                    trace_times = []
                    # trace_gaps = []
                    for seed in seeds:
                        trace_time = load_ours(mat_bench, f_model, r_type="pred_time", seed=seed, finetune=fi, p_val=pval)
                        #load_ours(mat_bench, f_model, r_type="pred_time", seed=seed)
                        # print(type(trace_time))
                        # if trace_time
                        trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), trace_time[-1])])
                        # trace_time = trace_time + [trace_time[-1]] * (100 - len(trace_time))
                        # trace_gap = deepcopy(trace_time)
                        if trace_time is not None:
                            # for j in range(1, 100):
                            #     y_0 = trace_by[1]
                            #     y_t = trace_by[j]
                            #     trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_times.append(trace_time)
                            # trace_gaps.append(trace_gap)

                    trace_time_his = np.stack(trace_times)
                    # trace_y_gaps = np.stack(trace_gaps)

                    mean = np.mean(trace_time_his, axis=0)[1:101]  # Over randseeds
                    sem = st.sem(trace_time_his, axis=0)[1:101]  # Over randseeds

                    T = np.arange(len(mean)) + 1
                    df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

                    ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{clustering_type}-{pval}")  #f_model}-{clustering_type} 'sd' shows standard deviation as shaded region
                    ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

    plt.xlim(1, max(T))
    # plt.ylim(0, 1.05)
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Prediction Time', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"{data_name}($\downarrow$)")
    plt.savefig(path + data_name + "_" + f_model + "_trace_pred_time_pval_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_average_prediction_time(path, f_models, data_name, clustering_types=["kmeans"]):
    #     """
    #       plot prediction time trace over BO rounds for different p_vals for each dataset.
    #     """
    # X = []
    data = []

    for pval in p_vals:
        for fi in finetuning:
            for f_model in f_models:
                llm_time, llm_gap, llm_regret = 0, 0, 0
                kmeans_time, kmeans_gap, kmeans_regret = 0, 0, 0
                for clustering_type in clustering_types:

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

                    trace_times = []
                    trace_gaps = []
                    trace_regrets = []
                    for seed in seeds:
                        trace_time = load_ours(mat_bench, f_model, r_type="pred_time", seed=seed, finetune=fi, p_val=pval)
                        trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), trace_time[-1])])
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi, p_val=pval)
                        trace_gap = deepcopy(trace_by)
                        trace_gap[0] = 0
                        if trace_by is not None:
                            for j in range(1, 101):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_gaps.append(trace_gap)
                        if trace_time is not None:
                            trace_times.append(trace_time)

                        trace_by = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, finetune=fi, p_val=pval)
                        trace_regret = deepcopy(trace_by)
                        if trace_by is not None:
                            for j in range(0, 101):
                                if mat_bench.maximization:
                                    trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                else:
                                    trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                            trace_regrets.append(trace_regret)

                    trace_time_his = np.stack(trace_times)
                    trace_y_gaps = np.stack(trace_gaps)
                    if data_name == "photoswitch":
                        print(f"Photoswitch: {trace_time_his.shape}, {trace_y_gaps.shape}")
                        print(trace_gaps)
                        print(mat_bench.ground_truth_opt)
                    trace_y_regrets = np.stack(trace_regrets)

                    # mean = np.mean(trace_time_his, axis=0)[1:101]  # Over randseeds
                    # sem = st.sem(trace_time_his, axis=0)[1:101]  # Over randseeds
                    time_mean = np.mean(np.mean(trace_time_his, axis=0)[10:101])  # Over randseeds
                    time_sem = st.sem(trace_time_his, axis=None)
                    gap_mean = np.mean(trace_y_gaps, axis=None)
                    gap_sem = st.sem(trace_y_gaps, axis=None)
                    regret_mean = np.mean(trace_y_regrets, axis=None)
                    regret_sem = st.sem(trace_y_regrets, axis=None)
                    if clustering_type == "kmeans":
                        kmeans_time = time_mean
                        kmeans_time_err = time_sem
                        kmeans_gap = gap_mean
                        kmeans_gap_err = gap_sem
                        kmeans_regret = regret_mean
                        kmeans_regret_err = regret_sem
                    elif clustering_type == "llms":
                        llm_time = time_mean
                        llm_time_err = time_sem
                        llm_gap = gap_mean
                        llm_gap_err = gap_sem
                        llm_regret = regret_mean
                        llm_regret_err = regret_sem
                    else:
                        print(f"Unknown clustering type: {clustering_type}")
                if pval == 0:
                    pval = "0 (No Reduction)"
                    time_ref = llm_time
                    kmeans_time = llm_time
                print(time_ref)
                data.extend([{
                    'p_value': pval,
                    'method': 'LLMs',
                    'metric': 'Reduced Time (%)',
                    'value': (time_ref - llm_time) / float(time_ref) * 100,
                    'std': llm_time_err
                }, {
                    'p_value': pval,
                    'method': 'LLMs',
                    'metric': 'Gap',
                    'value': llm_gap,
                    'std': llm_gap_err
                }, {
                    'p_value': pval,
                    'method': 'LLMs',
                    'metric': 'Regret',
                    'value': llm_regret,
                    'std': llm_regret_err
                }, {
                    'p_value': pval,
                    'method': 'K-means',
                    'metric': 'Reduced Time (%)',
                    'value': (time_ref - kmeans_time) / float(time_ref) * 100,
                    'std': kmeans_time_err
                }, {
                    'p_value': pval,
                    'method': 'K-means',
                    'metric': 'Gap',
                    'value': kmeans_gap,
                    'std': kmeans_gap_err
                }, {
                    'p_value': pval,
                    'method': 'K-means',
                    'metric': 'Regret',
                    'value': kmeans_regret,
                    'std': kmeans_regret_err
                }])

    fig, axes = plt.subplots(1, 3, figsize=(16, 3))
    metrics = ['Reduced Time (%)', 'Gap', 'Regret']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    df = pd.DataFrame(data)
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]

        # Filter data for current metric
        metric_data = df[df['metric'] == metric]

        # Create bar plot
        # sns.barplot(data=metric_data, x='p_value', y='value', hue='method', ax=ax, errorbar='sd', capsize=0.2, alpha=0.8)

        sns.barplot(data=metric_data, x='p_value', y='value', hue='method', ax=ax, alpha=0.8)
        if metric != 'Reduced Time (%)':
            add_manual_error_bars(ax, metric_data, 'p_value', 'value', 'method', 'std')
        # add_value_labels_separate(ax, metric_data, metric)

        ax.set_title(f'{metric} Comparison', fontsize=10, fontweight='bold')
        ax.set_xlabel('p-value', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Customize legend
        if idx == 0:
            ax.legend(title='Method', loc='upper left')
        else:
            ax.get_legend().remove()
    if data_name == "pce":
        data_name = "photovoltaics"
    plt.tight_layout()
    plt.suptitle(f'LLMs vs K-means Across Different p-values on {data_name.capitalize()}', fontsize=14, fontweight='bold')
    plt.savefig(path + data_name + "_" + f_model + "_pval_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_average_prediction_time_arcoss_datasets(path, f_models, data_names, clustering_types=["kmeans"]):
    #     """
    #       plot prediction time trace over BO rounds for different p_vals for each dataset.
    #     """
    # X = []
    data = []
    llm_time_ipv, llm_gaps, llm_regrets = {}, {}, {}
    llm_time_ipv_err, llm_gaps_err, llm_regrets_err = {}, {}, {}
    kmeans_time_ipv, kmeans_gaps, kmeans_regrets = {}, {}, {}
    kmeans_time_ipv_err, kmeans_gaps_err, kmeans_regrets_err = {}, {}, {}

    for pval in p_vals:
        for clustering_type in clustering_types:
            llm_time_ipv[clustering_type + "_" + str(pval)] = []
            llm_gaps[clustering_type + "_" + str(pval)] = []
            llm_regrets[clustering_type + "_" + str(pval)] = []
            llm_time_ipv_err[clustering_type + "_" + str(pval)] = []
            llm_gaps_err[clustering_type + "_" + str(pval)] = []
            llm_regrets_err[clustering_type + "_" + str(pval)] = []
            kmeans_time_ipv[clustering_type + "_" + str(pval)] = []
            kmeans_gaps[clustering_type + "_" + str(pval)] = []
            kmeans_regrets[clustering_type + "_" + str(pval)] = []
            kmeans_time_ipv_err[clustering_type + "_" + str(pval)] = []
            kmeans_gaps_err[clustering_type + "_" + str(pval)] = []
            kmeans_regrets_err[clustering_type + "_" + str(pval)] = []

    for data_name in data_names:
        global_min = 10000000
        global_max = 0
        trace_regrets = {}
        trace_times = {}
        trace_gaps = {}
        for pval in p_vals:
            for fi in finetuning:
                for f_model in f_models:
                    for clustering_type in clustering_types:

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
                        trace_times[clustering_type + "_" + str(pval)] = []
                        trace_gaps[clustering_type + "_" + str(pval)] = []
                        trace_regrets[clustering_type + "_" + str(pval)] = []

                        for seed in seeds:
                            trace_time = load_ours(mat_bench, f_model, r_type="pred_time", seed=seed, finetune=fi, p_val=pval)
                            trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), trace_time[-1])])
                            trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi, p_val=pval)
                            trace_gap = deepcopy(trace_by)
                            trace_gap[0] = 0
                            if trace_by is not None:
                                for j in range(1, 101):
                                    y_0 = trace_by[1]
                                    y_t = trace_by[j]
                                    trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                                trace_gaps[clustering_type + "_" + str(pval)].append(trace_gap)
                            if trace_time is not None:
                                trace_times[clustering_type + "_" + str(pval)].append(trace_time)

                            trace_by = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, finetune=fi, p_val=pval)
                            trace_regret = deepcopy(trace_by)
                            if trace_by is not None:
                                for j in range(0, 101):
                                    if mat_bench.maximization:
                                        trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                    else:
                                        trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                                trace_regrets[clustering_type + "_" + str(pval)].append(trace_regret)

                        cur_min, cur_max = update_min_max(trace_regrets[clustering_type + "_" + str(pval)])
                        if cur_min < global_min:
                            global_min = cur_min
                        if cur_max > global_max:
                            global_max = cur_max
        for pval in p_vals:
            for clustering_type in clustering_types:
                trace_regrets[clustering_type + "_" + str(pval)] = min_max_scale(trace_regrets[clustering_type + "_" + str(pval)], global_min, global_max)

                trace_time_his = np.stack(trace_times[clustering_type + "_" + str(pval)])
                trace_y_gaps = np.stack(trace_gaps[clustering_type + "_" + str(pval)])
                trace_y_regrets = np.stack(trace_regrets[clustering_type + "_" + str(pval)])

                # mean = np.mean(trace_time_his, axis=0)[1:101]  # Over randseeds
                # sem = st.sem(trace_time_his, axis=0)[1:101]  # Over randseeds
                time_mean = np.mean(np.mean(trace_time_his, axis=0)[10:101])  # Over randseeds
                time_sem = st.sem(trace_time_his, axis=None)
                gap_mean = np.mean(trace_y_gaps, axis=None)
                gap_sem = st.sem(trace_y_gaps, axis=None)
                regret_mean = np.mean(trace_y_regrets, axis=None)
                regret_sem = st.sem(trace_y_regrets, axis=None)

                if clustering_type == "llms":

                    llm_time = time_mean
                    llm_time_err = time_sem
                    llm_gap = gap_mean
                    llm_gap_err = gap_sem
                    llm_regret = regret_mean
                    llm_regret_err = regret_sem
                    if pval == 0:
                        time_ref = llm_time
                        print(time_ref)
                    llm_time_ipv[clustering_type + "_" + str(pval)].append((time_ref - llm_time) / float(time_ref) * 100)
                    llm_gaps[clustering_type + "_" + str(pval)].append(llm_gap)
                    llm_regrets[clustering_type + "_" + str(pval)].append(llm_regret)
                    llm_time_ipv_err[clustering_type + "_" + str(pval)].append(llm_time_err)
                    llm_gaps_err[clustering_type + "_" + str(pval)].append(llm_gap_err)
                    llm_regrets_err[clustering_type + "_" + str(pval)].append(llm_regret_err)
                elif clustering_type == "kmeans":

                    kmeans_time = time_mean
                    kmeans_time_err = time_sem
                    kmeans_gap = gap_mean
                    kmeans_gap_err = gap_sem
                    kmeans_regret = regret_mean
                    kmeans_regret_err = regret_sem
                    if pval == 0:
                        kmeans_time = llm_time
                    kmeans_time_ipv[clustering_type + "_" + str(pval)].append((time_ref - kmeans_time) / float(time_ref) * 100)
                    kmeans_gaps[clustering_type + "_" + str(pval)].append(kmeans_gap)
                    kmeans_regrets[clustering_type + "_" + str(pval)].append(kmeans_regret)
                    kmeans_time_ipv_err[clustering_type + "_" + str(pval)].append(kmeans_time_err)
                    kmeans_gaps_err[clustering_type + "_" + str(pval)].append(kmeans_gap_err)
                    kmeans_regrets_err[clustering_type + "_" + str(pval)].append(kmeans_regret_err)
                else:
                    print(f"Unknown clustering type: {clustering_type}")

    for pval in p_vals:
        for clustering_type in clustering_types:
            if clustering_type == "llms":
                print("LLMs")
                print(llm_time_ipv[clustering_type + "_" + str(pval)])
                print(llm_gaps[clustering_type + "_" + str(pval)])
                print("std")
                print(llm_time_ipv_err[clustering_type + "_" + str(pval)])
                print(llm_gaps_err[clustering_type + "_" + str(pval)])
                # print(llm_regrets[clustering_type + "_" + str(pval)])
                print(llm_regrets_err[clustering_type + "_" + str(pval)])
            elif clustering_type == "kmeans":
                print("K-means")
                print(kmeans_time_ipv[clustering_type + "_" + str(pval)])
                print(kmeans_gaps[clustering_type + "_" + str(pval)])
            else:
                print(f"Unknown clustering type: {clustering_type}")
            if pval == 0:
                pval0 = "0 (No Reduction)"
            else:
                pval0 = pval
            data.extend([{
                'p_value': pval0,
                'method': 'LLMs',
                'metric': 'Reduced Time (%)',
                'value': np.mean(llm_time_ipv[clustering_type + "_" + str(pval)]),
                'std': np.mean(llm_time_ipv_err[clustering_type + "_" + str(pval)])
            }, {
                'p_value': pval0,
                'method': 'LLMs',
                'metric': 'Gap',
                'value': np.mean(llm_gaps[clustering_type + "_" + str(pval)]),
                'std': np.mean(llm_gaps_err[clustering_type + "_" + str(pval)])
            }, {
                'p_value': pval0,
                'method': 'LLMs',
                'metric': 'Regret',
                'value': np.mean(llm_regrets[clustering_type + "_" + str(pval)]),
                'std': np.mean(llm_regrets_err[clustering_type + "_" + str(pval)])
            }, {
                'p_value': pval0,
                'method': 'K-means',
                'metric': 'Reduced Time (%)',
                'value': np.mean(kmeans_time_ipv[clustering_type + "_" + str(pval)]),
                'std': np.mean(kmeans_time_ipv_err[clustering_type + "_" + str(pval)])
            }, {
                'p_value': pval0,
                'method': 'K-means',
                'metric': 'Gap',
                'value': np.mean(kmeans_gaps[clustering_type + "_" + str(pval)]),
                'std': np.mean(kmeans_gaps_err[clustering_type + "_" + str(pval)])
            }, {
                'p_value': pval0,
                'method': 'K-means',
                'metric': 'Regret',
                'value': np.mean(kmeans_regrets[clustering_type + "_" + str(pval)]),
                'std': np.mean(kmeans_regrets_err[clustering_type + "_" + str(pval)])
            }])

    fig, axes = plt.subplots(1, 3, figsize=(16, 3))
    metrics = ['Reduced Time (%)', 'Gap', 'Regret']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    df = pd.DataFrame(data)
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]

        # Filter data for current metric
        metric_data = df[df['metric'] == metric]

        # Create bar plot
        # sns.barplot(data=metric_data, x='p_value', y='value', hue='method', ax=ax, errorbar='sd', capsize=0.1, alpha=0.8)

        # add_value_labels_separate(ax, metric_data, metric)
        sns.barplot(data=metric_data, x='p_value', y='value', hue='method', ax=ax, alpha=0.8)
        if metric != 'Reduced Time (%)':
            add_manual_error_bars(ax, metric_data, 'p_value', 'value', 'method', 'std')
        ax.set_title(f'{metric} Comparison', fontsize=10, fontweight='bold')
        ax.set_xlabel('p-value', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Customize legend
        if idx == 0:
            ax.legend(title='Method', loc='upper left')
        else:
            ax.get_legend().remove()
    plt.tight_layout()
    plt.suptitle(f'LLMs vs K-means Across Different p-values on All Datasets', fontsize=14, fontweight='bold')
    plt.savefig(path + f_model + "_pval_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    f_models = ["molformer"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
    clustering_types = ["llms", "kmeans"]
    for data_name in data_names:
        plot_regret_pval_ablation(path="./results/figs/0907/pval_ablation/", f_models=f_models, data_name=data_name, clustering_types=clustering_types)
        plot_gap_pval_ablation(path="./results/figs/0907/pval_ablation/", f_models=f_models, data_name=data_name, clustering_types=clustering_types)
        # plot_prediction_time(path="./results/figs/temp/", f_models=f_models, data_name=data_name, clustering_types=clustering_types)
        # plot_average_prediction_time(path="./results/figs/0907/times/", f_models=f_models, data_name=data_name, clustering_types=clustering_types)

    plot_average_prediction_time_arcoss_datasets(path="./results/figs/0907/times/", f_models=f_models, data_names=data_names, clustering_types=clustering_types)
