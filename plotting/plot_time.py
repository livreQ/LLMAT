#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Aug 19 15:08:35 2025
# File Name: plot_time.py
# Description:
"""
import os
from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from plotting.load_results import load_baselines
from plotting.load_results import load_ours

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

seeds = list(set(range(15)))
data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
# "solvation",
f_models = ["t5-base-chem"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
prompt_type = "just-smiles"
finetuning = False
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]


def plot_training_time(path, f_models, data_names, finetuning=False, base_lines=["laplace", "gp"]):

    for data_name in data_names:
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
                print(mat_bench.dataset_name)
                trace_times = []
                # trace_gaps = []
                for seed in seeds:
                    trace_time = load_ours(mat_bench, f_model, r_type="train_time", seed=seed, finetune=finetuning)
                    # print(type(trace_time))
                    # if trace_time
                    if trace_time is not None:
                        trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), 0)])
                        trace_times.append(trace_time)

                trace_time_his = np.stack(trace_times)

                mean = np.mean(trace_time_his, axis=0)[1:101]  # Over randseeds
                sem = st.sem(trace_time_his, axis=0)[1:101]  # Over randseeds

                T = np.arange(len(mean)) + 1
                df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT")  #f_model}-{clustering_type} 'sd' shows standard deviation as shaded region
                ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            for baseline in base_lines:
                trace_times = []
                # trace_gaps = []
                for seed in seeds:
                    # trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed)
                    trace_time = load_baselines(mat_bench, r_type="train_time", seed=seed, base_line=baseline, finetune=finetuning)
                    # trace_gap = deepcopy(trace_time)
                    if trace_time is not None:
                        # for j in range(1, 100):
                        #     y_0 = trace_by[1]
                        #     y_t = trace_by[j]
                        #     trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                        trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), 0)])
                        trace_times.append(trace_time)
                        # trace_gaps.append(trace_gap)

                trace_time_his = np.stack(trace_times)
                # trace_y_gaps = np.stack(trace_gaps)
                mean = np.mean(trace_time_his, axis=0)[0:101]  # Over randseeds
                sem = st.sem(trace_time_his, axis=0)[0:101]  # Over randseeds
                T = np.arange(len(mean)) + 1
                print(len(T), len(mean), len(sem))
                df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}")
                ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            plt.xlim(1, max(T))
            # plt.ylim(0, 1.05)
            plt.setp(ax1.get_legend().get_texts(), fontsize='16')
            plt.xlabel('BO Rounds: T', fontsize=16)
            plt.ylabel('Training Time', fontsize=16)
            plt.tight_layout()
            # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
            if data_name == "pce":
                data_name = "Photovoltaics"
            data_name = data_name.capitalize()
            plt.title(f"{data_name}($\downarrow$)")
            plt.savefig(path + data_name + "_" + f_model + "_trace_training_time.pdf", format='pdf', bbox_inches='tight')
            plt.show()


def plot_prediction_time(path, f_models, data_names, finetuning=False, base_lines=["laplace", "gp"]):
    for data_name in data_names:
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
                print(mat_bench.dataset_name)
                trace_times = []
                # trace_gaps = []
                for seed in seeds:
                    trace_time = load_ours(mat_bench, f_model, r_type="pred_time", seed=seed, finetune=finetuning)
                    # print(type(trace_time))
                    # if trace_time
                    # trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), trace_time[-1])])
                    trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), 0)])
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

                mean = np.mean(trace_time_his, axis=0)[0:101]  # Over randseeds
                sem = st.sem(trace_time_his, axis=0)[0:101]  # Over randseeds

                T = np.arange(len(mean)) + 1
                df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT")  #f_model}-{clustering_type} 'sd' shows standard deviation as shaded region
                ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            for baseline in base_lines:
                trace_times = []
                # trace_gaps = []
                for seed in seeds:
                    # trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed)
                    trace_time = load_baselines(mat_bench, r_type="pred_time", seed=seed, base_line=baseline, finetune=finetuning)
                    # trace_gap = deepcopy(trace_time)
                    if trace_time is not None:
                        # for j in range(1, 100):
                        #     y_0 = trace_by[1]
                        #     y_t = trace_by[j]
                        #     trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                        # trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), trace_time[-1])])
                        trace_time = np.concatenate([trace_time, np.full(100 - len(trace_time), 0)])
                        trace_times.append(trace_time)
                        # trace_gaps.append(trace_gap)

                trace_time_his = np.stack(trace_times)
                # trace_y_gaps = np.stack(trace_gaps)
                mean = np.mean(trace_time_his, axis=0)[0:101]  # Over randseeds
                sem = st.sem(trace_time_his, axis=0)[0:101]  # Over randseeds
                T = np.arange(len(mean)) + 1
                print(len(T), len(mean), len(sem))
                df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}")
                ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            plt.xlim(1, max(T))
            # plt.ylim(0, 1.05)
            plt.setp(ax1.get_legend().get_texts(), fontsize='16')
            plt.xlabel('BO Rounds: T', fontsize=16)
            plt.ylabel('Prediction Time', fontsize=16)
            plt.tight_layout()
            # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
            if data_name == "pce":
                data_name = "Photovoltaics"
            data_name = data_name.capitalize()
            plt.title(f"{data_name}($\downarrow$)")
            plt.savefig(path + data_name + "_" + f_model + "_trace_pred_time.pdf", format='pdf', bbox_inches='tight')
            plt.show()


def plot_time(path, f_models, data_names, finetuning=False, base_lines=["laplace", "gp"]):
    """
    Plot the training and prediction time for each dataset and feature model.
    """
    for data_name in data_names:
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
                print(mat_bench.dataset_name)
                trace_times = []
                # trace_gaps = []
                for seed in seeds:
                    # trace_time = load_ours(mat_bench, r_type="trace_time", seed=seed)
                    train_time = load_ours(mat_bench, f_model, r_type="train_time", seed=seed, finetune=finetuning)
                    pred_time = load_ours(mat_bench, f_model, r_type="pred_time", seed=seed, finetune=finetuning)
                    # print(type(trace_time))
                    # if trace_time
                    # train_time = np.concatenate([train_time, np.full(100 - len(train_time), train_time[-1])])
                    # pred_time = np.concatenate([pred_time, np.full(100 - len(pred_time), pred_time[-1])])
                    train_time = np.concatenate([train_time, np.full(100 - len(train_time), 0)])
                    pred_time = np.concatenate([pred_time, np.full(100 - len(pred_time), 0)])
                    trace_time = train_time + pred_time
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

                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT")  #f_model}-{clustering_type} 'sd' shows standard deviation as shaded region
                ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            for baseline in base_lines:
                trace_times = []
                # trace_gaps = []
                for seed in seeds:
                    # trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed)
                    train_time = load_baselines(mat_bench, r_type="train_time", seed=seed, base_line=baseline, finetune=finetuning)
                    # train_time = np.concatenate([train_time, np.full(100 - len(train_time), train_time[-1])])
                    train_time = np.concatenate([train_time, np.full(100 - len(train_time), 0)])
                    pred_time = load_baselines(mat_bench, r_type="pred_time", seed=seed, base_line=baseline, finetune=finetuning)
                    # pred_time = np.concatenate([pred_time, np.full(100 - len(pred_time), pred_time[-1])])
                    pred_time = np.concatenate([pred_time, np.full(100 - len(pred_time), 0)])
                    trace_time = train_time + pred_time
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
                mean = np.mean(trace_time_his, axis=0)[0:101]  # Over randseeds
                sem = st.sem(trace_time_his, axis=0)[0:101]  # Over randseeds
                T = np.arange(len(mean)) + 1
                print(len(T), len(mean), len(sem))
                df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}")
                ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            plt.xlim(1, max(T))
            # plt.ylim(0, 1.05)
            plt.setp(ax1.get_legend().get_texts(), fontsize='16')
            plt.xlabel('BO Rounds: T', fontsize=16)
            plt.ylabel('Training + Prediction Time', fontsize=16)
            plt.tight_layout()
            # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
            if data_name == "pce":
                data_name = "Photovoltaics"
            data_name = data_name.capitalize()
            plt.title(f"{data_name}($\downarrow$)")
            plt.savefig(path + data_name + "_" + f_model + "_trace_time.pdf", format='pdf', bbox_inches='tight')
            plt.show()


def plot_time_comparison_bars(path, f_models, data_names, finetuning=False, base_lines=["laplace", "gp"]):
    """
    Create bar plots comparing GP, Ours, and Laplace methods on train time, 
    prediction time, and overall time for each dataset.
    """
    # Collect data for all datasets and methods
    results_data = []

    for data_name in data_names:
        for f_model in f_models:
            # Initialize storage for this dataset-model combination
            method_times = {}

            # Process "Ours" method
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

                train_times = []
                pred_times = []

                for seed in seeds:
                    train_time = load_ours(mat_bench, f_model, r_type="train_time", seed=seed, finetune=finetuning)
                    pred_time = load_ours(mat_bench, f_model, r_type="pred_time", seed=seed, finetune=finetuning)

                    # Pad with zeros to length 100 and compute average over all rounds
                    if train_time is not None:
                        train_time = np.concatenate([train_time, np.full(100 - len(train_time), 0)])
                        train_times.append(np.mean(train_time))

                    if pred_time is not None:
                        pred_time = np.concatenate([pred_time, np.full(100 - len(pred_time), 0)])
                        pred_times.append(np.mean(pred_time))

                if train_times and pred_times:
                    method_times['Ours'] = {
                        'train_mean': np.mean(train_times),
                        'train_sem': st.sem(train_times),
                        'pred_mean': np.mean(pred_times),
                        'pred_sem': st.sem(pred_times),
                        'total_mean': np.mean(np.array(train_times) + np.array(pred_times)),
                        'total_sem': st.sem(np.array(train_times) + np.array(pred_times))
                    }

            # Process baseline methods (GP, Laplace)
            for baseline in ['GP', 'Laplace']:  # Assuming these are in base_lines
                train_times = []
                pred_times = []

                for seed in seeds:
                    train_time = load_baselines(mat_bench, r_type="train_time", seed=seed, base_line=baseline)
                    pred_time = load_baselines(mat_bench, r_type="pred_time", seed=seed, base_line=baseline)

                    # Pad with zeros to length 100 and compute average over all rounds
                    if train_time is not None:
                        train_time = np.concatenate([train_time, np.full(100 - len(train_time), 0)])
                        train_times.append(np.mean(train_time))

                    if pred_time is not None:
                        pred_time = np.concatenate([pred_time, np.full(100 - len(pred_time), 0)])
                        pred_times.append(np.mean(pred_time))

                if train_times and pred_times:
                    method_times[baseline] = {
                        'train_mean': np.mean(train_times),
                        'train_sem': st.sem(train_times),
                        'pred_mean': np.mean(pred_times),
                        'pred_sem': st.sem(pred_times),
                        'total_mean': np.mean(np.array(train_times) + np.array(pred_times)),
                        'total_sem': st.sem(np.array(train_times) + np.array(pred_times))
                    }

            # Create DataFrame for this dataset-model combination
            for method, times in method_times.items():
                results_data.extend([{
                    'Dataset': data_name,
                    'Model': f_model,
                    'Method': method,
                    'Time_Type': 'Training Time',
                    'Mean': times['train_mean'],
                    'SEM': times['train_sem']
                }, {
                    'Dataset': data_name,
                    'Model': f_model,
                    'Method': method,
                    'Time_Type': 'Pred Time',
                    'Mean': times['pred_mean'],
                    'SEM': times['pred_sem']
                }, {
                    'Dataset': data_name,
                    'Model': f_model,
                    'Method': method,
                    'Time_Type': 'Total Time',
                    'Mean': times['total_mean'],
                    'SEM': times['total_sem']
                }])

    # Convert to DataFrame
    df_results = pd.DataFrame(results_data)

    # Create subplots for each dataset
    n_datasets = len(data_names)
    n_models = len(f_models)

    fig, axes = plt.subplots(n_datasets, n_models, figsize=(6 * n_models, 5 * n_datasets))
    if n_datasets == 1 and n_models == 1:
        axes = [[axes]]
    elif n_datasets == 1:
        axes = [axes]
    elif n_models == 1:
        axes = [[ax] for ax in axes]

    for i, data_name in enumerate(data_names):
        for j, f_model in enumerate(f_models):
            ax = axes[i][j]

            # Filter data for current dataset and model
            subset = df_results[(df_results['Dataset'] == data_name) & (df_results['Model'] == f_model)]

            if not subset.empty:
                # Create bar plot
                sns.barplot(data=subset, x='Time_Type', y='Mean', hue='Method', ax=ax, capsize=0.05, errwidth=1.5)

                # Add error bars manually
                methods = subset['Method'].unique()
                time_types = ['Training Time', 'Pred Time', 'Total Time']
                x_pos = np.arange(len(time_types))
                width = 0.8 / len(methods)

                for k, method in enumerate(methods):
                    method_data = subset[subset['Method'] == method]
                    x_positions = x_pos + (k - len(methods) / 2 + 0.5) * width

                    for l, time_type in enumerate(time_types):
                        time_data = method_data[method_data['Time_Type'] == time_type]
                        if not time_data.empty:
                            mean_val = time_data['Mean'].iloc[0]
                            sem_val = time_data['SEM'].iloc[0]
                            ax.errorbar(x_positions[l], mean_val, yerr=sem_val, fmt='none', capsize=3, capthick=1, color='black')

                ax.set_title(f'{data_name} - {f_model}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time Type', fontsize=12)
                ax.set_ylabel('Average Time per BO Round (seconds)', fontsize=12)
                ax.tick_params(axis='x', rotation=45)

                # Improve legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc='upper right', fontsize=10)

            else:
                ax.set_title(f'{data_name} - {f_model} (No Data)', fontsize=14)
                ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{path}time_comparison_barplot.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # Also create a summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Average Time per BO Round Across All Seeds")
    print("=" * 80)

    for data_name in data_names:
        print(f"\nDataset: {data_name}")
        print("-" * 50)
        dataset_data = df_results[df_results['Dataset'] == data_name]

        if not dataset_data.empty:
            summary_table = dataset_data.pivot_table(index=['Method'], columns=['Time_Type'], values='Mean', aggfunc='mean')
            print(summary_table.round(3))
        else:
            print("No data available")


# Alternative function for side-by-side comparison across datasets
def plot_time_comparison_grouped(path, f_models, data_names, finetuning=False, base_lines=["laplace", "gp"]):
    """
    Create grouped bar plots comparing methods across all datasets side by side.
    """
    # Same data collection logic as above...
    results_data = []

    for data_name in data_names:
        for f_model in f_models:
            method_times = {}

            # Process "Ours" method
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

                train_times = []
                pred_times = []

                for seed in seeds:
                    train_time = load_ours(mat_bench, f_model, r_type="train_time", seed=seed, finetune=finetuning)
                    pred_time = load_ours(mat_bench, f_model, r_type="pred_time", seed=seed, finetune=finetuning)

                    # Pad with zeros to length 100 and compute average over all rounds
                    if train_time is not None:
                        train_time = np.concatenate([train_time, np.full(100 - len(train_time), 0)])
                        train_times.append(np.mean(train_time))

                    if pred_time is not None:
                        pred_time = np.concatenate([pred_time, np.full(100 - len(pred_time), 0)])
                        pred_times.append(np.mean(pred_time))

                if train_times and pred_times:
                    method_times['LLMAT'] = {
                        'train_mean': np.mean(train_times),
                        'pred_mean': np.mean(pred_times),
                        'total_mean': np.mean(np.array(train_times) + np.array(pred_times))
                    }

            # Process baselines
            for baseline in base_lines:
                train_times = []
                pred_times = []

                for seed in seeds:
                    train_time = load_baselines(mat_bench, r_type="train_time", seed=seed, base_line=baseline, finetune=finetuning)
                    pred_time = load_baselines(mat_bench, r_type="pred_time", seed=seed, base_line=baseline, finetune=finetuning)

                    # Compute average time per BO round (excluding zeros)
                    if len(train_time) > 0:
                        non_zero_train = train_time[train_time > 0] if hasattr(train_time, '__iter__') else [train_time] if train_time > 0 else []
                        if len(non_zero_train) > 0:
                            train_times.append(np.mean(non_zero_train))

                    if len(pred_time) > 0:
                        non_zero_pred = pred_time[pred_time > 0] if hasattr(pred_time, '__iter__') else [pred_time] if pred_time > 0 else []
                        if len(non_zero_pred) > 0:
                            pred_times.append(np.mean(non_zero_pred))

                if train_times and pred_times:
                    method_times[baseline] = {
                        'train_mean': np.mean(train_times),
                        'pred_mean': np.mean(pred_times),
                        'total_mean': np.mean(np.array(train_times) + np.array(pred_times))
                    }
            if data_name == "pce":
                data_name = "Photovoltaics"
            data_name = data_name.capitalize()
            # Add to results
            for method, times in method_times.items():
                method = method.upper()
                results_data.extend([{
                    'Dataset': data_name,
                    'Method': method,
                    'Time_Type': 'Training',
                    'Time': times['train_mean']
                }, {
                    'Dataset': data_name,
                    'Method': method,
                    'Time_Type': 'Prediction',
                    'Time': times['pred_mean']
                }, {
                    'Dataset': data_name,
                    'Method': method,
                    'Time_Type': 'Total',
                    'Time': times['total_mean']
                }])

    df_results = pd.DataFrame(results_data)

    # Create grouped bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 3))
    time_types = ['Training', 'Prediction', 'Total']

    for i, time_type in enumerate(time_types):
        subset = df_results[df_results['Time_Type'] == time_type]
        sns.barplot(data=subset, x='Dataset', y='Time', hue='Method', ax=axes[i])
        axes[i].set_title(f'{time_type} Time', fontsize=16, fontweight='bold')
        # axes[i].set_xlabel('Dataset', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Average Time per BO Round (s)', fontsize=12, fontweight='bold')
        # axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    plt.savefig(f"{path}time_comparison_grouped.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    f_models = ["t5-base-chem"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
    finetuning = False
    if finetuning:
        marker = "FT"
        base_lines = ["laplace"]
    else:
        base_lines = ["laplace", "gp"]
        marker = "FX"
    # plot_training_time(path="./results/figs/0907/times/" + marker, f_models=f_models, data_names=data_names, finetuning=finetuning, base_lines=base_lines)
    # plot_prediction_time(path="./results/figs/0907/times/" + marker, f_models=f_models, data_names=data_names, finetuning=finetuning, base_lines=base_lines)
    # plot_time(path="./results/figs/0907/times/" + marker, f_models=f_models, data_names=data_names, finetuning=finetuning, base_lines=base_lines)
    # plot_time_comparison_bars(path="./results/figs/temp/times/", f_models=f_models, data_names=data_names, finetuning=finetuning)
    plot_time_comparison_grouped(path="./results/figs/0907/times/" + marker, f_models=f_models, data_names=data_names, finetuning=finetuning, base_lines=base_lines)
