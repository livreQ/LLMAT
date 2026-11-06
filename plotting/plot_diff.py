#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Mon Aug 18 00:58:22 2025
# File Name: plot_diff.py
# Description:
"""
import os

#GAP and Regret improvement with finetuning
from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from plotting.load_results import load_ours, load_baselines
from math import pi

seeds = list(set(range(15)))
data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
# "solvation",
f_models = ["molformer", "t5-base-chem"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
prompt_type = "just-smiles"
finetuning = [True, False]
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]

our_algo = "llmat"
base_lines = ["laplace"]  #, "gp"]

# def min_max_scale(data, a=0, b=1):
#     data = np.array(data)
#     return (data - data.min()) / (data.max() - data.min()) * (b - a) + a


def update_min_max(data):
    all_values = np.concatenate([np.ravel(arr) for arr in data])
    global_min = all_values.min()
    global_max = all_values.max()
    return global_min, global_max


def min_max_scale(data, global_min, global_max, a=0, b=1):

    # all_values = np.concatenate([np.ravel(arr) for arr in data])
    # global_min = all_values.min()
    # global_max = all_values.max()

    scaled_data = []
    for arr in data:
        arr = np.array(arr)
        arr_scaled = (arr - global_min) / (global_max - global_min) * (b - a) + a
        scaled_data.append(arr_scaled)
    return scaled_data


def plot_regret_diff(path="./results/figs/"):
    for f_model in f_models:
        trace_regrets_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
        trace_regrets_alldata_ours = {"finetune": [], "fixed": []}
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
            # max and min get, scale
            print(global_min, global_max)
            for fi in finetuning:
                if fi:
                    label_mkr = "finetune"
                else:
                    label_mkr = "fixed"
                trace_regrets_ours[label_mkr] = min_max_scale(trace_regrets_ours[label_mkr], global_min, global_max)
                trace_regrets_alldata_ours[label_mkr] += trace_regrets_ours[label_mkr]
                for baseline in base_lines:
                    if baseline == "gp" and label_mkr == "finetune":
                        continue
                    trace_regrets_baseline[baseline + "_" + label_mkr] = min_max_scale(trace_regrets_baseline[baseline + "_" + label_mkr], global_min, global_max)
                    trace_regrets_alldata_baseline[baseline + "_" + label_mkr] += trace_regrets_baseline[baseline + "_" + label_mkr]

        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
                trace_y_regrets_ours_finetune = np.stack(trace_regrets_alldata_ours[label_mkr])
            else:
                label_mkr = "fixed"
                trace_y_regrets_ours_fixed = np.stack(trace_regrets_alldata_ours[label_mkr])

            for baseline in base_lines:
                if baseline == "gp" and label_mkr == "finetune":
                    continue
                if fi:
                    trace_y_regrets_finetune = np.stack(trace_regrets_alldata_baseline[baseline + "_" + label_mkr])
                else:
                    trace_y_regrets_fixed = np.stack(trace_regrets_alldata_baseline[baseline + "_" + label_mkr])
                # trace_y_regrets = min_max_scale(trace_y_regrets)

        trace_y_regrets_ours = trace_y_regrets_ours_finetune - trace_y_regrets_ours_fixed
        mean = np.mean(trace_y_regrets_ours, axis=0)[1:101]  # Over randseeds
        sem = st.sem(trace_y_regrets_ours, axis=0)[1:101]  # Over randseeds

        T = np.arange(len(mean)) + 1
        df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{f_model}")  # 'sd' shows standard deviation as shaded region
        ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

        trace_y_regrets = trace_y_regrets_finetune - trace_y_regrets_fixed
        mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
        sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds
        T = np.arange(len(mean)) + 1
        print(len(T), len(mean), len(sem))
        df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
        ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{f_model}")
        ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

    plt.xlim(1, max(T))
    plt.setp(ax1.get_legend().get_texts(), fontsize='16')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Regret (Finetune - Fixed)', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"Average Regret Improv Across Datasets ($\downarrow$)")
    plt.savefig(path + f_model + "_regret_finetune_average_diff.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_gap_diff(path="./results/figs/"):
    for f_model in f_models:
        trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
        trace_gaps_alldata_ours = {"finetune": [], "fixed": []}
        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"
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
                        if trace_by is not None:
                            for j in range(1, 100):
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
                            if trace_by is not None:
                                for j in range(1, 100):
                                    y_0 = trace_by[1]
                                    y_t = trace_by[j]
                                    trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                                trace_bys.append(trace_by)
                                trace_gaps.append(trace_gap)
                                trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)

            if fi:
                trace_y_gaps_all_ours_finetuned = np.stack(trace_gaps_alldata_ours[label_mkr])
            else:
                trace_y_gaps_all_ours_fixed = np.stack(trace_gaps_alldata_ours[label_mkr])

            for baseline in base_lines:
                if baseline == "gp" and label_mkr == "finetune":
                    continue
                if fi:
                    trace_y_gaps_all_finetune = np.stack(trace_gaps_alldata_baseline[baseline + "_" + label_mkr])
                else:
                    trace_y_gaps_all_fixed = np.stack(trace_gaps_alldata_baseline[baseline + "_" + label_mkr])

        trace_y_gaps_all_ours = trace_y_gaps_all_ours_finetuned - trace_y_gaps_all_ours_fixed
        mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
        sem = st.sem(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds

        T = np.arange(len(mean)) + 1
        df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{f_model}")  # 'sd' shows standard deviation as shaded region
        ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)

        trace_y_gaps_all = trace_y_gaps_all_finetune - trace_y_gaps_all_fixed
        mean = np.mean(trace_y_gaps_all, axis=0)[1:100]  # Over randseeds
        sem = st.sem(trace_y_gaps_all, axis=0)[1:100]  # Over randseeds
        T = np.arange(len(mean)) + 1
        print(len(T), len(mean), len(sem))
        df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
        ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{f_model}")
        ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

    plt.xlim(1, max(T))
    # plt.ylim(0, 1.05)
    plt.setp(ax1.get_legend().get_texts(), fontsize='14')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP (Finetune - Fixed)', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"Average GAP Improv Across Datasets ($\\uparrow$)")
    plt.savefig(path + f_model + "_GAP_finetune_average_diff.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def extract_metrics_from_gap_data(gap_data, bo_rounds):
    """
    从GAP数据中提取雷达图所需的关键指标
    
    Parameters:
    gap_data: numpy array, GAP值随BO轮次的变化
    bo_rounds: numpy array, BO轮次
    
    Returns:
    dict: 包含各项指标的字典
    """
    metrics = {}

    # 1. Final Performance
    # 取最后几轮的平均值作为最终性能
    metrics['final_performance'] = np.mean(gap_data[-5:])

    # 2. 收敛速度 (Convergence Speed)
    # 找到达到80%最终性能的轮次，然后归一化
    target_performance = 0.8 * metrics['final_performance']
    convergence_round = np.argmax(gap_data >= target_performance)
    # 归一化：越早收敛得分越高
    metrics['convergence_speed'] = max(0, 1 - convergence_round / len(bo_rounds))

    # 3. 早期性能 (Early Performance)
    # 前20%轮次的平均性能
    early_rounds = int(len(bo_rounds) * 0.2)
    metrics['early_performance'] = np.mean(gap_data[:early_rounds])

    # 4. 稳定性 (Stability)
    # 基于后半段数据的方差计算，方差越小稳定性越高
    late_half = gap_data[len(gap_data) // 2:]
    variance = np.var(late_half)
    metrics['stability'] = max(0, 1 - variance * 10)  # 调整因子

    # # 5. 学习效率 (Learning Efficiency)
    # # 计算改进率的平均值
    # improvements = np.diff(gap_data)
    # positive_improvements = improvements[improvements > 0]
    # if len(positive_improvements) > 0:
    #     metrics['learning_efficiency'] = np.mean(positive_improvements) * 20  # 调整因子
    # else:
    #     metrics['learning_efficiency'] = 0

    # 6. 整体曲线下面积 (Area Under Curve)
    # 使用梯形积分计算，然后归一化
    auc = np.trapz(gap_data, bo_rounds)
    max_possible_auc = np.trapz(np.ones_like(gap_data), bo_rounds)
    metrics['auc_score'] = auc / max_possible_auc

    return metrics


def normalize_metrics(metrics_dict, max_values=None):
    """
    将指标归一化到0-1范围
    """
    if max_values is None:
        max_values = {'final_performance': 1.0, 'convergence_speed': 1.0, 'early_performance': 1.0, 'stability': 1.0, 'learning_efficiency': 1.0, 'auc_score': 1.0}

    normalized = {}
    for key, value in metrics_dict.items():
        normalized[key] = min(1.0, max(0.0, value / max_values.get(key, 1.0)))

    return normalized


# def create_radar_chart(models_data, model_names, save_path=None):
#     """
#     创建雷达图对比多个模型

#     Parameters:
#     models_data: list of dict, 每个dict包含一个模型的归一化指标
#     model_names: list of str, 模型名称
#     save_path: str, 保存路径
#     """
#     # 设置指标名称和顺序
#     metrics_labels = ['Final Performance', 'Convergence Speed', 'Early Performance', 'Stability', 'AUC Score']

#     metrics_keys = ['final_performance', 'convergence_speed', 'early_performance', 'stability', 'auc_score']

#     # 计算角度
#     num_metrics = len(metrics_labels)
#     angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
#     angles += angles[:1]  # 闭合图形

#     # 创建图形
#     fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

#     # 颜色和样式设置
#     colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12', '#9b59b6']
#     line_styles = ['-', '--', '-.', ':', '-']

#     # 为每个模型绘制数据
#     for i, (model_data, model_name) in enumerate(zip(models_data, model_names)):
#         values = [model_data[key] for key in metrics_keys]
#         values += values[:1]  # 闭合图形

#         ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], markersize=8)

#         ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

#     # 设置标签
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(metrics_labels, fontsize=12)

#     # 设置径向轴
#     ax.set_ylim(0, 1)
#     ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#     ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
#     ax.grid(True)

#     # 添加图例和标题
#     plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
#     plt.title('Model Performance Comparison\n(Radar Chart)', size=16, fontweight='bold', pad=20)

#     # 调整布局
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')

#     plt.show()


def create_radar_chart(models_data, model_names, save_path=None, figure_title=None):
    """
    创建学术风格的雷达图对比多个模型
    
    Parameters:
    models_data: list of dict, 每个dict包含一个模型的归一化指标
    model_names: list of str, 模型名称
    save_path: str, 保存路径
    figure_title: str, 图表标题
    """
    # 设置matplotlib参数以匹配学术风格
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white'
    })

    # 设置指标名称和顺序
    metrics_labels = ['Final Performance', 'Convergence Speed', 'Early Performance', 'Stability', 'AUC Score']
    metrics_keys = ['final_performance', 'convergence_speed', 'early_performance', 'stability', 'auc_score']

    # 计算角度
    num_metrics = len(metrics_labels)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]  # 闭合图形

    # 学术期刊常用的颜色方案（色盲友好）
    academic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # 线型样式
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']

    # 创建图形 - 使用标准学术比例
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 设置网格样式
    ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-', color='gray')
    ax.set_ylim(0, 1)

    # 设置径向轴刻度
    yticks = np.arange(0.2, 1.2, 0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in yticks], fontsize=9, color='black')

    # 设置角度刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=10, color='black')

    # 为每个模型绘制数据
    for i, (model_data, model_name) in enumerate(zip(models_data, model_names)):
        values = [model_data[key] for key in metrics_keys]
        values += values[:1]  # 闭合图形

        color = academic_colors[i % len(academic_colors)]
        linestyle = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        # 绘制数据线
        ax.plot(angles, values, linestyle=linestyle, linewidth=1.5, label=model_name, color=color, marker=marker, markersize=6, markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.5, alpha=0.8)

        # 添加半透明填充
        ax.fill(angles, values, alpha=0.1, color=color)

    # 美化径向轴
    ax.tick_params(axis='y', colors='black', labelsize=9)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(0.8)

    # 添加图例 - 学术风格
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=10, frameon=True, fancybox=False, shadow=False, framealpha=1.0, facecolor='white', edgecolor='black',
                       borderpad=0.5)

    # 设置图例边框
    legend.get_frame().set_linewidth(0.8)

    # 添加标题
    if figure_title:
        plt.title(figure_title, fontsize=12, fontweight='bold', color='black', pad=20)
    else:
        plt.title('Model Performance Comparison', fontsize=12, fontweight='bold', color='black', pad=20)

    # 调整布局
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    plt.show()


# 示例使用代码
def plot_radar_chart_diff(path):
    """
    基于你的GAP图像数据的示例用法
    """

    bo_rounds = np.arange(0, 100)

    all_metrics = []
    model_names = []

    for f_model in f_models:
        trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": []}
        trace_gaps_alldata_ours = {"ours_finetune": [], "ours_fixed": []}
        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"
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
                        trace_gap[0] = 0
                        if trace_by is not None:
                            for j in range(1, 101):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_bys.append(trace_by)
                            trace_gaps.append(trace_gap)
                            trace_gaps_alldata_ours["ours_" + label_mkr].append(trace_gap)

                    for baseline in base_lines:
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
                                trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)

            if fi:
                trace_y_gaps_all_ours_finetuned = np.stack(trace_gaps_alldata_ours["ours_" + label_mkr])
            else:
                trace_y_gaps_all_ours_fixed = np.stack(trace_gaps_alldata_ours["ours_" + label_mkr])

            for baseline in base_lines:
                if baseline == "gp" and label_mkr == "finetune":
                    continue
                if fi:
                    trace_y_gaps_all_finetune = np.stack(trace_gaps_alldata_baseline[baseline + "_" + label_mkr])
                else:
                    trace_y_gaps_all_fixed = np.stack(trace_gaps_alldata_baseline[baseline + "_" + label_mkr])

        trace_y_gaps_all_ours = trace_y_gaps_all_ours_finetuned - trace_y_gaps_all_ours_fixed
        ours_mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
        # sem = st.sem(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
        ours_mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]
        trace_y_gaps_all = trace_y_gaps_all_finetune - trace_y_gaps_all_fixed
        baseline_mean = np.mean(trace_y_gaps_all, axis=0)[1:101]  # Over randseeds
        # sem = st.sem(trace_y_gaps_all, axis=0)[1:101]  # Over randseeds and datasets
        ours_metrics = extract_metrics_from_gap_data(ours_mean, bo_rounds)
        ours_normalized = normalize_metrics(ours_metrics)
        all_metrics.append(ours_normalized)
        model_names.append(f'{f_model}-"ours"')
        baseline_metrics = extract_metrics_from_gap_data(baseline_mean, bo_rounds)
        baseline_normalized = normalize_metrics(baseline_metrics)
        all_metrics.append(baseline_normalized)
        model_names.append(f'{f_model}-{baseline.upper()}')

    create_radar_chart(all_metrics, model_names, path + 'model_comparison_radar.pdf')

    return all_metrics, model_names


def plot_radar(path):

    bo_rounds = np.arange(0, 100)

    all_metrics = []
    model_names = []

    for f_model in f_models:
        trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": []}
        trace_gaps_alldata_ours = {"ours_finetune": [], "ours_fixed": []}
        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"
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
                        trace_gap[0] = 0
                        if trace_by is not None:
                            for j in range(1, 101):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_bys.append(trace_by)
                            trace_gaps.append(trace_gap)
                            trace_gaps_alldata_ours["ours_" + label_mkr].append(trace_gap)

                    for baseline in base_lines:
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
                                trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)

            trace_y_gaps_all_ours = np.stack(trace_gaps_alldata_ours["ours_" + label_mkr])
            print(trace_y_gaps_all_ours.shape)
            for baseline in base_lines:
                if baseline == "gp":
                    continue
                trace_y_gaps_all_baseline = np.stack(trace_gaps_alldata_baseline[baseline + "_" + label_mkr])
            ours_mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]
            baseline_mean = np.mean(trace_y_gaps_all_baseline, axis=0)[1:101]
            ours_metrics = extract_metrics_from_gap_data(ours_mean, bo_rounds)
            ours_normalized = normalize_metrics(ours_metrics)
            all_metrics.append(ours_normalized)
            model_names.append(f'{f_model}-"ours"-{label_mkr}')
            # baseline_metrics = extract_metrics_from_gap_data(baseline_mean, bo_rounds)
            # baseline_normalized = normalize_metrics(baseline_metrics)
            # all_metrics.append(baseline_normalized)
            # model_names.append(f'{f_model}-{baseline.upper()}-{label_mkr}')

    create_radar_chart(all_metrics, model_names, path + 'model_comparison_radar.pdf')

    return all_metrics, model_names


def normalize_radar_values(data_dict, metric_names):
    """Normalize values for radar chart (0-10 scale)"""
    normalized_data = {}

    # Find min/max for each metric across all models
    metric_ranges = {}
    for metric in metric_names:
        values = [data_dict[model][metric] for model in data_dict.keys()]
        metric_ranges[metric] = {'min': min(values), 'max': max(values)}

    # Normalize each model's values
    for model in data_dict.keys():
        normalized_data[model] = {}
        for metric in metric_names:
            val = data_dict[model][metric]
            min_val = metric_ranges[metric]['min']
            max_val = metric_ranges[metric]['max']

            # Normalize to 0-10 scale
            if max_val != min_val:
                normalized_val = ((val - min_val) / (max_val - min_val)) * 10
            else:
                normalized_val = 5.0  # Default to middle if no variation

            normalized_data[model][metric] = max(0, min(10, normalized_val))

    return normalized_data


def plot_radar_chart(data_dict, metric_names, save_path, save_plots=True):
    """Create and display radar chart"""

    # Normalize data for visualization
    normalized_data = normalize_radar_values(data_dict, metric_names)

    models = list(normalized_data.keys())

    # Set up the radar chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    # Define angles for each metric (in radians)
    angles = [n / float(len(metric_names)) * 2 * pi for n in range(len(metric_names))]
    angles += angles[:1]  # Complete the circle

    # Colors for each model (extend as needed)
    colors = ['#ff7f7f', '#7fbfff', '#7fff7f', '#ffbf7f', '#ff7fff', '#7fffff']

    # Plot each model
    for i, (model, color) in enumerate(zip(models, colors[:len(models)])):
        values = [normalized_data[model][metric] for metric in metric_names]
        values += values[:1]  # Complete the circle

        # Plot the line
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color, markersize=6)

        # Fill the area
        ax.fill(angles, values, alpha=0.15, color=color)

    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')

    # Set radial limits and labels
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10, alpha=0.7)

    # Grid styling
    ax.grid(True, alpha=0.3)

    # Title and legend
    plt.title('Finetuning Effectiveness Analysis\n(Higher values = better finetuning benefits)', size=16, fontweight='bold', pad=30)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)

    # Save and display
    plt.tight_layout()

    if save_plots:
        plt.savefig(f"{save_path}finetuning_radar_analysis.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f"{save_path}finetuning_radar_analysis.png", format='png', bbox_inches='tight', dpi=300)

    plt.show()

    # Print summary
    print("\nRadar Chart Data Summary:")
    print("=" * 60)
    for model in models:
        print(f"\n{model}:")
        for metric in metric_names:
            raw_val = data_dict[model][metric]
            norm_val = normalized_data[model][metric]
            print(f"  {metric}: {raw_val:.4f} (normalized: {norm_val:.1f}/10)")


# def create_radar_plot(path, save_plots=True):
#     """
#     Create radar plots showing finetuning effectiveness across different metrics

#     Parameters:
#     -----------
#     f_models : list
#         Feature models (e.g., ['molformer', 't5-base-chem'])
#     base_lines : list
#         Baseline models (e.g., ['laplace', 'gp'])
#     data_names : list
#         Dataset names for evaluation
#     clustering_types : list
#         Clustering methods to use
#     seeds : list
#         Random seeds for experiments
#     path : str
#         Path to save plots
#     """

#     # Storage for radar chart data
#     radar_metrics = {}

#     # Define radar chart metrics
#     metric_names = ['Peak GAP Improv', 'Final GAP Improv', 'Peak Regret Improv', 'Final Regret Improv', 'GAP Peak Improv Rounds', 'Regret Peak Improv Rounds']

#     finetuning = [True, False]  # finetune vs fixed

#     for f_model in f_models:
#         print(f"Processing model: {f_model}")

#         # Initialize storage for this model
#         trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
#         trace_gaps_alldata_ours = {"finetune": [], "fixed": []}
#         trace_regrets_alldata_ours = {"finetune": [], "fixed": []}  # Add regret tracking
#         trace_regrets_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}

#         for fi in finetuning:
#             label_mkr = "finetune" if fi else "fixed"

#             for data_name in data_names:
#                 for clustering_type in clustering_types:
#                     mat_bench = MATBench(
#                         data_name=data_name,
#                         run_subset_only=run_subset_only,
#                         feature_type=f_model,
#                         finetuning=fi,
#                         iupac=iupac,
#                         prompt_type=prompt_type,
#                         randseed=665,
#                         clustering_type=clustering_type,
#                         feature_reduction=feature_reduction,
#                     )

#                     # Process "ours" method
#                     trace_bys = []
#                     trace_gaps = []
#                     trace_regrets = []

#                     for seed in seeds:
#                         trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi)

#                         if trace_by is not None:
#                             trace_gap = deepcopy(trace_by)
#                             trace_regret = deepcopy(trace_by)
#                             trace_gap[0] = 0
#                             # Calculate GAP improvement
#                             for j in range(1, 101):
#                                 y_0 = trace_by[1]
#                                 y_t = trace_by[j]
#                                 trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

#                             # Calculate regret (simple regret: ground_truth - best_found)
#                             for j in range(1, 100):
#                                 y_t = trace_by[j]
#                                 trace_regret[j] = mat_bench.ground_truth_opt - y_t

#                             trace_bys.append(trace_by)
#                             trace_gaps.append(trace_gap)
#                             trace_regrets.append(trace_regret)
#                             trace_gaps_alldata_ours[label_mkr].append(trace_gap)
#                             trace_regrets_alldata_ours[label_mkr].append(trace_regret)

#                     # Process baseline methods
#                     for baseline in base_lines:
#                         if baseline == "gp" and fi:
#                             continue  # Skip GP finetuning as in original code

#                         trace_bys = []
#                         trace_gaps = []
#                         trace_regrets = []

#                         for seed in seeds:
#                             trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline, finetune=fi)

#                             if trace_by is not None:
#                                 trace_gap = deepcopy(trace_by)
#                                 trace_regret = deepcopy(trace_by)
#                                 trace_gap[0] = 0
#                                 # Calculate GAP improvement
#                                 for j in range(1, 101):
#                                     y_0 = trace_by[1]
#                                     y_t = trace_by[j]
#                                     trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

#                                 # Calculate regret
#                                 for j in range(1, 100):
#                                     y_t = trace_by[j]
#                                     trace_regret[j] = mat_bench.ground_truth_opt - y_t

#                                 trace_bys.append(trace_by)
#                                 trace_gaps.append(trace_gap)
#                                 trace_regrets.append(trace_regret)
#                                 trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)
#                                 trace_regrets_alldata_baseline[baseline + "_" + label_mkr].append(trace_regret)

#         # Calculate improvement metrics (finetuned - fixed)
#         print(f"Calculating radar metrics for {f_model}...")

#         # Stack arrays for processing
#         trace_y_gaps_all_ours_finetuned = np.stack(trace_gaps_alldata_ours["finetune"])[1:101]
#         trace_y_gaps_all_ours_fixed = np.stack(trace_gaps_alldata_ours["fixed"])[1:101]
#         trace_y_regrets_all_ours_finetuned = np.stack(trace_regrets_alldata_ours["finetune"])[1:101]
#         trace_y_regrets_all_ours_fixed = np.stack(trace_regrets_alldata_ours["fixed"])[1:101]

#         # Calculate improvements (finetuned - fixed)
#         gap_improvement_ours = trace_y_gaps_all_ours_finetuned - trace_y_gaps_all_ours_fixed
#         regret_improvement_ours = trace_y_regrets_all_ours_fixed - trace_y_regrets_all_ours_finetuned  # Lower regret is better

#         # Calculate radar metrics for "ours" method
#         final_rounds = slice(90, 100)  # rounds 80-99 for final improvement

#         # Find peak improvement rounds and values
#         gap_mean_over_seeds = np.mean(gap_improvement_ours, axis=0)  # Average across seeds
#         regret_mean_over_seeds = np.mean(regret_improvement_ours, axis=0)  # Average across seeds

#         # Find the round where peak improvement occurs (argmax) and the peak value
#         gap_peak_round = np.argmax(gap_mean_over_seeds) + 1  # +1 because index starts at 0 but rounds start at 1
#         regret_peak_round = np.argmax(regret_mean_over_seeds) + 1
#         gap_peak_value = np.max(gap_mean_over_seeds)  # Maximum GAP improvement value
#         regret_peak_value = np.max(regret_mean_over_seeds)  # Maximum regret improvement value

#         # Convert peak rounds to a 0-10 scale (smaller rounds = higher scores for sample efficiency)
#         # Rounds 1-10 = score 10, rounds 90-100 = score 1
#         gap_peak_score = max(1, min(10, 11 - (gap_peak_round / 10)))
#         regret_peak_score = max(1, min(10, 11 - (regret_peak_round / 10)))

#         radar_metrics[f"LLMAT-{f_model}"] = {
#             'Peak GAP Improv': gap_peak_value,  # Maximum GAP improvement achieved
#             'Final GAP Improv': np.mean(gap_improvement_ours[:, final_rounds]),  # Final rounds average
#             'Peak Regret Improv': regret_peak_value,  # Maximum regret improvement achieved
#             'Final Regret Improv': np.mean(regret_improvement_ours[:, final_rounds]),  # Final rounds average
#             'GAP Peak Improv Rounds': gap_peak_score,  # Higher score = earlier peak (better sample efficiency)
#             'Regret Peak Improv Rounds': regret_peak_score  # Higher score = earlier peak (better sample efficiency)
#         }

#         # Process baselines
#         for baseline in base_lines:
#             if baseline == "gp":
#                 continue  # Skip GP as it doesn't have finetuning

#             trace_y_gaps_baseline_finetuned = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_finetune"])
#             trace_y_gaps_baseline_fixed = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_fixed"])
#             trace_y_regrets_baseline_finetuned = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_finetune"])
#             trace_y_regrets_baseline_fixed = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_fixed"])

#             gap_improvement_baseline = trace_y_gaps_baseline_finetuned - trace_y_gaps_baseline_fixed
#             regret_improvement_baseline = trace_y_regrets_baseline_fixed - trace_y_regrets_baseline_finetuned

#             # Find peak improvement rounds and values for baseline
#             gap_mean_baseline = np.mean(gap_improvement_baseline, axis=0)
#             regret_mean_baseline = np.mean(regret_improvement_baseline, axis=0)

#             gap_peak_round_baseline = np.argmax(gap_mean_baseline) + 1
#             regret_peak_round_baseline = np.argmax(regret_mean_baseline) + 1
#             gap_peak_value_baseline = np.max(gap_mean_baseline)
#             regret_peak_value_baseline = np.max(regret_mean_baseline)

#             gap_peak_score_baseline = max(1, min(10, 11 - (gap_peak_round_baseline / 10)))
#             regret_peak_score_baseline = max(1, min(10, 11 - (regret_peak_round_baseline / 10)))

#             radar_metrics[f"{baseline.upper()}-{f_model}"] = {
#                 'Peak GAP Improv': gap_peak_value_baseline,  # Maximum GAP improvement achieved
#                 'Final GAP Improv': np.mean(gap_improvement_baseline[:, final_rounds]),  # Final rounds average
#                 'Peak Regret Improv': regret_peak_value_baseline,  # Maximum regret improvement achieved
#                 'Final Regret Improv': np.mean(regret_improvement_baseline[:, final_rounds]),  # Final rounds average
#                 'GAP Peak Improv Rounds': gap_peak_score_baseline,
#                 'Regret Peak Improv Rounds': regret_peak_score_baseline
#             }

#     # Create radar plot
#     print("Creating radar plot...")
#     plot_radar_chart(radar_metrics, metric_names, path, save_plots)

#     return radar_metrics

# def normalize_radar_values(data_dict, metric_names):
#     """Normalize values for radar chart (2-10 scale) to ensure worst case shows as visible polygon"""
#     normalized_data = {}

#     # Find min/max for each metric across all models
#     metric_ranges = {}
#     for metric in metric_names:
#         values = [data_dict[model][metric] for model in data_dict.keys()]
#         metric_ranges[metric] = {'min': min(values), 'max': max(values), 'range': max(values) - min(values)}

#     # Normalize each model's values - use 2-10 scale instead of 0-10 to ensure visibility
#     MIN_RADAR_VALUE = 2.0  # Minimum radar value to ensure polygon visibility
#     MAX_RADAR_VALUE = 10.0
#     RADAR_RANGE = MAX_RADAR_VALUE - MIN_RADAR_VALUE

#     for model in data_dict.keys():
#         normalized_data[model] = {}
#         for metric in metric_names:
#             val = data_dict[model][metric]
#             min_val = metric_ranges[metric]['min']
#             max_val = metric_ranges[metric]['max']
#             val_range = metric_ranges[metric]['range']

#             # Handle different normalization strategies based on metric type
#             if 'Peak Rounds' in metric:
#                 # Peak rounds: already on good scale, but ensure 2-10 range
#                 # Original score is 1-10, rescale to 2-10
#                 original_score = val
#                 normalized_val = MIN_RADAR_VALUE + ((original_score - 1) / 9) * RADAR_RANGE
#             else:
#                 # GAP and Regret improvements: can be negative
#                 if val_range > 0:
#                     # Standard rescaling to 2-10 range
#                     normalized_val = MIN_RADAR_VALUE + ((val - min_val) / val_range) * RADAR_RANGE
#                 else:
#                     # All values are the same - put at middle
#                     normalized_val = (MIN_RADAR_VALUE + MAX_RADAR_VALUE) / 2

#                 # For mixed positive/negative values, adjust the scaling
#                 if min_val < 0 and max_val > 0:
#                     # Mixed case: zero improvement should be around score 5-6
#                     zero_point = (MIN_RADAR_VALUE + MAX_RADAR_VALUE) / 2
#                     if val >= 0:
#                         # Positive values: scale from zero_point to MAX_RADAR_VALUE
#                         if max_val > 0:
#                             normalized_val = zero_point + ((val / max_val) * (MAX_RADAR_VALUE - zero_point))
#                         else:
#                             normalized_val = zero_point
#                     else:
#                         # Negative values: scale from MIN_RADAR_VALUE to zero_point
#                         if min_val < 0:
#                             normalized_val = zero_point - ((abs(val) / abs(min_val)) * (zero_point - MIN_RADAR_VALUE))
#                         else:
#                             normalized_val = zero_point

#             # Ensure final bounds
#             normalized_val = max(MIN_RADAR_VALUE, min(MAX_RADAR_VALUE, normalized_val))
#             normalized_data[model][metric] = normalized_val

#     return normalized_data

# def plot_radar_chart(data_dict, metric_names, save_path, save_plots=True):
#     """Create and display radar chart"""

#     # Normalize data for visualization
#     normalized_data = normalize_radar_values(data_dict, metric_names)

#     models = list(normalized_data.keys())

#     # Set up the radar chart
#     fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

#     # Define angles for each metric (in radians)
#     angles = [n / float(len(metric_names)) * 2 * pi for n in range(len(metric_names))]
#     angles += angles[:1]  # Complete the circle

#     # Colors for each model (extend as needed)
#     colors = ['#ff7f7f', '#7fbfff', '#7fff7f', '#ffbf7f', '#ff7fff', '#7fffff']

#     # Plot each model
#     for i, (model, color) in enumerate(zip(models, colors[:len(models)])):
#         values = [normalized_data[model][metric] for metric in metric_names]
#         values += values[:1]  # Complete the circle

#         # Plot the line
#         ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color, markersize=6)

#         # Fill the area
#         ax.fill(angles, values, alpha=0.15, color=color)

#     # Customize the chart
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')

#     # Set radial limits and labels - adjust for 2-10 scale
#     ax.set_ylim(0, 10)
#     ax.set_yticks([2, 4, 6, 8, 10])
#     ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10, alpha=0.7)

#     # Add reference circles for better interpretation
#     ax.axhline(y=2, color='red', linestyle='--', alpha=0.3, linewidth=0.8)  # Minimum visibility line
#     ax.axhline(y=6, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)  # Neutral performance line

#     # Grid styling
#     ax.grid(True, alpha=0.3)

#     # Title and legend with updated scale info
#     plt.title('Finetuning Effectiveness Analysis\n(Scale: 2-10 for visibility, Peak = Max Value, Final = rounds 80-99, Peak Rounds: Higher = Earlier)', size=16, fontweight='bold',
#               pad=30)

#     plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)

#     # Save and display
#     plt.tight_layout()

#     if save_plots:
#         plt.savefig(f"{save_path}finetuning_radar_analysis.pdf", format='pdf', bbox_inches='tight', dpi=300)
#         plt.savefig(f"{save_path}finetuning_radar_analysis.png", format='png', bbox_inches='tight', dpi=300)

#     plt.show()

#     # Print summary with enhanced interpretation for negative values
#     print("\nRadar Chart Data Summary:")
#     print("=" * 60)
#     for model in models:
#         print(f"\n{model}:")
#         for metric in metric_names:
#             raw_val = data_dict[model][metric]
#             norm_val = normalized_data[model][metric]

#             # Add interpretation for different metric types with 2-10 scale
#             if 'Peak Rounds' in metric:
#                 # Convert score back to estimated round number for interpretation
#                 # Rescale from 2-10 back to original 1-10, then to round number
#                 original_score = 1 + ((raw_val - 2) / 8) * 9
#                 estimated_peak_round = int(11 - original_score) * 10
#                 print(f"  {metric}: {raw_val:.1f}/10 (≈round {estimated_peak_round}, normalized: {norm_val:.1f}/10)")
#             elif 'Peak GAP' in metric or 'Peak Regret' in metric:
#                 improvement_type = "POSITIVE" if raw_val > 0 else "NEGATIVE" if raw_val < 0 else "ZERO"
#                 print(f"  {metric}: {raw_val:.4f} ({improvement_type} max improvement, radar: {norm_val:.1f}/10)")
#             elif 'Final GAP' in metric or 'Final Regret' in metric:
#                 improvement_type = "POSITIVE" if raw_val > 0 else "NEGATIVE" if raw_val < 0 else "ZERO"
#                 print(f"  {metric}: {raw_val:.4f} ({improvement_type} final improvement, radar: {norm_val:.1f}/10)")
#             else:
#                 print(f"  {metric}: {raw_val:.4f} (radar: {norm_val:.1f}/10)")

#         # Enhanced interpretation guide
#         print(f"  → Radar Interpretation Guide:")
#         print(f"    • Radar 8-10: Excellent performance")
#         print(f"    • Radar 6-8:  Good performance")
#         print(f"    • Radar 4-6:  Average/Neutral performance")
#         print(f"    • Radar 2-4:  Poor performance")
#         print(f"    • Values below 6: Likely negative or minimal improvement")

#         # Enhanced peak analysis with improvement direction
#         print(f"  → Analysis:")
#         gap_peak_score = data_dict[model]['GAP Peak Rounds']
#         regret_peak_score = data_dict[model]['Regret Peak Rounds']
#         gap_peak_val = data_dict[model]['Peak GAP']
#         regret_peak_val = data_dict[model]['Peak Regret']
#         gap_final_val = data_dict[model]['Final GAP']
#         regret_final_val = data_dict[model]['Final Regret']

#         gap_peak_est = int(11 - gap_peak_score) * 10
#         regret_peak_est = int(11 - regret_peak_score) * 10

#         # GAP analysis
#         if gap_peak_val > 0:
#             gap_benefit = f"BENEFICIAL (peak: +{gap_peak_val:.3f} at ≈round {gap_peak_est})"
#         elif gap_peak_val < 0:
#             gap_benefit = f"HARMFUL (peak: {gap_peak_val:.3f} at ≈round {gap_peak_est})"
#         else:
#             gap_benefit = f"NO EFFECT (peak: {gap_peak_val:.3f} at ≈round {gap_peak_est})"

#         # Regret analysis
#         if regret_peak_val > 0:
#             regret_benefit = f"BENEFICIAL (peak: +{regret_peak_val:.3f} at ≈round {regret_peak_est})"
#         elif regret_peak_val < 0:
#             regret_benefit = f"HARMFUL (peak: {regret_peak_val:.3f} at ≈round {regret_peak_est})"
#         else:
#             regret_benefit = f"NO EFFECT (peak: {regret_peak_val:.3f} at ≈round {regret_peak_est})"

#         print(f"    GAP finetuning: {gap_benefit}")
#         print(f"    Regret finetuning: {regret_benefit}")

#         # Final vs Peak comparison
#         gap_sustain = "maintained" if abs(gap_final_val - gap_peak_val) < 0.01 else "declined" if gap_final_val < gap_peak_val else "improved"
#         regret_sustain = "maintained" if abs(regret_final_val - regret_peak_val) < 0.01 else "declined" if regret_final_val < regret_peak_val else "improved"
#         print(f"    Benefits sustainability: GAP {gap_sustain}, Regret {gap_sustain}")

#     # Summary of best/worst performers
#     print(f"\n{'='*60}")
#     print("OVERALL ASSESSMENT:")
#     print("=" * 60)

#     gap_peak_vals = [(model, data_dict[model]['Peak GAP']) for model in models]
#     regret_peak_vals = [(model, data_dict[model]['Peak Regret']) for model in models]

#     best_gap = max(gap_peak_vals, key=lambda x: x[1])
#     worst_gap = min(gap_peak_vals, key=lambda x: x[1])
#     best_regret = max(regret_peak_vals, key=lambda x: x[1])
#     worst_regret = min(regret_peak_vals, key=lambda x: x[1])

#     print(f"Best GAP finetuning: {best_gap[0]} (+{best_gap[1]:.4f})")
#     print(f"Worst GAP finetuning: {worst_gap[0]} ({worst_gap[1]:.4f})")
#     print(f"Best Regret finetuning: {best_regret[0]} (+{best_regret[1]:.4f})")
#     print(f"Worst Regret finetuning: {worst_regret[0]} ({worst_regret[1]:.4f})")

#     # Identify models where finetuning hurts performance
#     harmful_models = []
#     for model in models:
#         gap_harmful = data_dict[model]['Peak GAP'] < 0
#         regret_harmful = data_dict[model]['Peak Regret'] < 0
#         if gap_harmful or regret_harmful:
#             issues = []
#             if gap_harmful:
#                 issues.append("GAP")
#             if regret_harmful:
#                 issues.append("Regret")
#             harmful_models.append(f"{model} ({'/'.join(issues)} degraded)")

#     if harmful_models:
#         print(f"\n⚠️  Models where finetuning hurts performance:")
#         for model in harmful_models:
#             print(f"   • {model}")
#     else:
#         print(f"\n✅ All models benefit from finetuning!")

# def create_radar_plot(path, save_plots=True):
#     """
#     Create radar plots showing finetuning effectiveness across different metrics

#     Parameters:
#     -----------
#     f_models : list
#         Feature models (e.g., ['molformer', 't5-base-chem'])
#     base_lines : list
#         Baseline models (e.g., ['laplace', 'gp'])
#     data_names : list
#         Dataset names for evaluation
#     clustering_types : list
#         Clustering methods to use
#     seeds : list
#         Random seeds for experiments
#     path : str
#         Path to save plots
#     """

#     # Storage for radar chart data
#     radar_metrics = {}

#     # Define radar chart metrics
#     metric_names = ['Peak GAP Improv', 'Final GAP Improv', 'Peak Regret Improv', 'Final Regret Improv', 'GAP Peak Improv Rounds', 'Regret Peak Improv Rounds']

#     finetuning = [True, False]  # finetune vs fixed

#     for f_model in f_models:
#         print(f"Processing model: {f_model}")

#         # Initialize storage for this model
#         trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
#         trace_gaps_alldata_ours = {"finetune": [], "fixed": []}
#         trace_regrets_alldata_ours = {"finetune": [], "fixed": []}  # Add regret tracking
#         trace_regrets_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}

#         for fi in finetuning:
#             label_mkr = "finetune" if fi else "fixed"

#             for data_name in data_names:
#                 for clustering_type in clustering_types:
#                     mat_bench = MATBench(
#                         data_name=data_name,
#                         run_subset_only=run_subset_only,
#                         feature_type=f_model,
#                         finetuning=fi,
#                         iupac=iupac,
#                         prompt_type=prompt_type,
#                         randseed=665,
#                         clustering_type=clustering_type,
#                         feature_reduction=feature_reduction,
#                     )

#                     # Process "ours" method
#                     trace_bys = []
#                     trace_gaps = []
#                     trace_regrets = []

#                     for seed in seeds:
#                         trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi)

#                         if trace_by is not None:
#                             trace_gap = deepcopy(trace_by)
#                             trace_regret = deepcopy(trace_by)
#                             trace_gap[0] = 0
#                             # Calculate GAP improvement
#                             for j in range(1, 101):
#                                 y_0 = trace_by[1]
#                                 y_t = trace_by[j]
#                                 trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

#                             # Calculate regret (simple regret: ground_truth - best_found)
#                             for j in range(1, 100):
#                                 y_t = trace_by[j]
#                                 trace_regret[j] = mat_bench.ground_truth_opt - y_t

#                             trace_bys.append(trace_by)
#                             trace_gaps.append(trace_gap)
#                             trace_regrets.append(trace_regret)
#                             trace_gaps_alldata_ours[label_mkr].append(trace_gap)
#                             trace_regrets_alldata_ours[label_mkr].append(trace_regret)

#                     # Process baseline methods
#                     for baseline in base_lines:
#                         if baseline == "gp" and fi:
#                             continue  # Skip GP finetuning as in original code

#                         trace_bys = []
#                         trace_gaps = []
#                         trace_regrets = []

#                         for seed in seeds:
#                             trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline, finetune=fi)

#                             if trace_by is not None:
#                                 trace_gap = deepcopy(trace_by)
#                                 trace_regret = deepcopy(trace_by)
#                                 trace_gap[0] = 0
#                                 # Calculate GAP improvement
#                                 for j in range(1, 101):
#                                     y_0 = trace_by[1]
#                                     y_t = trace_by[j]
#                                     trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

#                                 # Calculate regret
#                                 for j in range(1, 100):
#                                     y_t = trace_by[j]
#                                     trace_regret[j] = mat_bench.ground_truth_opt - y_t

#                                 trace_bys.append(trace_by)
#                                 trace_gaps.append(trace_gap)
#                                 trace_regrets.append(trace_regret)
#                                 trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)
#                                 trace_regrets_alldata_baseline[baseline + "_" + label_mkr].append(trace_regret)

#         # Calculate improvement metrics (finetuned - fixed)
#         print(f"Calculating radar metrics for {f_model}...")

#         # Stack arrays for processing
#         trace_y_gaps_all_ours_finetuned = np.stack(trace_gaps_alldata_ours["finetune"])[1:101]
#         trace_y_gaps_all_ours_fixed = np.stack(trace_gaps_alldata_ours["fixed"])[1:101]
#         trace_y_regrets_all_ours_finetuned = np.stack(trace_regrets_alldata_ours["finetune"])[1:101]
#         trace_y_regrets_all_ours_fixed = np.stack(trace_regrets_alldata_ours["fixed"])[1:101]

#         # Calculate improvements (finetuned - fixed)
#         gap_improvement_ours = trace_y_gaps_all_ours_finetuned - trace_y_gaps_all_ours_fixed
#         regret_improvement_ours = trace_y_regrets_all_ours_fixed - trace_y_regrets_all_ours_finetuned  # Lower regret is better

#         # Calculate radar metrics for "ours" method
#         final_rounds = slice(90, 100)  # rounds 80-99 for final improvement

#         # Find peak improvement rounds and values
#         gap_mean_over_seeds = np.mean(gap_improvement_ours, axis=0)  # Average across seeds
#         regret_mean_over_seeds = np.mean(regret_improvement_ours, axis=0)  # Average across seeds

#         # Find the round where peak improvement occurs (argmax) and the peak value
#         gap_peak_round = np.argmax(gap_mean_over_seeds) + 1  # +1 because index starts at 0 but rounds start at 1
#         regret_peak_round = np.argmax(regret_mean_over_seeds) + 1
#         gap_peak_value = np.max(gap_mean_over_seeds)  # Maximum GAP improvement value
#         regret_peak_value = np.max(regret_mean_over_seeds)  # Maximum regret improvement value

#         # Convert peak rounds to a 0-10 scale (smaller rounds = higher scores for sample efficiency)
#         # Rounds 1-10 = score 10, rounds 90-100 = score 1
#         gap_peak_score = max(1, min(10, 11 - (gap_peak_round / 10)))
#         regret_peak_score = max(1, min(10, 11 - (regret_peak_round / 10)))

#         radar_metrics[f"LLMAT-{f_model}"] = {
#             'Peak GAP Improv': gap_peak_value,  # Maximum GAP improvement achieved
#             'Final GAP Improv': np.mean(gap_improvement_ours[:, final_rounds]),  # Final rounds average
#             'Peak Regret Improv': regret_peak_value,  # Maximum regret improvement achieved
#             'Final Regret Improv': np.mean(regret_improvement_ours[:, final_rounds]),  # Final rounds average
#             'GAP Peak Improv Rounds': gap_peak_score,  # Higher score = earlier peak (better sample efficiency)
#             'Regret Peak Improv Rounds': regret_peak_score  # Higher score = earlier peak (better sample efficiency)
#         }

#         # Process baselines
#         for baseline in base_lines:
#             if baseline == "gp":
#                 continue  # Skip GP as it doesn't have finetuning

#             trace_y_gaps_baseline_finetuned = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_finetune"])
#             trace_y_gaps_baseline_fixed = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_fixed"])
#             trace_y_regrets_baseline_finetuned = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_finetune"])
#             trace_y_regrets_baseline_fixed = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_fixed"])

#             gap_improvement_baseline = trace_y_gaps_baseline_finetuned - trace_y_gaps_baseline_fixed
#             regret_improvement_baseline = trace_y_regrets_baseline_fixed - trace_y_regrets_baseline_finetuned

#             # Find peak improvement rounds and values for baseline
#             gap_mean_baseline = np.mean(gap_improvement_baseline, axis=0)
#             regret_mean_baseline = np.mean(regret_improvement_baseline, axis=0)

#             gap_peak_round_baseline = np.argmax(gap_mean_baseline) + 1
#             regret_peak_round_baseline = np.argmax(regret_mean_baseline) + 1
#             gap_peak_value_baseline = np.max(gap_mean_baseline)
#             regret_peak_value_baseline = np.max(regret_mean_baseline)

#             gap_peak_score_baseline = max(1, min(10, 11 - (gap_peak_round_baseline / 10)))
#             regret_peak_score_baseline = max(1, min(10, 11 - (regret_peak_round_baseline / 10)))

#             radar_metrics[f"{baseline.upper()}-{f_model}"] = {
#                 'Peak GAP Improv': gap_peak_value_baseline,  # Maximum GAP improvement achieved
#                 'Final GAP Improv': np.mean(gap_improvement_baseline[:, final_rounds]),  # Final rounds average
#                 'Peak Regret Improv': regret_peak_value_baseline,  # Maximum regret improvement achieved
#                 'Final Regret Improv': np.mean(regret_improvement_baseline[:, final_rounds]),  # Final rounds average
#                 'GAP Peak Improv Rounds': gap_peak_score_baseline,
#                 'Regret Peak Improv Rounds': regret_peak_score_baseline
#             }

#     # Create radar plot
#     print("Creating radar plot...")
#     plot_academic_radar_chart(radar_metrics, metric_names, path, save_plots)

#     return radar_metrics

# def normalize_radar_values(data_dict, metric_names):
#     """Normalize values for radar chart (0-1 scale) to match academic style"""
#     normalized_data = {}

#     # Find min/max for each metric across all models
#     metric_ranges = {}
#     for metric in metric_names:
#         values = [data_dict[model][metric] for model in data_dict.keys()]
#         metric_ranges[metric] = {'min': min(values), 'max': max(values), 'range': max(values) - min(values)}

#     # Normalize each model's values to 0-1 scale
#     for model in data_dict.keys():
#         normalized_data[model] = {}
#         for metric in metric_names:
#             val = data_dict[model][metric]
#             min_val = metric_ranges[metric]['min']
#             max_val = metric_ranges[metric]['max']
#             val_range = metric_ranges[metric]['range']

#             # Handle different normalization strategies
#             if 'Peak Rounds' in metric:
#                 # Peak rounds: already on 1-10 scale, normalize to 0-1
#                 normalized_val = (val - 1) / 9
#             else:
#                 # GAP and Regret improvements: handle negative values properly
#                 if val_range > 0:
#                     normalized_val = (val - min_val) / val_range
#                 else:
#                     normalized_val = 0.5  # Neutral value when all are the same

#                 # For mixed positive/negative values, center zero at 0.5
#                 if min_val < 0 and max_val > 0:
#                     if val >= 0:
#                         normalized_val = 0.5 + (val / max_val) * 0.5
#                     else:
#                         normalized_val = 0.5 - (abs(val) / abs(min_val)) * 0.5

#             # Ensure bounds [0, 1]
#             normalized_val = max(0.0, min(1.0, normalized_val))
#             normalized_data[model][metric] = normalized_val

#     return normalized_data

# def plot_academic_radar_chart(data_dict, metric_names, save_path, save_plots=True):
#     """Create academic-style radar chart matching the reference image"""

#     # Set academic style parameters
#     plt.rcParams.update({
#         'font.size': 11,
#         'font.family': 'sans-serif',
#         'axes.linewidth': 0.8,
#         'axes.edgecolor': '#333333',
#         'xtick.color': '#333333',
#         'ytick.color': '#333333',
#         'text.color': '#333333'
#     })

#     # Normalize data for visualization
#     normalized_data = normalize_radar_values(data_dict, metric_names)
#     models = list(normalized_data.keys())

#     # Create figure with clean background
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('#F8F8F8')  # Very light gray background

#     # Define angles for each metric (in radians)
#     angles = [n / float(len(metric_names)) * 2 * pi for n in range(len(metric_names))]
#     angles += angles[:1]  # Complete the circle

#     # Academic color scheme - professional and color-blind friendly
#     colors = ['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
#     line_styles = ['-', '--', '-.', ':', '-', '--']
#     markers = ['o', 's', '^', 'v', 'D', 'p']

#     # Set up radial grid with academic styling
#     ax.set_ylim(0, 1)

#     # Custom radial ticks (like the reference image)
#     radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
#     ax.set_yticks(radial_ticks)
#     ax.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], fontsize=9, color='#666666', alpha=0.7)

#     # Style the grid to match academic papers
#     ax.grid(True, alpha=0.3, linewidth=0.8, color='#CCCCCC')
#     ax.spines['polar'].set_visible(False)

#     # Plot each model with academic styling
#     for i, model in enumerate(models):
#         values = [normalized_data[model][metric] for metric in metric_names]
#         values += values[:1]  # Complete the circle

#         color = colors[i % len(colors)]
#         linestyle = line_styles[i % len(line_styles)]
#         marker = markers[i % len(markers)]

#         # Plot line with academic styling
#         ax.plot(angles, values, linestyle=linestyle, linewidth=2.0, color=color, marker=marker, markersize=6, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.0,
#                 alpha=0.9, label=model)

#         # Fill area with low alpha for subtle effect
#         ax.fill(angles, values, alpha=0.08, color=color)

#     # Set angular labels with clean formatting
#     ax.set_xticks(angles[:-1])

#     # Shorten labels to match academic style
#     short_labels = ['Peak GAP\nImprov', 'Final GAP\nImprov', 'Peak Regret\nImprov', 'Final Regret\nImprov', 'GAP Peak\nRounds', 'Regret Peak\nRounds']

#     ax.set_xticklabels(short_labels, fontsize=10, color='#333333', ha='center')

#     # Academic-style legend
#     legend = ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10, frameon=True, fancybox=False, shadow=False, framealpha=1.0, facecolor='white',
#                        edgecolor='#CCCCCC', borderpad=0.8)

#     # Style legend frame
#     legend.get_frame().set_linewidth(0.8)
#     for text in legend.get_texts():
#         text.set_color('#333333')

#     # Clean, academic title
#     plt.title('Finetuning Effectiveness Analysis', fontsize=14, fontweight='bold', color='#333333', pad=25)

#     # Add subtle radial reference lines (like in the reference image)
#     for tick in [0.2, 0.4, 0.6, 0.8]:
#         circle = plt.Circle((0, 0), tick, transform=ax.transData._b, fill=False, color='#E0E0E0', alpha=0.6, linewidth=0.8)
#         ax.add_patch(circle)

#     plt.tight_layout()

#     # Save with academic quality
#     if save_plots:
#         plt.savefig(f"{save_path}finetuning_radar_analysis_academic.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
#         plt.savefig(f"{save_path}finetuning_radar_analysis_academic.png", format='png', bbox_inches='tight', dpi=300, facecolor='white')

#     plt.show()

#     # Enhanced summary output
#     print_academic_summary(data_dict, normalized_data, metric_names, models)

# def print_academic_summary(data_dict, normalized_data, metric_names, models):
#     """Print academic-style summary of results"""

#     print("\n" + "=" * 70)
#     print("FINETUNING EFFECTIVENESS ANALYSIS - ACADEMIC SUMMARY")
#     print("=" * 70)

#     # Performance ranking
#     print("\nModel Performance Ranking (by average normalized score):")
#     print("-" * 55)

#     avg_scores = {}
#     for model in models:
#         avg_score = np.mean([normalized_data[model][metric] for metric in metric_names])
#         avg_scores[model] = avg_score

#     ranked_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

#     for rank, (model, score) in enumerate(ranked_models, 1):
#         print(f"{rank:2d}. {model:<20} Average Score: {score:.3f}")

#     # Key findings
#     print(f"\nKey Findings:")
#     print("-" * 20)

#     # Best performer per metric
#     for metric in metric_names:
#         best_model = max(models, key=lambda m: data_dict[m][metric])
#         best_value = data_dict[best_model][metric]
#         print(f"• {metric:<25}: {best_model} ({best_value:.4f})")

#     # Identify beneficial vs harmful finetuning
#     beneficial_models = []
#     harmful_models = []

#     for model in models:
#         gap_metrics = ['Peak GAP Improv', 'Final GAP Improv']
#         regret_metrics = ['Peak Regret Improv', 'Final Regret Improv']

#         gap_beneficial = any(data_dict[model][m] > 0 for m in gap_metrics)
#         regret_beneficial = any(data_dict[model][m] > 0 for m in regret_metrics)

#         if gap_beneficial or regret_beneficial:
#             beneficial_models.append(model)
#         else:
#             harmful_models.append(model)

#     print(f"\nFinetuning Impact Assessment:")
#     print("-" * 35)
#     print(f"✓ Models benefiting from finetuning: {len(beneficial_models)}")
#     for model in beneficial_models:
#         print(f"   • {model}")

#     if harmful_models:
#         print(f"✗ Models harmed by finetuning: {len(harmful_models)}")
#         for model in harmful_models:
#             print(f"   • {model}")

#     print(f"\nStatistical Summary:")
#     print("-" * 25)
#     for metric in metric_names:
#         values = [data_dict[model][metric] for model in models]
#         print(f"{metric:<25}: μ={np.mean(values):6.4f}, σ={np.std(values):6.4f}")

# def create_radar_plot(path, save_plots=True):
#     """
#     Create radar plots showing finetuning effectiveness across different metrics

#     Parameters:
#     -----------
#     f_models : list
#         Feature models (e.g., ['molformer', 't5-base-chem'])
#     base_lines : list
#         Baseline models (e.g., ['laplace', 'gp'])
#     data_names : list
#         Dataset names for evaluation
#     clustering_types : list
#         Clustering methods to use
#     seeds : list
#         Random seeds for experiments
#     path : str
#         Path to save plots
#     """

#     # Storage for radar chart data
#     radar_metrics = {}

#     # Define radar chart metrics
#     metric_names = ['Peak GAP Improv', 'Final GAP Improv', 'Peak Regret Improv', 'Final Regret Improv', 'GAP Peak Improv Rounds', 'Regret Peak Improv Rounds']

#     finetuning = [True, False]  # finetune vs fixed

#     for f_model in f_models:
#         print(f"Processing model: {f_model}")

#         # Initialize storage for this model
#         trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
#         trace_gaps_alldata_ours = {"finetune": [], "fixed": []}
#         trace_regrets_alldata_ours = {"finetune": [], "fixed": []}  # Add regret tracking
#         trace_regrets_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}

#         for fi in finetuning:
#             label_mkr = "finetune" if fi else "fixed"

#             for data_name in data_names:
#                 for clustering_type in clustering_types:
#                     mat_bench = MATBench(
#                         data_name=data_name,
#                         run_subset_only=run_subset_only,
#                         feature_type=f_model,
#                         finetuning=fi,
#                         iupac=iupac,
#                         prompt_type=prompt_type,
#                         randseed=665,
#                         clustering_type=clustering_type,
#                         feature_reduction=feature_reduction,
#                     )

#                     # Process "ours" method
#                     trace_bys = []
#                     trace_gaps = []
#                     trace_regrets = []

#                     for seed in seeds:
#                         trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi)

#                         if trace_by is not None:
#                             trace_gap = deepcopy(trace_by)
#                             trace_regret = deepcopy(trace_by)
#                             trace_gap[0] = 0
#                             # Calculate GAP improvement
#                             for j in range(1, 101):
#                                 y_0 = trace_by[1]
#                                 y_t = trace_by[j]
#                                 trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

#                             # Calculate regret (simple regret: ground_truth - best_found)
#                             for j in range(1, 100):
#                                 y_t = trace_by[j]
#                                 trace_regret[j] = mat_bench.ground_truth_opt - y_t

#                             trace_bys.append(trace_by)
#                             trace_gaps.append(trace_gap)
#                             trace_regrets.append(trace_regret)
#                             trace_gaps_alldata_ours[label_mkr].append(trace_gap)
#                             trace_regrets_alldata_ours[label_mkr].append(trace_regret)

#                     # Process baseline methods
#                     for baseline in base_lines:
#                         if baseline == "gp" and fi:
#                             continue  # Skip GP finetuning as in original code

#                         trace_bys = []
#                         trace_gaps = []
#                         trace_regrets = []

#                         for seed in seeds:
#                             trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline, finetune=fi)

#                             if trace_by is not None:
#                                 trace_gap = deepcopy(trace_by)
#                                 trace_regret = deepcopy(trace_by)
#                                 trace_gap[0] = 0
#                                 # Calculate GAP improvement
#                                 for j in range(1, 101):
#                                     y_0 = trace_by[1]
#                                     y_t = trace_by[j]
#                                     trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

#                                 # Calculate regret
#                                 for j in range(1, 100):
#                                     y_t = trace_by[j]
#                                     trace_regret[j] = mat_bench.ground_truth_opt - y_t

#                                 trace_bys.append(trace_by)
#                                 trace_gaps.append(trace_gap)
#                                 trace_regrets.append(trace_regret)
#                                 trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)
#                                 trace_regrets_alldata_baseline[baseline + "_" + label_mkr].append(trace_regret)

#         # Calculate improvement metrics (finetuned - fixed)
#         print(f"Calculating radar metrics for {f_model}...")

#         # Stack arrays for processing
#         trace_y_gaps_all_ours_finetuned = np.stack(trace_gaps_alldata_ours["finetune"])[1:101]
#         trace_y_gaps_all_ours_fixed = np.stack(trace_gaps_alldata_ours["fixed"])[1:101]
#         trace_y_regrets_all_ours_finetuned = np.stack(trace_regrets_alldata_ours["finetune"])[1:101]
#         trace_y_regrets_all_ours_fixed = np.stack(trace_regrets_alldata_ours["fixed"])[1:101]

#         # Calculate improvements (finetuned - fixed)
#         gap_improvement_ours = trace_y_gaps_all_ours_finetuned - trace_y_gaps_all_ours_fixed
#         regret_improvement_ours = trace_y_regrets_all_ours_fixed - trace_y_regrets_all_ours_finetuned  # Lower regret is better

#         # Calculate radar metrics for "ours" method
#         final_rounds = slice(90, 100)  # rounds 80-99 for final improvement

#         # Find peak improvement rounds and values
#         gap_mean_over_seeds = np.mean(gap_improvement_ours, axis=0)  # Average across seeds
#         regret_mean_over_seeds = np.mean(regret_improvement_ours, axis=0)  # Average across seeds

#         # Find the round where peak improvement occurs (argmax) and the peak value
#         gap_peak_round = np.argmax(gap_mean_over_seeds) + 1  # +1 because index starts at 0 but rounds start at 1
#         regret_peak_round = np.argmax(regret_mean_over_seeds) + 1
#         gap_peak_value = np.max(gap_mean_over_seeds)  # Maximum GAP improvement value
#         regret_peak_value = np.max(regret_mean_over_seeds)  # Maximum regret improvement value

#         # Convert peak rounds to a 0-10 scale (smaller rounds = higher scores for sample efficiency)
#         # Rounds 1-10 = score 10, rounds 90-100 = score 1
#         gap_peak_score = max(1, min(10, 11 - (gap_peak_round / 10)))
#         regret_peak_score = max(1, min(10, 11 - (regret_peak_round / 10)))

#         radar_metrics[f"LLMAT-{f_model}"] = {
#             'Peak GAP Improv': gap_peak_value,  # Maximum GAP improvement achieved
#             'Final GAP Improv': np.mean(gap_improvement_ours[:, final_rounds]),  # Final rounds average
#             'Peak Regret Improv': regret_peak_value,  # Maximum regret improvement achieved
#             'Final Regret Improv': np.mean(regret_improvement_ours[:, final_rounds]),  # Final rounds average
#             'GAP Peak Improv Rounds': gap_peak_score,  # Higher score = earlier peak (better sample efficiency)
#             'Regret Peak Improv Rounds': regret_peak_score  # Higher score = earlier peak (better sample efficiency)
#         }

#         # Process baselines
#         for baseline in base_lines:
#             if baseline == "gp":
#                 continue  # Skip GP as it doesn't have finetuning

#             trace_y_gaps_baseline_finetuned = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_finetune"])
#             trace_y_gaps_baseline_fixed = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_fixed"])
#             trace_y_regrets_baseline_finetuned = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_finetune"])
#             trace_y_regrets_baseline_fixed = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_fixed"])

#             gap_improvement_baseline = trace_y_gaps_baseline_finetuned - trace_y_gaps_baseline_fixed
#             regret_improvement_baseline = trace_y_regrets_baseline_fixed - trace_y_regrets_baseline_finetuned

#             # Find peak improvement rounds and values for baseline
#             gap_mean_baseline = np.mean(gap_improvement_baseline, axis=0)
#             regret_mean_baseline = np.mean(regret_improvement_baseline, axis=0)

#             gap_peak_round_baseline = np.argmax(gap_mean_baseline) + 1
#             regret_peak_round_baseline = np.argmax(regret_mean_baseline) + 1
#             gap_peak_value_baseline = np.max(gap_mean_baseline)
#             regret_peak_value_baseline = np.max(regret_mean_baseline)

#             gap_peak_score_baseline = max(1, min(10, 11 - (gap_peak_round_baseline / 10)))
#             regret_peak_score_baseline = max(1, min(10, 11 - (regret_peak_round_baseline / 10)))

#             radar_metrics[f"{baseline.upper()}-{f_model}"] = {
#                 'Peak GAP Improv': gap_peak_value_baseline,  # Maximum GAP improvement achieved
#                 'Final GAP Improv': np.mean(gap_improvement_baseline[:, final_rounds]),  # Final rounds average
#                 'Peak Regret Improv': regret_peak_value_baseline,  # Maximum regret improvement achieved
#                 'Final Regret Improv': np.mean(regret_improvement_baseline[:, final_rounds]),  # Final rounds average
#                 'GAP Peak Improv Rounds': gap_peak_score_baseline,
#                 'Regret Peak Improv Rounds': regret_peak_score_baseline
#             }

#     # Create radar plot
#     print("Creating radar plot...")
#     plot_academic_radar_chart(radar_metrics, metric_names, path, save_plots)

#     return radar_metrics

# def normalize_radar_values(data_dict, metric_names):
#     """Normalize values for radar chart (0.2-1.0 scale) to ensure worst case visibility"""
#     normalized_data = {}

#     # Set minimum visibility threshold
#     MIN_RADAR_VALUE = 0.2  # Ensures worst case is still visible
#     MAX_RADAR_VALUE = 1.0
#     RADAR_RANGE = MAX_RADAR_VALUE - MIN_RADAR_VALUE

#     # Find min/max for each metric across all models
#     metric_ranges = {}
#     for metric in metric_names:
#         values = [data_dict[model][metric] for model in data_dict.keys()]
#         metric_ranges[metric] = {'min': min(values), 'max': max(values), 'range': max(values) - min(values)}

#     # Normalize each model's values to 0.2-1.0 scale
#     for model in data_dict.keys():
#         normalized_data[model] = {}
#         for metric in metric_names:
#             val = data_dict[model][metric]
#             min_val = metric_ranges[metric]['min']
#             max_val = metric_ranges[metric]['max']
#             val_range = metric_ranges[metric]['range']

#             # Handle different normalization strategies
#             if 'Peak Rounds' in metric:
#                 # Peak rounds: already on 1-10 scale, normalize to 0.2-1.0
#                 original_normalized = (val - 1) / 9
#                 normalized_val = MIN_RADAR_VALUE + original_normalized * RADAR_RANGE
#             else:
#                 # GAP and Regret improvements: handle negative values properly
#                 if val_range > 0:
#                     original_normalized = (val - min_val) / val_range
#                     normalized_val = MIN_RADAR_VALUE + original_normalized * RADAR_RANGE
#                 else:
#                     normalized_val = (MIN_RADAR_VALUE + MAX_RADAR_VALUE) / 2  # Middle value when all are the same

#                 # For mixed positive/negative values, adjust scaling
#                 if min_val < 0 and max_val > 0:
#                     zero_point = (MIN_RADAR_VALUE + MAX_RADAR_VALUE) / 2
#                     if val >= 0:
#                         if max_val > 0:
#                             normalized_val = zero_point + (val / max_val) * (MAX_RADAR_VALUE - zero_point)
#                         else:
#                             normalized_val = zero_point
#                     else:
#                         if min_val < 0:
#                             normalized_val = zero_point - (abs(val) / abs(min_val)) * (zero_point - MIN_RADAR_VALUE)
#                         else:
#                             normalized_val = zero_point

#             # Ensure bounds [0.2, 1.0]
#             normalized_val = max(MIN_RADAR_VALUE, min(MAX_RADAR_VALUE, normalized_val))
#             normalized_data[model][metric] = normalized_val

#     return normalized_data

# def plot_academic_radar_chart(data_dict, metric_names, save_path, save_plots=True):
#     """Create academic-style radar chart matching the reference image"""

#     # Set academic style parameters
#     plt.rcParams.update({
#         'font.size': 11,
#         'font.family': 'sans-serif',
#         'axes.linewidth': 0.8,
#         'axes.edgecolor': '#333333',
#         'xtick.color': '#333333',
#         'ytick.color': '#333333',
#         'text.color': '#333333'
#     })

#     # Normalize data for visualization
#     normalized_data = normalize_radar_values(data_dict, metric_names)
#     models = list(normalized_data.keys())

#     # Create figure with lighter background
#     fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('#F0F0F0')  # Lighter gray background

#     # Define angles for each metric (in radians)
#     angles = [n / float(len(metric_names)) * 2 * pi for n in range(len(metric_names))]
#     angles += angles[:1]  # Complete the circle

#     # High contrast color scheme for better line differentiation
#     colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C']
#     line_styles = ['-', '--', '-.', ':', '-', '--']
#     markers = ['o', 's', '^', 'v', 'D', 'p']

#     # Set up radial grid with academic styling - adjusted for 0.2-1.0 scale
#     ax.set_ylim(0, 1)

#     # Custom radial ticks starting from 0.2 to show worst case polygon
#     radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
#     ax.set_yticks(radial_ticks)
#     ax.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], fontsize=10, color='#555555', alpha=0.8)

#     # Make the radar circles more obvious with darker, thicker lines
#     ax.grid(True, alpha=0.5, linewidth=1.2, color='#999999')
#     ax.spines['polar'].set_visible(False)

#     # Plot each model with academic styling
#     for i, model in enumerate(models):
#         values = [normalized_data[model][metric] for metric in metric_names]
#         values += values[:1]  # Complete the circle

#         color = colors[i % len(colors)]
#         linestyle = line_styles[i % len(line_styles)]
#         marker = markers[i % len(markers)]

#         # Plot line with high contrast styling
#         ax.plot(angles, values, linestyle=linestyle, linewidth=3.0, color=color, marker=marker, markersize=8, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5,
#                 alpha=0.95, label=model)

#         # Fill area with slightly higher alpha for better visibility
#         ax.fill(angles, values, alpha=0.15, color=color)

#     # Set angular labels with clean formatting
#     ax.set_xticks(angles[:-1])

#     # Shorten labels to match academic style
#     short_labels = ['Peak GAP\nImprov', 'Final GAP\nImprov', 'Peak Regret\nImprov', 'Final Regret\nImprov', 'GAP Peak\nRounds', 'Regret Peak\nRounds']

#     ax.set_xticklabels(short_labels, fontsize=11, color='#333333', ha='center', fontweight='bold')

#     # Clean, academic title with more space for legend
#     plt.title('Finetuning Effectiveness Analysis', fontsize=16, fontweight='bold', color='#333333', pad=40)

#     # Horizontal legend at the bottom
#     legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=11, frameon=True, fancybox=False, shadow=True, framealpha=0.95, facecolor='white',
#                        edgecolor='#888888', borderpad=1.0, ncol=2)

#     # Style legend frame
#     legend.get_frame().set_linewidth(1.0)
#     for text in legend.get_texts():
#         text.set_color('#333333')
#         text.set_fontweight('bold')

#     # Add more prominent radial reference circles
#     for tick in [0.2, 0.4, 0.6, 0.8]:
#         circle = plt.Circle((0, 0), tick, transform=ax.transData._b, fill=False, color='#AAAAAA', alpha=0.8, linewidth=1.2)
#         ax.add_patch(circle)

#     # Add a note about the scale adjustment with adjusted position
#     fig.text(0.02, 0.05, 'Note: Scale adjusted (0.2-1.0) for polygon visibility', fontsize=9, color='#666666', style='italic')

#     plt.tight_layout()

#     # Save with academic quality
#     if save_plots:
#         plt.savefig(f"{save_path}finetuning_radar_analysis_academic.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
#         plt.savefig(f"{save_path}finetuning_radar_analysis_academic.png", format='png', bbox_inches='tight', dpi=300, facecolor='white')

#     plt.show()

#     # Enhanced summary output
#     print_academic_summary(data_dict, normalized_data, metric_names, models)

# def print_academic_summary(data_dict, normalized_data, metric_names, models):
#     """Print academic-style summary of results"""

#     print("\n" + "=" * 70)
#     print("FINETUNING EFFECTIVENESS ANALYSIS - ACADEMIC SUMMARY")
#     print("=" * 70)

#     # Performance ranking
#     print("\nModel Performance Ranking (by average normalized score):")
#     print("-" * 55)

#     avg_scores = {}
#     for model in models:
#         avg_score = np.mean([normalized_data[model][metric] for metric in metric_names])
#         avg_scores[model] = avg_score

#     ranked_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

#     for rank, (model, score) in enumerate(ranked_models, 1):
#         print(f"{rank:2d}. {model:<20} Average Score: {score:.3f}")

#     # Key findings
#     print(f"\nKey Findings:")
#     print("-" * 20)

#     # Best performer per metric
#     for metric in metric_names:
#         best_model = max(models, key=lambda m: data_dict[m][metric])
#         best_value = data_dict[best_model][metric]
#         print(f"• {metric:<25}: {best_model} ({best_value:.4f})")

#     # Identify beneficial vs harmful finetuning
#     beneficial_models = []
#     harmful_models = []

#     for model in models:
#         gap_metrics = ['Peak GAP Improv', 'Final GAP Improv']
#         regret_metrics = ['Peak Regret Improv', 'Final Regret Improv']

#         gap_beneficial = any(data_dict[model][m] > 0 for m in gap_metrics)
#         regret_beneficial = any(data_dict[model][m] > 0 for m in regret_metrics)

#         if gap_beneficial or regret_beneficial:
#             beneficial_models.append(model)
#         else:
#             harmful_models.append(model)

#     print(f"\nFinetuning Impact Assessment:")
#     print("-" * 35)
#     print(f"✓ Models benefiting from finetuning: {len(beneficial_models)}")
#     for model in beneficial_models:
#         print(f"   • {model}")

#     if harmful_models:
#         print(f"✗ Models harmed by finetuning: {len(harmful_models)}")
#         for model in harmful_models:
#             print(f"   • {model}")

#     print(f"\nStatistical Summary:")
#     print("-" * 25)
#     for metric in metric_names:
#         values = [data_dict[model][metric] for model in models]
#         print(f"{metric:<25}: μ={np.mean(values):6.4f}, σ={np.std(values):6.4f}")


def create_radar_plot(path, save_plots=True):
    """
    Create radar plots showing finetuning effectiveness across different metrics
    
    Parameters:
    -----------
    f_models : list
        Feature models (e.g., ['molformer', 't5-base-chem'])
    base_lines : list  
        Baseline models (e.g., ['laplace', 'gp'])
    data_names : list
        Dataset names for evaluation
    clustering_types : list
        Clustering methods to use
    seeds : list
        Random seeds for experiments
    path : str
        Path to save plots
    """

    # Storage for radar chart data
    radar_metrics = {}

    # Define radar chart metrics
    metric_names = ['Peak GAP Improv', 'Final GAP Improv', 'Peak Regret Improv', 'Final Regret Improv', 'GAP Peak Improv Rounds', 'Regret Peak Improv Rounds']

    finetuning = [True, False]  # finetune vs fixed

    for f_model in f_models:
        print(f"Processing model: {f_model}")

        # Initialize storage for this model
        trace_gaps_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}
        trace_gaps_alldata_ours = {"finetune": [], "fixed": []}
        trace_regrets_alldata_ours = {"finetune": [], "fixed": []}  # Add regret tracking
        trace_regrets_alldata_baseline = {"laplace_finetune": [], "laplace_fixed": [], "gp_finetune": [], "gp_fixed": []}

        for fi in finetuning:
            label_mkr = "finetune" if fi else "fixed"

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

                    # Process "ours" method
                    trace_bys = []
                    trace_gaps = []
                    trace_regrets = []

                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi)

                        if trace_by is not None:
                            trace_gap = deepcopy(trace_by)
                            trace_regret = deepcopy(trace_by)
                            trace_gap[0] = 0
                            # Calculate GAP improvement
                            for j in range(1, 101):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

                            # Calculate regret (simple regret: ground_truth - best_found)
                            for j in range(1, 100):
                                y_t = trace_by[j]
                                trace_regret[j] = mat_bench.ground_truth_opt - y_t

                            trace_bys.append(trace_by)
                            trace_gaps.append(trace_gap)
                            trace_regrets.append(trace_regret)
                            trace_gaps_alldata_ours[label_mkr].append(trace_gap)
                            trace_regrets_alldata_ours[label_mkr].append(trace_regret)

                    # Process baseline methods
                    for baseline in base_lines:
                        if baseline == "gp" and fi:
                            continue  # Skip GP finetuning as in original code

                        trace_bys = []
                        trace_gaps = []
                        trace_regrets = []

                        for seed in seeds:
                            trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline, finetune=fi)

                            if trace_by is not None:
                                trace_gap = deepcopy(trace_by)
                                trace_regret = deepcopy(trace_by)
                                trace_gap[0] = 0
                                # Calculate GAP improvement
                                for j in range(1, 101):
                                    y_0 = trace_by[1]
                                    y_t = trace_by[j]
                                    trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)

                                # Calculate regret
                                for j in range(1, 100):
                                    y_t = trace_by[j]
                                    trace_regret[j] = mat_bench.ground_truth_opt - y_t

                                trace_bys.append(trace_by)
                                trace_gaps.append(trace_gap)
                                trace_regrets.append(trace_regret)
                                trace_gaps_alldata_baseline[baseline + "_" + label_mkr].append(trace_gap)
                                trace_regrets_alldata_baseline[baseline + "_" + label_mkr].append(trace_regret)

        # Calculate improvement metrics (finetuned - fixed)
        print(f"Calculating radar metrics for {f_model}...")

        # Stack arrays for processing
        trace_y_gaps_all_ours_finetuned = np.stack(trace_gaps_alldata_ours["finetune"])[1:101]
        trace_y_gaps_all_ours_fixed = np.stack(trace_gaps_alldata_ours["fixed"])[1:101]
        trace_y_regrets_all_ours_finetuned = np.stack(trace_regrets_alldata_ours["finetune"])[1:101]
        trace_y_regrets_all_ours_fixed = np.stack(trace_regrets_alldata_ours["fixed"])[1:101]

        # Calculate improvements (finetuned - fixed)
        gap_improvement_ours = trace_y_gaps_all_ours_finetuned - trace_y_gaps_all_ours_fixed
        regret_improvement_ours = trace_y_regrets_all_ours_fixed - trace_y_regrets_all_ours_finetuned  # Lower regret is better

        # Calculate radar metrics for "ours" method
        final_rounds = slice(90, 100)  # rounds 80-99 for final improvement

        # Find peak improvement rounds and values
        gap_mean_over_seeds = np.mean(gap_improvement_ours, axis=0)  # Average across seeds
        regret_mean_over_seeds = np.mean(regret_improvement_ours, axis=0)  # Average across seeds

        # Find the round where peak improvement occurs (argmax) and the peak value
        gap_peak_round = np.argmax(gap_mean_over_seeds) + 1  # +1 because index starts at 0 but rounds start at 1
        regret_peak_round = np.argmax(regret_mean_over_seeds) + 1
        gap_peak_value = np.max(gap_mean_over_seeds)  # Maximum GAP improvement value
        regret_peak_value = np.max(regret_mean_over_seeds)  # Maximum regret improvement value

        # Convert peak rounds to a 0-10 scale (smaller rounds = higher scores for sample efficiency)
        # Rounds 1-10 = score 10, rounds 90-100 = score 1
        gap_peak_score = max(1, min(10, 11 - (gap_peak_round / 10)))
        regret_peak_score = max(1, min(10, 11 - (regret_peak_round / 10)))

        radar_metrics[f"LLMAT-{f_model}"] = {
            'Peak GAP Improv': gap_peak_value,  # Maximum GAP improvement achieved
            'Final GAP Improv': np.mean(gap_improvement_ours[:, final_rounds]),  # Final rounds average
            'Peak Regret Improv': regret_peak_value,  # Maximum regret improvement achieved
            'Final Regret Improv': np.mean(regret_improvement_ours[:, final_rounds]),  # Final rounds average
            'GAP Peak Improv Rounds': gap_peak_score,  # Higher score = earlier peak (better sample efficiency)
            'Regret Peak Improv Rounds': regret_peak_score  # Higher score = earlier peak (better sample efficiency)
        }

        # Process baselines
        for baseline in base_lines:
            if baseline == "gp":
                continue  # Skip GP as it doesn't have finetuning

            trace_y_gaps_baseline_finetuned = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_finetune"])
            trace_y_gaps_baseline_fixed = np.stack(trace_gaps_alldata_baseline[f"{baseline.upper()}_fixed"])
            trace_y_regrets_baseline_finetuned = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_finetune"])
            trace_y_regrets_baseline_fixed = np.stack(trace_regrets_alldata_baseline[f"{baseline.upper()}_fixed"])

            gap_improvement_baseline = trace_y_gaps_baseline_finetuned - trace_y_gaps_baseline_fixed
            regret_improvement_baseline = trace_y_regrets_baseline_fixed - trace_y_regrets_baseline_finetuned

            # Find peak improvement rounds and values for baseline
            gap_mean_baseline = np.mean(gap_improvement_baseline, axis=0)
            regret_mean_baseline = np.mean(regret_improvement_baseline, axis=0)

            gap_peak_round_baseline = np.argmax(gap_mean_baseline) + 1
            regret_peak_round_baseline = np.argmax(regret_mean_baseline) + 1
            gap_peak_value_baseline = np.max(gap_mean_baseline)
            regret_peak_value_baseline = np.max(regret_mean_baseline)

            gap_peak_score_baseline = max(1, min(10, 11 - (gap_peak_round_baseline / 10)))
            regret_peak_score_baseline = max(1, min(10, 11 - (regret_peak_round_baseline / 10)))

            radar_metrics[f"{baseline.upper()}-{f_model}"] = {
                'Peak GAP Improv': gap_peak_value_baseline,  # Maximum GAP improvement achieved
                'Final GAP Improv': np.mean(gap_improvement_baseline[:, final_rounds]),  # Final rounds average
                'Peak Regret Improv': regret_peak_value_baseline,  # Maximum regret improvement achieved
                'Final Regret Improv': np.mean(regret_improvement_baseline[:, final_rounds]),  # Final rounds average
                'GAP Peak Improv Rounds': gap_peak_score_baseline,
                'Regret Peak Improv Rounds': regret_peak_score_baseline
            }

    # Create radar plot
    print("Creating radar plot...")
    plot_academic_radar_chart(radar_metrics, metric_names, path, save_plots)

    return radar_metrics


def normalize_radar_values(data_dict, metric_names):
    """Normalize values for radar chart (0.2-1.0 scale) to ensure worst case visibility"""
    normalized_data = {}

    # Set minimum visibility threshold
    MIN_RADAR_VALUE = 0.2  # Ensures worst case is still visible
    MAX_RADAR_VALUE = 1.0
    RADAR_RANGE = MAX_RADAR_VALUE - MIN_RADAR_VALUE

    # Find min/max for each metric across all models
    metric_ranges = {}
    for metric in metric_names:
        values = [data_dict[model][metric] for model in data_dict.keys()]
        metric_ranges[metric] = {'min': min(values), 'max': max(values), 'range': max(values) - min(values)}

    # Normalize each model's values to 0.2-1.0 scale
    for model in data_dict.keys():
        normalized_data[model] = {}
        for metric in metric_names:
            val = data_dict[model][metric]
            min_val = metric_ranges[metric]['min']
            max_val = metric_ranges[metric]['max']
            val_range = metric_ranges[metric]['range']

            # Handle different normalization strategies
            if 'Peak Rounds' in metric:
                # Peak rounds: already on 1-10 scale, normalize to 0.2-1.0
                original_normalized = (val - 1) / 9
                normalized_val = MIN_RADAR_VALUE + original_normalized * RADAR_RANGE
            else:
                # GAP and Regret improvements: handle negative values properly
                if val_range > 0:
                    original_normalized = (val - min_val) / val_range
                    normalized_val = MIN_RADAR_VALUE + original_normalized * RADAR_RANGE
                else:
                    normalized_val = (MIN_RADAR_VALUE + MAX_RADAR_VALUE) / 2  # Middle value when all are the same

                # For mixed positive/negative values, adjust scaling
                if min_val < 0 and max_val > 0:
                    zero_point = (MIN_RADAR_VALUE + MAX_RADAR_VALUE) / 2
                    if val >= 0:
                        if max_val > 0:
                            normalized_val = zero_point + (val / max_val) * (MAX_RADAR_VALUE - zero_point)
                        else:
                            normalized_val = zero_point
                    else:
                        if min_val < 0:
                            normalized_val = zero_point - (abs(val) / abs(min_val)) * (zero_point - MIN_RADAR_VALUE)
                        else:
                            normalized_val = zero_point

            # Ensure bounds [0.2, 1.0]
            normalized_val = max(MIN_RADAR_VALUE, min(MAX_RADAR_VALUE, normalized_val))
            normalized_data[model][metric] = normalized_val

    return normalized_data


def plot_academic_radar_chart(data_dict, metric_names, save_path, save_plots=True):
    """Create academic-style radar chart matching the reference image"""

    # Set academic style parameters
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333'
    })

    # Normalize data for visualization
    normalized_data = normalize_radar_values(data_dict, metric_names)
    models = list(normalized_data.keys())

    # Create figure with lighter background
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F0F0F0')  # Lighter gray background

    # Define angles for each metric (in radians)
    angles = [n / float(len(metric_names)) * 2 * pi for n in range(len(metric_names))]
    angles += angles[:1]  # Complete the circle

    # High contrast color scheme for better line differentiation
    colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'v', 'D', 'p']

    # Set up radial grid with academic styling - adjusted for 0.2-1.0 scale
    ax.set_ylim(0, 1)

    # Custom radial ticks starting from 0.2 to show worst case polygon
    radial_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in radial_ticks], fontsize=10, color='#555555', alpha=0.8)

    # Make the radar circles more obvious with darker, thicker lines
    ax.grid(True, alpha=0.5, linewidth=1.2, color='#999999')
    ax.spines['polar'].set_visible(False)

    # Plot each model with academic styling
    for i, model in enumerate(models):
        values = [normalized_data[model][metric] for metric in metric_names]
        values += values[:1]  # Complete the circle

        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        # Plot line with high contrast styling
        ax.plot(angles, values, linestyle=linestyle, linewidth=3.0, color=color, marker=marker, markersize=8, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5,
                alpha=0.95, label=model)

        # Fill area with slightly higher alpha for better visibility
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set angular labels with clean formatting
    ax.set_xticks(angles[:-1])

    # Shorten labels to match academic style
    short_labels = ['Peak GAP\nImprov', 'Final GAP\nImprov', 'Peak Regret\nImprov', 'Final Regret\nImprov', 'GAP Peak\nRounds', 'Regret Peak\nRounds']

    ax.set_xticklabels(short_labels, fontsize=14, color='#333333', ha='center', fontweight='bold')

    # Clean, academic title with space for top legend
    # plt.title('Finetuning Effectiveness Analysis', fontsize=16, fontweight='bold', color='#333333', pad=20)

    # # Horizontal legend at the top with reduced margin
    # legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), fontsize=11, frameon=True, fancybox=False, shadow=True, framealpha=0.95, facecolor='white',
    #                    edgecolor='#888888', borderpad=1.0, ncol=2)

    # # Style legend frame
    # legend.get_frame().set_linewidth(1.0)
    # for text in legend.get_texts():
    #     text.set_color('#333333')
    #     text.set_fontweight('bold')
    # Horizontal legend at the top with reduced margin and no box
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=14, frameon=False, borderpad=1.0, ncol=2)

    # Style legend text
    for text in legend.get_texts():
        text.set_color('#333333')
        text.set_fontweight('bold')

    # Add more prominent radial reference circles
    for tick in [0.2, 0.4, 0.6, 0.8]:
        circle = plt.Circle((0, 0), tick, transform=ax.transData._b, fill=False, color='#AAAAAA', alpha=0.8, linewidth=1.2)
        ax.add_patch(circle)

    # Add a note about the scale adjustment
    # fig.text(0.02, 0.02, 'Note: Scale adjusted (0.2-1.0) for polygon visibility', fontsize=9, color='#666666', style='italic')

    plt.tight_layout()

    # Save with academic quality
    if save_plots:
        plt.savefig(f"{save_path}finetuning_radar_analysis_academic.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        plt.savefig(f"{save_path}finetuning_radar_analysis_academic.png", format='png', bbox_inches='tight', dpi=300, facecolor='white')

    plt.show()

    # Enhanced summary output
    print_academic_summary(data_dict, normalized_data, metric_names, models)


def print_academic_summary(data_dict, normalized_data, metric_names, models):
    """Print academic-style summary of results"""

    print("\n" + "=" * 70)
    print("FINETUNING EFFECTIVENESS ANALYSIS - ACADEMIC SUMMARY")
    print("=" * 70)

    # Performance ranking
    print("\nModel Performance Ranking (by average normalized score):")
    print("-" * 55)

    avg_scores = {}
    for model in models:
        avg_score = np.mean([normalized_data[model][metric] for metric in metric_names])
        avg_scores[model] = avg_score

    ranked_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (model, score) in enumerate(ranked_models, 1):
        print(f"{rank:2d}. {model:<20} Average Score: {score:.3f}")

    # Key findings
    print(f"\nKey Findings:")
    print("-" * 20)

    # Best performer per metric
    for metric in metric_names:
        best_model = max(models, key=lambda m: data_dict[m][metric])
        best_value = data_dict[best_model][metric]
        print(f"• {metric:<25}: {best_model} ({best_value:.4f})")

    # Identify beneficial vs harmful finetuning
    beneficial_models = []
    harmful_models = []

    for model in models:
        gap_metrics = ['Peak GAP Improv', 'Final GAP Improv']
        regret_metrics = ['Peak Regret Improv', 'Final Regret Improv']

        gap_beneficial = any(data_dict[model][m] > 0 for m in gap_metrics)
        regret_beneficial = any(data_dict[model][m] > 0 for m in regret_metrics)

        if gap_beneficial or regret_beneficial:
            beneficial_models.append(model)
        else:
            harmful_models.append(model)

    print(f"\nFinetuning Impact Assessment:")
    print("-" * 35)
    print(f"✓ Models benefiting from finetuning: {len(beneficial_models)}")
    for model in beneficial_models:
        print(f"   • {model}")

    if harmful_models:
        print(f"✗ Models harmed by finetuning: {len(harmful_models)}")
        for model in harmful_models:
            print(f"   • {model}")

    print(f"\nStatistical Summary:")
    print("-" * 25)
    for metric in metric_names:
        values = [data_dict[model][metric] for model in models]
        print(f"{metric:<25}: μ={np.mean(values):6.4f}, σ={np.std(values):6.4f}")


if __name__ == "__main__":
    # plot_regret_diff(path="./results/figs/temp/")
    # plot_gap_diff(path="./results/figs/temp/")
    # plot_radar(path="./results/figs/temp/")
    create_radar_plot(path="./results/figs/0907/", save_plots=True)
