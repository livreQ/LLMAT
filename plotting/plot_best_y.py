#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Aug 19 15:08:46 2025
# File Name: plot_best_y.py
# Description:
"""

from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotting.load_results import load_ours, load_baselines

seeds = list(set(range(15)))
data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
# "solvation",
f_models = ["molformer"]  #,molformer "gpt2-medium", "t5-base", "gpt2-large"]
prompt_type = "just-smiles"
finetuning = [True, False]
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]

our_algo = "llmat"
base_lines = ["laplace", "gp"]

Properties = {
    "Redoxmer": "Redox Potential",
    "Laser": "Fluorescence Oscillator Strength",
    "Solvation": "Solvation Energy",
    "Kinase": "Docking Score",
    "Photovoltaics": "Power Conversion Efficiency",
    "Photoswitches": "$\pi-\pi^*$ Transition Wavelength"
}
Datasets = {
    "redox-mer": "Redoxmer",
    "laser": "Laser",
    "solvation": "Solvation",
    "kinase": "Kinase",
    "pce": "Photovoltaics",
    "photoswitch": "Photoswitches",
}

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


def plot_best_y(path, f_models, data_names):
    """
    Plot the best y values for each dataset and feature model.
    """

    for data_name in data_names:
        # plt.figure(figsize=(6, 3))
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
                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, finetune=fi)
                        if trace_by is not None:
                            trace_bys.append(trace_by)
                    # print(trace_bys)
                    trace_best_y = np.stack(trace_bys)
                    mean = np.mean(trace_best_y, axis=0)[1:101]  # Over randseeds
                    sem = st.sem(trace_best_y, axis=0)[1:101]  # Over randseeds
                    T = np.arange(len(mean)) + 1
                    df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                    if fi:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}", linestyle='--')
                    else:
                        ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}")  #{f_model}-{clustering_type} 'sd' shows standard deviation as shaded region
                    ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)
                    ax1.axhline(mat_bench.ground_truth_opt, c="k", ls="dashed", zorder=1000)

                    for baseline in base_lines:
                        if baseline == "gp" and label_mkr == "finetune":
                            continue
                        trace_bys = []
                        for seed in seeds:
                            trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline, finetune=fi)
                            if trace_by is not None:
                                trace_bys.append(trace_by)
                        trace_best_y = np.stack(trace_bys)
                        mean = np.mean(trace_best_y, axis=0)[1:101]  # Over randseeds
                        sem = st.sem(trace_best_y, axis=0)[1:101]  # Over randseeds
                        T = np.arange(len(mean)) + 1
                        print(len(T), len(mean), len(sem))
                        df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                        if fi:
                            ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}", linestyle='--')
                        else:
                            ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}-{label_mkr}")  #-{f_model}
                        ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

                plt.xlim(1, max(T))
                plt.setp(ax1.get_legend().get_texts(), fontsize='14')
                plt.xlabel('BO Rounds: T', fontsize=14)
                plt.ylabel(Properties[Datasets[data_name]], fontsize=14)
                plt.tight_layout()
                # savefig('xx.pdf', format='pdf', bbox_inches='tight')
                # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
                if mat_bench.maximization:
                    plt.title(f"{Datasets[data_name]}($\\uparrow$)", fontsize=14)
                else:
                    plt.title(f"{Datasets[data_name]}($\downarrow$)", fontsize=14)
                plt.savefig(path + data_name + "_" + f_model + "_finetune.pdf", format='pdf', bbox_inches='tight')
                plt.show()


if __name__ == "__main__":

    f_models = ["molformer"]  #, "gpt2-medium", "t5-base", "gpt2-large"]
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]

    plot_best_y(path="./results/figs/0907/Bestys/", f_models=f_models, data_names=data_names)
