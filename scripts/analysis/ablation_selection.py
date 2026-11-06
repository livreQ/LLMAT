#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Ablation study: UCT vs Greedy node selection in MCTS.
Compares BO performance (GAP, regret, timing) between
UCT (exploration + exploitation) and greedy (exploitation only).
"""
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
f_models = ["molformer"]
prompt_type = "just-smiles"
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]

SELECTION_TYPES = ["uct", "greedy"]
TREE_DEPTH = 3


def get_suffix(seed, n_init_data=10, acqf="ei", gamma=0.5, eta=0.005,
               tree_depth=3, finetune=False, selection_type="uct"):
    if finetune:
        lora_lr = "3e-5"
        lora_reg = 0
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
    else:
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    if selection_type == "greedy":
        suffix += "_greedy"
    return suffix


def load_result(mat_bench, f_model, r_type, seed, tree_depth=3,
                finetune=False, selection_type="uct"):
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/llmat"
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"{path}/{clustering}"
    data_name = mat_bench.dataset_name.split("/")[0]

    suffix = get_suffix(seed, n_init_data=10, acqf="ei", gamma=0.5,
                        eta=0.005, tree_depth=tree_depth, finetune=finetune,
                        selection_type=selection_type)

    file_map = {
        "best_y": f"trace_best_y_{suffix}.npy",
        "trace_time": f"trace_timing_{suffix}.npy",
        "train_time": f"timing_train_{suffix}.npy",
        "pred_time": f"timing_preds_{suffix}.npy",
        "trace_acqvals": f"trace_acqvals_{suffix}.npy",
        "trace_y_his": f"trace_y_his_{suffix}.npy",
    }

    fpath = os.path.join(path, file_map.get(r_type, ""))
    if not os.path.exists(fpath):
        return None
    return np.load(fpath, allow_pickle=True)


def compute_gap_curve(trace_by, mat_bench):
    gap = np.zeros_like(trace_by, dtype=float)
    for j in range(1, min(100, len(trace_by))):
        y_0 = trace_by[1]
        y_t = trace_by[j]
        gap[j] = np.nan_to_num(
            (y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0
        )
    return gap


def compute_regret_curve(trace_yh, mat_bench):
    regret = deepcopy(trace_yh)
    for j in range(0, min(101, len(trace_yh))):
        if mat_bench.maximization:
            regret[j] = mat_bench.ground_truth_opt - np.mean(trace_yh[1:j + 2])
        else:
            regret[j] = np.mean(trace_yh[1:j + 2]) - mat_bench.ground_truth_opt
    return regret


def collect_all_curves(selection_type, tree_depth=TREE_DEPTH, finetune=False):
    all_gap = []
    all_regret = []
    all_times = []

    for f_model in f_models:
        for data_name in data_names:
            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetune,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                for seed in seeds:
                    trace_by = load_result(
                        mat_bench, f_model, "best_y", seed,
                        tree_depth=tree_depth, finetune=finetune,
                        selection_type=selection_type,
                    )
                    if trace_by is not None:
                        all_gap.append(compute_gap_curve(trace_by, mat_bench))

                    trace_yh = load_result(
                        mat_bench, f_model, "trace_y_his", seed,
                        tree_depth=tree_depth, finetune=finetune,
                        selection_type=selection_type,
                    )
                    if trace_yh is not None:
                        all_regret.append(compute_regret_curve(trace_yh, mat_bench))

                    trace_t = load_result(
                        mat_bench, f_model, "trace_time", seed,
                        tree_depth=tree_depth, finetune=finetune,
                        selection_type=selection_type,
                    )
                    if trace_t is not None:
                        all_times.append(np.sum(trace_t))

    return all_gap, all_regret, all_times


def collect_per_dataset(selection_type, tree_depth=TREE_DEPTH, finetune=False):
    results = {}
    for data_name in data_names:
        gaps, regrets, times = [], [], []
        for f_model in f_models:
            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetune,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )
                for seed in seeds:
                    trace_by = load_result(
                        mat_bench, f_model, "best_y", seed,
                        tree_depth=tree_depth, finetune=finetune,
                        selection_type=selection_type,
                    )
                    if trace_by is not None:
                        gaps.append(compute_gap_curve(trace_by, mat_bench))

                    trace_yh = load_result(
                        mat_bench, f_model, "trace_y_his", seed,
                        tree_depth=tree_depth, finetune=finetune,
                        selection_type=selection_type,
                    )
                    if trace_yh is not None:
                        regrets.append(compute_regret_curve(trace_yh, mat_bench))

                    trace_t = load_result(
                        mat_bench, f_model, "trace_time", seed,
                        tree_depth=tree_depth, finetune=finetune,
                        selection_type=selection_type,
                    )
                    if trace_t is not None:
                        times.append(np.sum(trace_t))

        results[data_name] = {"gap": gaps, "regret": regrets, "time": times}
    return results


def print_summary_table(tree_depth=TREE_DEPTH, finetune=False):
    print("\n" + "=" * 120)
    print(f"Selection Type Ablation: UCT vs Greedy  (tree_depth={tree_depth-2}, finetune={finetune})")
    print("=" * 120)

    header = (f"{'Selection':<10} | {'n':>4} | {'GAP AUC (↑)':>16} | {'GAP@T=25':>16} | "
              f"{'GAP@T=50':>16} | {'GAP@T=100':>16} | {'Regret@100 (↓)':>18} | {'Time(s)':>14}")
    print(header)
    print("-" * 120)

    for sel in SELECTION_TYPES:
        all_gap, all_regret, all_times = collect_all_curves(
            sel, tree_depth=tree_depth, finetune=finetune
        )

        n = len(all_gap)
        if n > 0:
            gap_arr = np.stack(all_gap)
            gap_auc = np.mean(np.trapz(gap_arr[:, 1:100], axis=1))
            gap_auc_sem = st.sem(np.trapz(gap_arr[:, 1:100], axis=1))
            gap_t25 = np.mean(gap_arr[:, 25])
            gap_t25_sem = st.sem(gap_arr[:, 25])
            gap_t50 = np.mean(gap_arr[:, 50])
            gap_t50_sem = st.sem(gap_arr[:, 50])
            gap_t100 = np.mean(gap_arr[:, 99])
            gap_t100_sem = st.sem(gap_arr[:, 99])
        else:
            gap_auc = gap_auc_sem = gap_t25 = gap_t25_sem = 0
            gap_t50 = gap_t50_sem = gap_t100 = gap_t100_sem = 0

        n_reg = len(all_regret)
        if n_reg > 0:
            reg_arr = np.stack(all_regret)
            reg_final = np.mean(reg_arr[:, 100])
            reg_final_sem = st.sem(reg_arr[:, 100])
        else:
            reg_final = reg_final_sem = 0

        n_time = len(all_times)
        time_mean = np.mean(all_times) if n_time else 0
        time_sem = st.sem(all_times) if n_time > 1 else 0

        auc_str = f"{gap_auc:.2f} ± {gap_auc_sem:.2f}"
        g25_str = f"{gap_t25:.4f} ± {gap_t25_sem:.4f}"
        g50_str = f"{gap_t50:.4f} ± {gap_t50_sem:.4f}"
        g100_str = f"{gap_t100:.4f} ± {gap_t100_sem:.4f}"
        reg_str = f"{reg_final:.4f} ± {reg_final_sem:.4f}"
        time_str = f"{time_mean:.1f} ± {time_sem:.1f}"
        print(f"{sel:<10} | {n:>4} | {auc_str:>16} | {g25_str:>16} | "
              f"{g50_str:>16} | {g100_str:>16} | {reg_str:>18} | {time_str:>14}")

    print("=" * 120)


def print_per_dataset_table(tree_depth=TREE_DEPTH, finetune=False):
    print("\n" + "=" * 100)
    print(f"Per-Dataset Breakdown: UCT vs Greedy  (tree_depth={tree_depth-2}, finetune={finetune})")
    print("=" * 100)

    header = f"{'Dataset':<12} | {'Selection':<10} | {'n':>4} | {'GAP AUC':>10} | {'GAP@100':>10} | {'Regret@100':>12} | {'Time(s)':>10}"
    print(header)
    print("-" * 100)

    for sel in SELECTION_TYPES:
        per_ds = collect_per_dataset(sel, tree_depth=tree_depth, finetune=finetune)
        for data_name in data_names:
            d = per_ds[data_name]
            n = len(d["gap"])
            if n > 0:
                gap_arr = np.stack(d["gap"])
                gap_auc = np.mean(np.trapz(gap_arr[:, 1:100], axis=1))
                gap_t100 = np.mean(gap_arr[:, 99])
            else:
                gap_auc = gap_t100 = float("nan")

            n_reg = len(d["regret"])
            if n_reg > 0:
                reg_arr = np.stack(d["regret"])
                reg_final = np.mean(reg_arr[:, 100])
            else:
                reg_final = float("nan")

            n_time = len(d["time"])
            time_mean = np.mean(d["time"]) if n_time else float("nan")

            print(f"{data_name:<12} | {sel:<10} | {n:>4} | {gap_auc:>10.2f} | "
                  f"{gap_t100:>10.4f} | {reg_final:>12.4f} | {time_mean:>10.1f}")
        print("-" * 100)


def plot_gap_comparison(path, tree_depth=TREE_DEPTH, finetune=False):
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5})

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"uct": "#2196F3", "greedy": "#FF5722"}
    labels = {"uct": "UCT (explore+exploit)", "greedy": "Greedy (exploit only)"}

    for sel in SELECTION_TYPES:
        all_gap, _, _ = collect_all_curves(sel, tree_depth=tree_depth, finetune=finetune)
        if not all_gap:
            print(f"  [SKIP] No data for {sel}")
            continue

        gap_arr = np.stack(all_gap)
        mean = np.mean(gap_arr, axis=0)[1:100]
        sem = st.sem(gap_arr, axis=0)[1:100]
        T = np.arange(len(mean)) + 1

        ax.plot(T, mean, color=colors[sel], linewidth=2.0, label=labels[sel])
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.12, color=colors[sel])

    ax.set_xlim(1, 99)
    ax.set_ylim(0, 1)
    ax.set_xlabel("BO Rounds: T", fontsize=14, fontweight="bold")
    ax.set_ylabel("GAP", fontsize=14, fontweight="bold")
    ax.set_title("UCT vs Greedy Selection — GAP ($\\uparrow$)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, framealpha=0.9, fancybox=True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "gap_uct_vs_greedy.pdf"), format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_regret_comparison(path, tree_depth=TREE_DEPTH, finetune=False):
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5})

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"uct": "#2196F3", "greedy": "#FF5722"}
    labels = {"uct": "UCT (explore+exploit)", "greedy": "Greedy (exploit only)"}

    for sel in SELECTION_TYPES:
        _, all_regret, _ = collect_all_curves(sel, tree_depth=tree_depth, finetune=finetune)
        if not all_regret:
            print(f"  [SKIP] No data for {sel}")
            continue

        reg_arr = np.stack(all_regret)
        for i in range(len(reg_arr)):
            init = reg_arr[i, 0]
            if init > 1e-10:
                reg_arr[i] = reg_arr[i] / init

        mean = np.mean(reg_arr, axis=0)[1:101]
        sem = st.sem(reg_arr, axis=0)[1:101]
        T = np.arange(len(mean)) + 1

        ax.plot(T, mean, color=colors[sel], linewidth=2.0, label=labels[sel])
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.12, color=colors[sel])

    ax.set_yscale("log")
    ax.set_xlim(1, 100)
    ax.set_xlabel("BO Rounds: T", fontsize=14, fontweight="bold")
    ax.set_ylabel("Relative Regret", fontsize=14, fontweight="bold")
    ax.set_title("UCT vs Greedy Selection — Regret ($\\downarrow$)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, framealpha=0.9, fancybox=True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "regret_uct_vs_greedy.pdf"), format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_per_dataset_gap(path, tree_depth=TREE_DEPTH, finetune=False):
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5})

    n_ds = len(data_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = {"uct": "#2196F3", "greedy": "#FF5722"}
    labels = {"uct": "UCT", "greedy": "Greedy"}

    for idx, data_name in enumerate(data_names):
        ax = axes[idx]
        for sel in SELECTION_TYPES:
            per_ds = collect_per_dataset(sel, tree_depth=tree_depth, finetune=finetune)
            d = per_ds[data_name]
            if not d["gap"]:
                continue

            gap_arr = np.stack(d["gap"])
            mean = np.mean(gap_arr, axis=0)[1:100]
            sem = st.sem(gap_arr, axis=0)[1:100]
            T = np.arange(len(mean)) + 1

            ax.plot(T, mean, color=colors[sel], linewidth=1.5, label=labels[sel])
            ax.fill_between(T, mean - sem, mean + sem, alpha=0.1, color=colors[sel])

        ax.set_title(data_name, fontsize=12, fontweight="bold")
        ax.set_xlim(1, 99)
        ax.set_ylim(0, 1)
        if idx >= 3:
            ax.set_xlabel("BO Rounds", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel("GAP", fontsize=11)
        ax.legend(fontsize=9, frameon=True)
        sns.despine(ax=ax, top=True, right=True)

    plt.suptitle("UCT vs Greedy — Per-Dataset GAP ($\\uparrow$)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "gap_uct_vs_greedy_per_dataset.pdf"), format="pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    tree_depth = TREE_DEPTH

    print_summary_table(tree_depth=tree_depth, finetune=False)
    print_per_dataset_table(tree_depth=tree_depth, finetune=False)

    fig_path = "./results/figs/selection_ablation/"
    plot_gap_comparison(fig_path, tree_depth=tree_depth, finetune=False)
    plot_regret_comparison(fig_path, tree_depth=tree_depth, finetune=False)
    plot_per_dataset_gap(fig_path, tree_depth=tree_depth, finetune=False)
