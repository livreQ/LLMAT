#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Plot per-node training loss curves and partition quality:
meta-learning (eta>0) vs no meta-learning (eta=0).

Metrics:
  - Loss curve per epoch: convergence speed of node classifiers
  - Partition quality: mean(pos targets) - mean(neg targets) after split

Usage:
  python plot_meta_learning_ablation.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

RESULTS_BASE = "results"

DATA_NAMES = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
CLUSTERING = "kmeans"
SEEDS = list(range(15))

ETA_META = 0.005
ETA_NONE = 0
TREE_DEPTH = 3
GAMMA = 0.5
N_INIT = 10
ACQF = "ei"


def find_log_paths(data_name, eta, seed):
    suffix = f"{N_INIT}_{ACQF}_{GAMMA}_{eta}_{TREE_DEPTH}_{seed}"
    pattern = f"{RESULTS_BASE}/{data_name}/*/llmat/{CLUSTERING}/node_train_logs_{suffix}.npy"
    return glob.glob(pattern)


def load_all_logs(eta):
    all_logs = []
    for data_name in DATA_NAMES:
        for seed in SEEDS:
            for fpath in find_log_paths(data_name, eta, seed):
                logs = np.load(fpath, allow_pickle=True)
                for entry in logs:
                    entry["dataset"] = data_name
                    entry["seed"] = seed
                    entry["eta"] = eta
                    all_logs.append(entry)
    return all_logs


DATASET_DISPLAY = {
    "redox-mer": "Redoxmer",
    "solvation": "Solvation",
    "kinase": "Kinase",
    "laser": "Laser",
    "pce": "Photovoltaics",
    "photoswitch": "Photoswitches",
}


def group_by_depth(logs):
    by_depth = defaultdict(list)
    for entry in logs:
        by_depth[entry["depth"]].append(entry)
    return by_depth


def group_by_dataset_depth(logs):
    by_ds_depth = defaultdict(lambda: defaultdict(list))
    for entry in logs:
        by_ds_depth[entry["dataset"]][entry["depth"]].append(entry)
    return by_ds_depth


def plot_loss_by_depth(logs_meta, logs_none, save_path="./results/figs/meta_learning/"):
    os.makedirs(save_path, exist_ok=True)
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5})

    ds_depth_meta = group_by_dataset_depth(logs_meta)
    ds_depth_none = group_by_dataset_depth(logs_none)

    datasets = [d for d in DATA_NAMES
                if d in ds_depth_meta or d in ds_depth_none]
    if not datasets:
        print("[SKIP] No data found.")
        return

    all_depths = set()
    for d in datasets:
        all_depths |= set(ds_depth_meta.get(d, {}).keys())
        all_depths |= set(ds_depth_none.get(d, {}).keys())
    all_depths = sorted(all_depths)
    n_depths = len(all_depths)
    n_datasets = len(datasets)

    colors = {"meta": "#2196F3", "none": "#FF5722"}

    fig, axes = plt.subplots(n_datasets, n_depths,
                             figsize=(4.5 * n_depths, 3.2 * n_datasets),
                             sharey="row", squeeze=False)

    for row, ds in enumerate(datasets):
        meta_depths = ds_depth_meta.get(ds, {})
        none_depths = ds_depth_none.get(ds, {})
        for col, depth in enumerate(all_depths):
            ax = axes[row][col]
            for label, by_depth, color in [
                (f"$\\eta$={ETA_META} (meta)", meta_depths, colors["meta"]),
                (f"$\\eta$=0 (no meta)", none_depths, colors["none"]),
            ]:
                entries = by_depth.get(depth, [])
                if not entries:
                    continue
                loss_curves = np.array([e["epoch_losses"] for e in entries])
                mean_loss = np.mean(loss_curves, axis=0)
                sem_loss = np.std(loss_curves, axis=0) / np.sqrt(len(loss_curves))
                epochs = np.arange(1, loss_curves.shape[1] + 1)
                ax.plot(epochs, mean_loss, color=color, linewidth=2.0, label=label)
                ax.fill_between(epochs, mean_loss - sem_loss, mean_loss + sem_loss,
                                alpha=0.15, color=color)

            if row == 0:
                ax.set_title(f"Depth {depth}", fontsize=11, fontweight="bold")
            if row == n_datasets - 1:
                ax.set_xlabel("Epoch", fontsize=11)
            if col == 0:
                ax.set_ylabel(DATASET_DISPLAY.get(ds, ds), fontsize=11, fontweight="bold")
            if row == 0 and col == n_depths - 1:
                ax.legend(fontsize=8, frameon=True, loc="upper right")
            sns.despine(ax=ax, top=True, right=True)

    plt.suptitle("Per-Node Training Loss: Meta-Learning vs No Meta-Learning",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "meta_learning_loss_by_depth.pdf"),
                format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(save_path, "meta_learning_loss_by_depth.png"),
                format="png", bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Saved to {save_path}meta_learning_loss_by_depth.pdf")


def plot_partition_quality(logs_meta, logs_none, save_path="./results/figs/meta_learning/"):
    """Bar chart: partition quality and final loss, per dataset × depth."""
    os.makedirs(save_path, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    all_logs = logs_meta + logs_none
    if not all_logs:
        print("[SKIP] No data found.")
        return

    rows = []
    for e in all_logs:
        label = f"Meta ($\\eta$>0)" if e["eta"] > 0 else "No Meta ($\\eta$=0)"
        pos_m = e.get("pos_mean", float("nan"))
        neg_m = e.get("neg_mean", float("nan"))
        if not np.isnan(pos_m) and not np.isnan(neg_m):
            rows.append({
                "Dataset": DATASET_DISPLAY.get(e["dataset"], e["dataset"]),
                "Depth": e["depth"],
                "Method": label,
                "Partition Gap": pos_m - neg_m,
                "Final Loss": e["epoch_losses"][-1],
            })

    if not rows:
        print("[SKIP] No partition data.")
        return

    df = pd.DataFrame(rows)
    datasets = [DATASET_DISPLAY[d] for d in DATA_NAMES
                if DATASET_DISPLAY[d] in df["Dataset"].unique()]
    n_ds = len(datasets)

    palette = {"Meta ($\\eta$>0)": "#2196F3", "No Meta ($\\eta$=0)": "#FF5722"}

    fig, axes = plt.subplots(n_ds, 2, figsize=(10, 2.8 * n_ds), squeeze=False)

    for i, ds in enumerate(datasets):
        sub = df[df["Dataset"] == ds]
        sns.barplot(data=sub, x="Depth", y="Partition Gap", hue="Method",
                    ax=axes[i][0], palette=palette, errorbar="se")
        axes[i][0].set_ylabel(ds, fontsize=11, fontweight="bold")
        axes[i][0].get_legend().remove()
        if i == 0:
            axes[i][0].set_title("Partition Quality ($\\uparrow$)", fontweight="bold")

        sns.barplot(data=sub, x="Depth", y="Final Loss", hue="Method",
                    ax=axes[i][1], palette=palette, errorbar="se")
        axes[i][1].set_ylabel("")
        axes[i][1].get_legend().remove()
        if i == 0:
            axes[i][1].set_title("Final Training Loss ($\\downarrow$)", fontweight="bold")

        if i < n_ds - 1:
            axes[i][0].set_xlabel("")
            axes[i][1].set_xlabel("")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_path, "meta_learning_partition_quality.pdf"),
                format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(save_path, "meta_learning_partition_quality.png"),
                format="png", bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Saved to {save_path}meta_learning_partition_quality.pdf")


def print_summary(logs_meta, logs_none):
    ds_depth_meta = group_by_dataset_depth(logs_meta)
    ds_depth_none = group_by_dataset_depth(logs_none)

    datasets = [d for d in DATA_NAMES
                if d in ds_depth_meta or d in ds_depth_none]

    all_depths = set()
    for d in datasets:
        all_depths |= set(ds_depth_meta.get(d, {}).keys())
        all_depths |= set(ds_depth_none.get(d, {}).keys())
    all_depths = sorted(all_depths)

    print("\n" + "=" * 135)
    print("Meta-Learning Ablation: Per-Node Training Metrics (by Dataset)")
    print("=" * 135)
    header = (f"{'Dataset':<14} | {'Depth':>5} | {'Method':<16} | {'N':>5} | {'Avg n':>6} | "
              f"{'Init Loss':>14} | {'Final Loss':>14} | "
              f"{'Partition Gap':>14} | {'n_pos':>6} | {'n_neg':>6}")
    print(header)
    print("-" * 135)

    for ds in datasets:
        ds_label = DATASET_DISPLAY.get(ds, ds)
        meta_depths = ds_depth_meta.get(ds, {})
        none_depths = ds_depth_none.get(ds, {})
        ds_depths = sorted(set(list(meta_depths.keys()) + list(none_depths.keys())))

        for depth in ds_depths:
            for label, by_depth in [("eta=0.005(meta)", meta_depths),
                                    ("eta=0 (no meta)", none_depths)]:
                entries = by_depth.get(depth, [])
                if not entries:
                    continue
                n = len(entries)
                avg_samples = np.mean([e["n_samples"] for e in entries])
                init_losses = [e["epoch_losses"][0] for e in entries]
                final_losses = [e["epoch_losses"][-1] for e in entries]

                gaps = []
                n_pos_list, n_neg_list = [], []
                for e in entries:
                    pm = e.get("pos_mean", float("nan"))
                    nm = e.get("neg_mean", float("nan"))
                    if not np.isnan(pm) and not np.isnan(nm):
                        gaps.append(pm - nm)
                    n_pos_list.append(e.get("n_pos", 0))
                    n_neg_list.append(e.get("n_neg", 0))

                il = f"{np.mean(init_losses):.4f}±{np.std(init_losses)/np.sqrt(n):.4f}"
                fl = f"{np.mean(final_losses):.4f}±{np.std(final_losses)/np.sqrt(n):.4f}"
                gp = f"{np.mean(gaps):.4f}±{np.std(gaps)/np.sqrt(len(gaps)):.4f}" if gaps else "N/A"
                np_ = f"{np.mean(n_pos_list):.1f}"
                nn_ = f"{np.mean(n_neg_list):.1f}"
                print(f"{ds_label:<14} | {depth:>5} | {label:<16} | {n:>5} | {avg_samples:>6.1f} | "
                      f"{il:>14} | {fl:>14} | {gp:>14} | {np_:>6} | {nn_:>6}")
        print("-" * 135)


if __name__ == "__main__":
    print("Loading meta-learning logs (eta=0.005)...")
    logs_meta = load_all_logs(ETA_META)
    print(f"  Found {len(logs_meta)} node entries")

    print("Loading no-meta-learning logs (eta=0.0)...")
    logs_none = load_all_logs(ETA_NONE)
    print(f"  Found {len(logs_none)} node entries")

    if not logs_meta and not logs_none:
        print("\nNo node_train_logs found. Run experiments first:")
        print("  python src/main/run_all.py -a llmat --eta 0.005 --seed 0")
        print("  python src/main/run_all.py -a llmat --eta 0 --seed 0")
        exit(0)

    print_summary(logs_meta, logs_none)

    save_path = "./results/figs/meta_learning/"
    plot_loss_by_depth(logs_meta, logs_none, save_path)
    plot_partition_quality(logs_meta, logs_none, save_path)
