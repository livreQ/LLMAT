#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Visualization showing that LLMs cannot predict absolute property values correctly
but can predict relative ranks correctly.

Generates:
  1. Value scatter: GT vs Predicted (shows poor absolute prediction)
  2. Rank scatter: GT rank vs Predicted rank (shows good rank preservation)
  3. Summary bar chart: regression metrics (bad) vs rank metrics (good)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "MAT")
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASETS = {
    "redox-mer": {
        "csv": os.path.join(DATA_DIR, "redox_mer.csv"),
        "smiles_col": "SMILES",
        "property_col": "Ered",
        "unit": "V",
    },
}

LLM_PROPERTY_DIR = os.path.join(DATA_DIR, "llm_property_api")

API_LABELS = {
    "gemini": "Gemini",
    "qwen3": "Qwen 3",
}


def load_ground_truth(dataset_name):
    cfg = DATASETS[dataset_name]
    df = pd.read_csv(cfg["csv"])
    smiles_to_val = {}
    for _, row in df.iterrows():
        smiles_to_val[row[cfg["smiles_col"]]] = row[cfg["property_col"]]
    return df, smiles_to_val


def load_predictions(api_type, dataset_name):
    folder = os.path.join(LLM_PROPERTY_DIR, api_type, dataset_name)
    if not os.path.isdir(folder):
        return None
    preds = {}
    batch_idx = 0
    while True:
        npy_file = os.path.join(folder, f"{dataset_name}_{batch_idx}_result.npy")
        if not os.path.exists(npy_file):
            break
        data = np.load(npy_file, allow_pickle=True)
        for smi, val in data:
            try:
                preds[smi] = float(val)
            except (ValueError, TypeError):
                pass
        batch_idx += 1
    return preds if preds else None


def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5


def plot_value_vs_rank(dataset_name="redox-mer"):
    set_plot_style()

    cfg = DATASETS[dataset_name]
    prop_col = cfg["property_col"]
    unit = cfg["unit"]
    df, smiles_to_val = load_ground_truth(dataset_name)

    api_types = [a for a in ["gemini", "qwen3"] if load_predictions(a, dataset_name)]
    if not api_types:
        print("No LLM property predictions found.")
        return

    n_apis = len(api_types)

    # --- Figure 1: Value scatter (top) + Rank scatter (bottom) ---
    fig, axes = plt.subplots(2, n_apis, figsize=(6 * n_apis, 10))
    if n_apis == 1:
        axes = axes.reshape(2, 1)

    for col, api_type in enumerate(api_types):
        preds = load_predictions(api_type, dataset_name)
        common = sorted([s for s in preds if s in smiles_to_val])
        gt_vals = np.array([smiles_to_val[s] for s in common])
        pred_vals = np.array([preds[s] for s in common])

        # Metrics
        pearson_r, _ = pearsonr(gt_vals, pred_vals)
        spearman_r, _ = spearmanr(gt_vals, pred_vals)
        kendall_r, _ = kendalltau(gt_vals, pred_vals)
        mae = mean_absolute_error(gt_vals, pred_vals)
        r2 = r2_score(gt_vals, pred_vals)

        # Ranks (1 = smallest)
        gt_ranks = np.argsort(np.argsort(gt_vals)) + 1
        pred_ranks = np.argsort(np.argsort(pred_vals)) + 1

        # --- Top row: Value scatter ---
        ax_val = axes[0, col]
        ax_val.scatter(gt_vals, pred_vals, alpha=0.35, s=15, c='#3498db', edgecolors='none')

        # Identity line spanning the union of both ranges
        all_min = min(gt_vals.min(), pred_vals.min())
        all_max = max(gt_vals.max(), pred_vals.max())
        margin = (all_max - all_min) * 0.05
        ax_val.plot([all_min - margin, all_max + margin],
                    [all_min - margin, all_max + margin],
                    'r--', linewidth=1.5, label='y = x')

        ax_val.set_xlabel(f"Ground Truth {prop_col} ({unit})")
        ax_val.set_ylabel(f"Predicted {prop_col} ({unit})")
        ax_val.set_title(f"{API_LABELS[api_type]} — Absolute Values")

        textstr = (f"Pearson r = {pearson_r:.3f}\n"
                   f"R$^2$ = {r2:.3f}\n"
                   f"MAE = {mae:.3f}")
        ax_val.text(0.05, 0.95, textstr, transform=ax_val.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffcccc', alpha=0.85))
        ax_val.legend(loc='lower right', fontsize=11)

        # --- Bottom row: Rank scatter ---
        ax_rank = axes[1, col]
        ax_rank.scatter(gt_ranks, pred_ranks, alpha=0.35, s=15, c='#2ecc71', edgecolors='none')

        # Identity line
        ax_rank.plot([1, len(common)], [1, len(common)],
                     'r--', linewidth=1.5, label='Perfect rank')

        ax_rank.set_xlabel(f"Ground Truth Rank")
        ax_rank.set_ylabel(f"Predicted Rank")
        ax_rank.set_title(f"{API_LABELS[api_type]} — Ranks")

        textstr = (f"Spearman $\\rho$ = {spearman_r:.3f}\n"
                   f"Kendall $\\tau$ = {kendall_r:.3f}")
        ax_rank.text(0.05, 0.95, textstr, transform=ax_rank.transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#ccffcc', alpha=0.85))
        ax_rank.legend(loc='lower right', fontsize=11)

    fig.suptitle(f"LLM Property Prediction: Absolute Values vs Ranks — {dataset_name}",
                 fontsize=17, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"llm_value_vs_rank_{dataset_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()

    # --- Figure 2: Summary bar chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    api_names = [API_LABELS[a] for a in api_types]
    x = np.arange(len(api_names))
    width = 0.35

    # Collect metrics
    all_r2, all_pearson, all_spearman, all_kendall = [], [], [], []
    for api_type in api_types:
        preds = load_predictions(api_type, dataset_name)
        common = sorted([s for s in preds if s in smiles_to_val])
        gt_vals = np.array([smiles_to_val[s] for s in common])
        pred_vals = np.array([preds[s] for s in common])
        pr, _ = pearsonr(gt_vals, pred_vals)
        sr, _ = spearmanr(gt_vals, pred_vals)
        kr, _ = kendalltau(gt_vals, pred_vals)
        r2 = r2_score(gt_vals, pred_vals)
        all_r2.append(r2)
        all_pearson.append(pr)
        all_spearman.append(sr)
        all_kendall.append(kr)

    # Left: Value prediction quality (R², Pearson)
    bars1 = ax1.bar(x - width / 2, all_r2, width, label='R$^2$', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, all_pearson, width, label='Pearson r', color='#e67e22', alpha=0.8)
    ax1.set_ylabel("Score")
    ax1.set_title("Value Prediction Quality", fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(api_names)
    ax1.legend()
    ax1.set_ylim(min(min(all_r2), 0) - 0.1, 1.05)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11)

    # Right: Rank prediction quality (Spearman, Kendall)
    bars3 = ax2.bar(x - width / 2, all_spearman, width, label='Spearman $\\rho$', color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x + width / 2, all_kendall, width, label='Kendall $\\tau$', color='#27ae60', alpha=0.8)
    ax2.set_ylabel("Score")
    ax2.set_title("Rank Prediction Quality", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(api_names)
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11)

    fig.suptitle(f"Value vs Rank Prediction: Metric Comparison — {dataset_name}",
                 fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"llm_value_vs_rank_bars_{dataset_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()

    # --- Figure 3: Normalized value overlay ---
    # Normalize both GT and predictions to [0,1] to show rank alignment despite value mismatch
    fig, axes = plt.subplots(1, n_apis, figsize=(6 * n_apis, 5))
    if n_apis == 1:
        axes = [axes]

    for col, api_type in enumerate(api_types):
        ax = axes[col]
        preds = load_predictions(api_type, dataset_name)
        common = sorted([s for s in preds if s in smiles_to_val])
        gt_vals = np.array([smiles_to_val[s] for s in common])
        pred_vals = np.array([preds[s] for s in common])

        # Sort by GT value
        sort_idx = np.argsort(gt_vals)
        gt_sorted = gt_vals[sort_idx]
        pred_sorted = pred_vals[sort_idx]

        # Normalize to [0, 1]
        gt_norm = (gt_sorted - gt_sorted.min()) / (gt_sorted.max() - gt_sorted.min())
        pred_norm = (pred_sorted - pred_sorted.min()) / (pred_sorted.max() - pred_sorted.min())

        mol_idx = np.arange(len(common))

        ax.plot(mol_idx, gt_norm, color='#3498db', alpha=0.7, linewidth=1.5, label='GT (normalized)')
        ax.plot(mol_idx, pred_norm, color='#e74c3c', alpha=0.7, linewidth=1.5, label='Pred (normalized)')
        ax.fill_between(mol_idx, gt_norm, pred_norm, alpha=0.1, color='gray')

        spearman_r, _ = spearmanr(gt_vals, pred_vals)
        ax.set_xlabel("Molecules (sorted by GT value)")
        ax.set_ylabel("Normalized Value [0, 1]")
        ax.set_title(f"{API_LABELS[api_type]} (Spearman $\\rho$ = {spearman_r:.3f})", fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f"Normalized Value Overlay (sorted by GT) — {dataset_name}\n"
                 f"Absolute values differ, but trends align",
                 fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fname = f"llm_normalized_overlay_{dataset_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


if __name__ == "__main__":
    for dataset_name in DATASETS:
        plot_value_vs_rank(dataset_name)
