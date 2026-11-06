#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Analyze LLM clustering performance for Gemini, Qwen3, and GPT.
Compare against ground truth clusters derived from evenly splitting
molecules into 5 bins based on their property values.
Also verifies correctness of rank ordering (cluster 0=lowest, 4=highest).
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    confusion_matrix,
)
from scipy.stats import spearmanr, kendalltau


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# LLM result directories
LLM_DIRS = {
    "GPT":    os.path.join(DATA_DIR, "llm_clustering"),
    "Gemini": os.path.join(DATA_DIR, "llm_clustering_api", "gemini"),
    "Qwen3":  os.path.join(DATA_DIR, "llm_clustering_api", "qwen3"),
}

# Dataset config: (csv_path, smiles_col, property_col, subfolder_name)
DATASETS = {
    "redox-mer": {
        "csv": os.path.join(DATA_DIR, "redox_mer.csv"),
        "smiles_col": "SMILES",
        "property_col": "Ered",
    },
}


def load_ground_truth(dataset_name):
    """Load ground truth data and create 5 equal-frequency clusters."""
    cfg = DATASETS[dataset_name]
    df = pd.read_csv(cfg["csv"])
    # Create ground truth clusters: evenly split into 5 bins by property value
    df["gt_cluster"] = pd.qcut(df[cfg["property_col"]], q=5, labels=[0, 1, 2, 3, 4])
    df["gt_cluster"] = df["gt_cluster"].astype(int)
    # Build SMILES -> (property_value, gt_cluster) mapping
    smiles_to_info = {}
    for _, row in df.iterrows():
        smiles_to_info[row[cfg["smiles_col"]]] = {
            "property": row[cfg["property_col"]],
            "gt_cluster": row["gt_cluster"],
        }
    return df, smiles_to_info


def load_llm_clusters(llm_name, dataset_name):
    """Load all batch results for a given LLM and dataset."""
    folder = os.path.join(LLM_DIRS[llm_name], dataset_name)
    if not os.path.exists(folder):
        return None

    all_results = {}
    batch_idx = 0
    while True:
        npy_file = os.path.join(folder, f"{dataset_name}_{batch_idx}_result.npy")
        if not os.path.exists(npy_file):
            break
        data = np.load(npy_file, allow_pickle=True)
        for smiles, label in data:
            label_str = str(label).strip()
            if "-" in label_str:
                label_str = label_str.split("-")[1]
            all_results[smiles] = int(label_str)
        batch_idx += 1

    return all_results


def hungarian_accuracy(gt_labels, pred_labels, n_clusters=5):
    """
    Compute clustering accuracy using the Hungarian algorithm
    to find the best permutation mapping between predicted and ground truth labels.
    """
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(n_clusters)))
    # Hungarian algorithm finds the assignment that maximizes total matches
    row_ind, col_ind = linear_sum_assignment(-cm)
    best_acc = cm[row_ind, col_ind].sum() / len(gt_labels)
    # Build mapping: pred_label -> gt_label
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return best_acc, mapping


def verify_rank_ordering(llm_clusters, smiles_to_info):
    """
    Verify that LLM cluster labels are rank-consistent with property values.
    Cluster 0 should correspond to lowest property values, cluster 4 to highest.

    Returns:
        mean_props: dict mapping cluster_id -> mean property value
        is_monotonic: whether mean property values increase with cluster label
        spearman_corr: Spearman correlation between cluster labels and property values
        kendall_corr: Kendall tau correlation between cluster labels and property values
    """
    cluster_props = {i: [] for i in range(5)}
    label_list = []
    prop_list = []

    for smiles, cluster in llm_clusters.items():
        if smiles in smiles_to_info:
            prop_val = smiles_to_info[smiles]["property"]
            if 0 <= cluster <= 4:
                cluster_props[cluster].append(prop_val)
                label_list.append(cluster)
                prop_list.append(prop_val)

    mean_props = {}
    for c in range(5):
        if cluster_props[c]:
            mean_props[c] = np.mean(cluster_props[c])
        else:
            mean_props[c] = float("nan")

    # Check monotonicity
    valid_means = [mean_props[c] for c in range(5) if not np.isnan(mean_props[c])]
    is_monotonic = all(valid_means[i] <= valid_means[i + 1] for i in range(len(valid_means) - 1))

    # Correlation between assigned cluster label and actual property value
    spearman_corr, spearman_p = spearmanr(label_list, prop_list)
    kendall_corr, kendall_p = kendalltau(label_list, prop_list)

    return mean_props, is_monotonic, (spearman_corr, spearman_p), (kendall_corr, kendall_p)


def analyze_dataset(dataset_name):
    """Run full analysis for a dataset across all LLMs."""
    print("=" * 80)
    print(f"DATASET: {dataset_name}")
    print(f"Property: {DATASETS[dataset_name]['property_col']}")
    print("=" * 80)

    df, smiles_to_info = load_ground_truth(dataset_name)
    cfg = DATASETS[dataset_name]

    # Show ground truth cluster statistics
    print(f"\nGround Truth Clusters (5 equal-frequency bins of {cfg['property_col']}):")
    print("-" * 60)
    for c in range(5):
        mask = df["gt_cluster"] == c
        props = df.loc[mask, cfg["property_col"]]
        print(f"  Cluster {c}: n={len(props):4d}, "
              f"{cfg['property_col']} range=[{props.min():.4f}, {props.max():.4f}], "
              f"mean={props.mean():.4f}")

    results_summary = {}

    for llm_name in ["GPT", "Gemini", "Qwen3"]:
        print(f"\n{'─' * 70}")
        print(f"  LLM: {llm_name}")
        print(f"{'─' * 70}")

        llm_clusters = load_llm_clusters(llm_name, dataset_name)
        if llm_clusters is None:
            print(f"  [SKIP] No data found for {llm_name}/{dataset_name}")
            continue

        # Match LLM clusters to ground truth
        common_smiles = [s for s in llm_clusters if s in smiles_to_info]
        n_total = len(llm_clusters)
        n_matched = len(common_smiles)
        print(f"  Molecules: {n_total} total, {n_matched} matched to ground truth")

        if n_matched == 0:
            print("  [SKIP] No matching molecules found")
            continue

        gt_labels = np.array([smiles_to_info[s]["gt_cluster"] for s in common_smiles])
        pred_labels = np.array([llm_clusters[s] for s in common_smiles])

        # Check label distribution
        print(f"\n  Label distribution (LLM predicted):")
        for c in range(5):
            count = np.sum(pred_labels == c)
            print(f"    Cluster {c}: {count:4d} ({100 * count / len(pred_labels):.1f}%)")

        # --- Clustering Quality Metrics ---
        ari = adjusted_rand_score(gt_labels, pred_labels)
        nmi = normalized_mutual_info_score(gt_labels, pred_labels)
        direct_acc = accuracy_score(gt_labels, pred_labels)
        hungarian_acc, best_mapping = hungarian_accuracy(gt_labels, pred_labels)

        print(f"\n  Clustering Quality Metrics:")
        print(f"    Adjusted Rand Index (ARI):         {ari:.4f}")
        print(f"    Normalized Mutual Info (NMI):       {nmi:.4f}")
        print(f"    Direct Accuracy (label match):      {direct_acc:.4f}")
        print(f"    Hungarian Accuracy (best mapping):  {hungarian_acc:.4f}")
        print(f"    Best label mapping (pred->gt):      {best_mapping}")

        # --- Rank Ordering Verification ---
        mean_props, is_monotonic, (sp_corr, sp_p), (kt_corr, kt_p) = verify_rank_ordering(
            llm_clusters, smiles_to_info
        )

        print(f"\n  Rank Ordering Verification:")
        print(f"    Mean {cfg['property_col']} per LLM cluster:")
        for c in range(5):
            if not np.isnan(mean_props[c]):
                n_in_cluster = np.sum(pred_labels == c)
                print(f"      Cluster {c}: mean={mean_props[c]:.4f}  (n={n_in_cluster})")
            else:
                print(f"      Cluster {c}: [empty]")

        print(f"    Monotonically increasing?  {'YES' if is_monotonic else 'NO'}")
        print(f"    Spearman correlation:       {sp_corr:.4f} (p={sp_p:.2e})")
        print(f"    Kendall tau correlation:    {kt_corr:.4f} (p={kt_p:.2e})")

        # --- Pairwise Rank Accuracy ---
        # For each pair of molecules, check if the relative ordering of cluster
        # labels agrees with the relative ordering of property values
        props = np.array([smiles_to_info[s]["property"] for s in common_smiles])
        n_concordant = 0
        n_discordant = 0
        n_tied = 0
        for i in range(len(common_smiles)):
            for j in range(i + 1, len(common_smiles)):
                prop_diff = props[i] - props[j]
                label_diff = pred_labels[i] - pred_labels[j]
                if prop_diff * label_diff > 0:
                    n_concordant += 1
                elif prop_diff * label_diff < 0:
                    n_discordant += 1
                else:
                    n_tied += 1

        total_pairs = n_concordant + n_discordant + n_tied
        pairwise_acc = (n_concordant + n_tied) / total_pairs if total_pairs > 0 else 0
        print(f"\n  Pairwise Rank Consistency:")
        print(f"    Concordant pairs:  {n_concordant:6d} ({100 * n_concordant / total_pairs:.1f}%)")
        print(f"    Discordant pairs:  {n_discordant:6d} ({100 * n_discordant / total_pairs:.1f}%)")
        print(f"    Tied pairs:        {n_tied:6d} ({100 * n_tied / total_pairs:.1f}%)")
        print(f"    Pairwise accuracy: {pairwise_acc:.4f}")

        # --- Confusion Matrix ---
        cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(5)))
        print(f"\n  Confusion Matrix (rows=GT, cols=Pred):")
        header = "      " + "".join(f"  P{c}" for c in range(5))
        print(header)
        for r in range(5):
            row_str = f"  GT{r} " + "".join(f"{cm[r, c]:4d}" for c in range(5))
            print(row_str)

        results_summary[llm_name] = {
            "ari": ari,
            "nmi": nmi,
            "direct_acc": direct_acc,
            "hungarian_acc": hungarian_acc,
            "spearman": sp_corr,
            "kendall": kt_corr,
            "monotonic": is_monotonic,
            "pairwise_acc": pairwise_acc,
            "n_molecules": n_matched,
        }

    # --- Summary Table ---
    if results_summary:
        print(f"\n{'=' * 80}")
        print(f"SUMMARY TABLE: {dataset_name}")
        print(f"{'=' * 80}")
        header = f"{'LLM':<8} {'ARI':>7} {'NMI':>7} {'DirAcc':>7} {'HunAcc':>7} {'Spear':>7} {'Kend':>7} {'Mono':>6} {'PairAcc':>8} {'N':>6}"
        print(header)
        print("-" * len(header))
        for llm_name, metrics in results_summary.items():
            print(
                f"{llm_name:<8} "
                f"{metrics['ari']:>7.4f} "
                f"{metrics['nmi']:>7.4f} "
                f"{metrics['direct_acc']:>7.4f} "
                f"{metrics['hungarian_acc']:>7.4f} "
                f"{metrics['spearman']:>7.4f} "
                f"{metrics['kendall']:>7.4f} "
                f"{'YES' if metrics['monotonic'] else 'NO':>6} "
                f"{metrics['pairwise_acc']:>8.4f} "
                f"{metrics['n_molecules']:>6}"
            )

    return results_summary


def main():
    all_results = {}
    for dataset_name in DATASETS:
        results = analyze_dataset(dataset_name)
        all_results[dataset_name] = results
    return all_results


if __name__ == "__main__":
    main()
