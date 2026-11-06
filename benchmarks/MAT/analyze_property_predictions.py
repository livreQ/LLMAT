#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Analyze direct property prediction performance of GPT, Gemini, and Qwen3.
Compares LLM-predicted numerical values against ground truth,
and also derives clusters from predictions to compare with the
clustering approach.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATASETS = {
    "redox-mer": {
        "csv": os.path.join(DATA_DIR, "redox_mer.csv"),
        "smiles_col": "SMILES",
        "property_col": "Ered",
    },
}

LLM_PROPERTY_DIR = os.path.join(DATA_DIR, "llm_property_api")
LLM_CLUSTER_DIRS = {
    "GPT":    os.path.join(DATA_DIR, "llm_clustering"),
    "Gemini": os.path.join(DATA_DIR, "llm_clustering_api", "gemini"),
    "Qwen3":  os.path.join(DATA_DIR, "llm_clustering_api", "qwen3"),
}


def load_ground_truth(dataset_name):
    """Load ground truth and compute quantile bin edges for 5 clusters."""
    cfg = DATASETS[dataset_name]
    df = pd.read_csv(cfg["csv"])
    prop = cfg["property_col"]
    # Compute the 5 equal-frequency bin edges from full dataset
    _, bin_edges = pd.qcut(df[prop], q=5, retbins=True, labels=False)
    df["gt_cluster"] = pd.qcut(df[prop], q=5, labels=[0, 1, 2, 3, 4]).astype(int)
    smiles_to_info = {}
    for _, row in df.iterrows():
        smiles_to_info[row[cfg["smiles_col"]]] = {
            "property": row[prop],
            "gt_cluster": row["gt_cluster"],
        }
    return df, smiles_to_info, bin_edges


def load_llm_property_predictions(api_type, dataset_name):
    """Load direct property predictions from an LLM."""
    folder = os.path.join(LLM_PROPERTY_DIR, api_type, dataset_name)
    if not os.path.exists(folder):
        return None

    all_results = {}
    batch_idx = 0
    while True:
        npy_file = os.path.join(folder, f"{dataset_name}_{batch_idx}_result.npy")
        if not os.path.exists(npy_file):
            break
        data = np.load(npy_file, allow_pickle=True)
        for smi, val in data:
            try:
                all_results[smi] = float(val)
            except (ValueError, TypeError):
                all_results[smi] = np.nan
        batch_idx += 1
    return all_results if all_results else None


def load_llm_clusters(llm_name, dataset_name):
    """Load cluster assignments from an LLM."""
    folder = os.path.join(LLM_CLUSTER_DIRS[llm_name], dataset_name)
    if not os.path.exists(folder):
        return None
    all_results = {}
    batch_idx = 0
    while True:
        npy_file = os.path.join(folder, f"{dataset_name}_{batch_idx}_result.npy")
        if not os.path.exists(npy_file):
            break
        data = np.load(npy_file, allow_pickle=True)
        for smi, label in data:
            label_str = str(label).strip()
            if "-" in label_str:
                label_str = label_str.split("-")[1]
            all_results[smi] = int(label_str)
        batch_idx += 1
    return all_results if all_results else None


def hungarian_accuracy(gt_labels, pred_labels, n_clusters=5):
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(n_clusters)))
    row_ind, col_ind = linear_sum_assignment(-cm)
    best_acc = cm[row_ind, col_ind].sum() / len(gt_labels)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return best_acc, mapping


def predictions_to_clusters(predictions, bin_edges):
    """
    Convert numerical predictions to cluster labels using the same
    quantile bin edges as ground truth.
    """
    clusters = np.digitize(predictions, bin_edges[1:-1])  # bins 0..4
    clusters = np.clip(clusters, 0, 4)
    return clusters


def analyze_regression_metrics(gt_vals, pred_vals, property_col):
    """Compute regression metrics for direct property prediction."""
    mae = mean_absolute_error(gt_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(gt_vals, pred_vals))
    r2 = r2_score(gt_vals, pred_vals)
    pearson_r, pearson_p = pearsonr(gt_vals, pred_vals)
    spearman_r, spearman_p = spearmanr(gt_vals, pred_vals)
    kendall_r, kendall_p = kendalltau(gt_vals, pred_vals)

    # Relative error
    mape = np.mean(np.abs((gt_vals - pred_vals) / gt_vals)) * 100

    print(f"    MAE:                {mae:.4f}")
    print(f"    RMSE:               {rmse:.4f}")
    print(f"    MAPE:               {mape:.2f}%")
    print(f"    R²:                 {r2:.4f}")
    print(f"    Pearson r:          {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"    Spearman rho:       {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"    Kendall tau:        {kendall_r:.4f} (p={kendall_p:.2e})")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "pearson": pearson_r,
        "spearman": spearman_r,
        "kendall": kendall_r,
    }


def analyze_derived_clusters(gt_clusters, pred_clusters, label):
    """Compute clustering metrics for clusters derived from predictions."""
    ari = adjusted_rand_score(gt_clusters, pred_clusters)
    nmi = normalized_mutual_info_score(gt_clusters, pred_clusters)
    direct_acc = accuracy_score(gt_clusters, pred_clusters)
    hungarian_acc, best_mapping = hungarian_accuracy(gt_clusters, pred_clusters)

    print(f"    ARI:                {ari:.4f}")
    print(f"    NMI:                {nmi:.4f}")
    print(f"    Direct Accuracy:    {direct_acc:.4f}")
    print(f"    Hungarian Accuracy: {hungarian_acc:.4f}")
    print(f"    Best mapping:       {best_mapping}")

    # Confusion matrix
    cm = confusion_matrix(gt_clusters, pred_clusters, labels=list(range(5)))
    print(f"    Confusion Matrix (rows=GT, cols=Pred):")
    header = "          " + "".join(f"  P{c}" for c in range(5))
    print(header)
    for r in range(5):
        row_str = f"      GT{r} " + "".join(f"{cm[r, c]:4d}" for c in range(5))
        print(row_str)

    return {
        "ari": ari,
        "nmi": nmi,
        "direct_acc": direct_acc,
        "hungarian_acc": hungarian_acc,
    }


def analyze_rank_correctness(gt_vals, pred_vals):
    """Check pairwise rank consistency between predictions and ground truth."""
    n = len(gt_vals)
    n_concordant = 0
    n_discordant = 0
    n_tied = 0
    total = 0
    # Use vectorized approach for speed
    for i in range(n):
        gt_diff = gt_vals[i+1:] - gt_vals[i]
        pred_diff = pred_vals[i+1:] - pred_vals[i]
        products = gt_diff * pred_diff
        n_concordant += np.sum(products > 0)
        n_discordant += np.sum(products < 0)
        n_tied += np.sum(products == 0)
        total += len(gt_diff)

    pairwise_acc = (n_concordant + n_tied) / total if total > 0 else 0
    print(f"    Concordant pairs:  {n_concordant:8d} ({100*n_concordant/total:.1f}%)")
    print(f"    Discordant pairs:  {n_discordant:8d} ({100*n_discordant/total:.1f}%)")
    print(f"    Tied pairs:        {n_tied:8d} ({100*n_tied/total:.1f}%)")
    print(f"    Pairwise accuracy: {pairwise_acc:.4f}")
    return pairwise_acc


def analyze_dataset(dataset_name):
    """Run full analysis for a dataset."""
    cfg = DATASETS[dataset_name]
    prop_col = cfg["property_col"]

    print("=" * 80)
    print(f"DATASET: {dataset_name}  |  Property: {prop_col}")
    print("=" * 80)

    df, smiles_to_info, bin_edges = load_ground_truth(dataset_name)
    print(f"Ground truth bin edges: {bin_edges}")
    print(f"Total molecules: {len(df)}\n")

    all_regression = {}
    all_cluster_from_pred = {}
    all_cluster_direct = {}

    for api_type in ["gpt", "gemini", "qwen3"]:
        print(f"\n{'━' * 70}")
        print(f"  LLM: {api_type.upper()} — Direct Property Prediction")
        print(f"{'━' * 70}")

        preds = load_llm_property_predictions(api_type, dataset_name)
        if preds is None:
            print(f"  [SKIP] No property prediction data for {api_type}/{dataset_name}")
            print(f"  Run: python benchmarks/MAT/prompt_property_api.py --api {api_type} --dataset {dataset_name}")
            continue

        # Match with ground truth
        common = [s for s in preds if s in smiles_to_info and not np.isnan(preds[s])]
        print(f"  Molecules: {len(preds)} total, {len(common)} valid & matched\n")

        if len(common) < 10:
            print("  [SKIP] Too few valid predictions")
            continue

        gt_vals = np.array([smiles_to_info[s]["property"] for s in common])
        pred_vals = np.array([preds[s] for s in common])
        gt_clusters = np.array([smiles_to_info[s]["gt_cluster"] for s in common])

        # --- 1. Regression Metrics ---
        print(f"  [1] Regression Metrics ({prop_col}):")
        reg_metrics = analyze_regression_metrics(gt_vals, pred_vals, prop_col)
        all_regression[api_type] = reg_metrics

        # --- 2. Prediction Distribution ---
        print(f"\n  [2] Prediction Distribution:")
        print(f"    GT   range: [{gt_vals.min():.4f}, {gt_vals.max():.4f}], "
              f"mean={gt_vals.mean():.4f}, std={gt_vals.std():.4f}")
        print(f"    Pred range: [{pred_vals.min():.4f}, {pred_vals.max():.4f}], "
              f"mean={pred_vals.mean():.4f}, std={pred_vals.std():.4f}")

        # --- 3. Rank Correctness ---
        print(f"\n  [3] Pairwise Rank Correctness:")
        pairwise_acc = analyze_rank_correctness(gt_vals, pred_vals)
        all_regression[api_type]["pairwise_acc"] = pairwise_acc

        # --- 4. Derived Clusters (from numerical predictions) ---
        pred_clusters = predictions_to_clusters(pred_vals, bin_edges)
        print(f"\n  [4] Clusters Derived from Predicted Values (using GT bin edges):")
        print(f"    Derived cluster distribution:")
        for c in range(5):
            cnt = np.sum(pred_clusters == c)
            print(f"      Cluster {c}: {cnt:4d} ({100*cnt/len(pred_clusters):.1f}%)")
        clust_metrics = analyze_derived_clusters(gt_clusters, pred_clusters,
                                                  f"{api_type}-derived")
        all_cluster_from_pred[api_type] = clust_metrics

        # --- 5. Error Analysis by GT Cluster ---
        print(f"\n  [5] Error Analysis by GT Cluster:")
        print(f"    {'Cluster':>8} {'N':>5} {'MAE':>8} {'MeanGT':>8} {'MeanPred':>8} {'Bias':>8}")
        for c in range(5):
            mask = gt_clusters == c
            if mask.sum() == 0:
                continue
            c_gt = gt_vals[mask]
            c_pred = pred_vals[mask]
            c_mae = np.mean(np.abs(c_gt - c_pred))
            c_bias = np.mean(c_pred - c_gt)
            print(f"    {c:>8d} {mask.sum():>5d} {c_mae:>8.4f} "
                  f"{c_gt.mean():>8.4f} {c_pred.mean():>8.4f} {c_bias:>+8.4f}")

    # --- Compare with direct clustering approach ---
    print(f"\n\n{'=' * 80}")
    print(f"COMPARISON: Direct Clustering vs Derived Clusters from Property Predictions")
    print(f"{'=' * 80}")

    for api_type in ["gpt", "gemini", "qwen3"]:
        llm_key = {"gpt": "GPT", "gemini": "Gemini", "qwen3": "Qwen3"}[api_type]
        direct_clusters = load_llm_clusters(llm_key, dataset_name)
        if direct_clusters is None:
            continue

        preds = load_llm_property_predictions(api_type, dataset_name)
        if preds is None:
            continue

        common = [s for s in preds
                  if s in smiles_to_info
                  and s in direct_clusters
                  and not np.isnan(preds[s])]

        if len(common) < 10:
            continue

        gt_clusters = np.array([smiles_to_info[s]["gt_cluster"] for s in common])
        direct_labels = np.array([direct_clusters[s] for s in common])
        pred_vals = np.array([preds[s] for s in common])
        derived_labels = predictions_to_clusters(pred_vals, bin_edges)

        # Metrics for direct clustering
        d_ari = adjusted_rand_score(gt_clusters, direct_labels)
        d_nmi = normalized_mutual_info_score(gt_clusters, direct_labels)
        d_hacc, _ = hungarian_accuracy(gt_clusters, direct_labels)

        # Spearman for direct clusters (label vs property)
        gt_vals = np.array([smiles_to_info[s]["property"] for s in common])
        d_sp, _ = spearmanr(direct_labels, gt_vals)

        # Metrics for derived clusters
        p_ari = adjusted_rand_score(gt_clusters, derived_labels)
        p_nmi = normalized_mutual_info_score(gt_clusters, derived_labels)
        p_hacc, _ = hungarian_accuracy(gt_clusters, derived_labels)
        p_sp, _ = spearmanr(derived_labels, gt_vals)

        print(f"\n  {llm_key} (n={len(common)}):")
        print(f"    {'Method':<25} {'ARI':>7} {'NMI':>7} {'HunAcc':>7} {'Spearman':>9}")
        print(f"    {'-'*55}")
        print(f"    {'Direct Clustering':<25} {d_ari:>7.4f} {d_nmi:>7.4f} "
              f"{d_hacc:>7.4f} {d_sp:>9.4f}")
        print(f"    {'Derived from Prediction':<25} {p_ari:>7.4f} {p_nmi:>7.4f} "
              f"{p_hacc:>7.4f} {p_sp:>9.4f}")

        all_cluster_direct[api_type] = {
            "ari": d_ari, "nmi": d_nmi, "hungarian_acc": d_hacc, "spearman": d_sp,
        }

    # --- Final Summary Table ---
    if all_regression:
        print(f"\n\n{'=' * 80}")
        print("REGRESSION SUMMARY")
        print(f"{'=' * 80}")
        header = (f"{'LLM':<8} {'MAE':>7} {'RMSE':>7} {'MAPE%':>7} {'R²':>7} "
                  f"{'Pearson':>8} {'Spear':>7} {'Kend':>7} {'PairAcc':>8}")
        print(header)
        print("-" * len(header))
        for api, m in all_regression.items():
            print(f"{api:<8} {m['mae']:>7.4f} {m['rmse']:>7.4f} {m['mape']:>7.2f} "
                  f"{m['r2']:>7.4f} {m['pearson']:>8.4f} {m['spearman']:>7.4f} "
                  f"{m['kendall']:>7.4f} {m['pairwise_acc']:>8.4f}")

    if all_cluster_from_pred:
        print(f"\n{'=' * 80}")
        print("DERIVED CLUSTERING SUMMARY (clusters from predicted values)")
        print(f"{'=' * 80}")
        header = f"{'LLM':<8} {'ARI':>7} {'NMI':>7} {'DirAcc':>7} {'HunAcc':>7}"
        print(header)
        print("-" * len(header))
        for api, m in all_cluster_from_pred.items():
            print(f"{api:<8} {m['ari']:>7.4f} {m['nmi']:>7.4f} "
                  f"{m['direct_acc']:>7.4f} {m['hungarian_acc']:>7.4f}")


def main():
    for dataset_name in DATASETS:
        analyze_dataset(dataset_name)


if __name__ == "__main__":
    main()
