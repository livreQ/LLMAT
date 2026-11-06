import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter

DATA_DIR = "benchmarks/MAT/data"

DATASETS_INFO = {
    "redox-mer": {
        "csv": "redox_mer_with_iupac.csv.gz",
        "target": "Ered",
        "smiles_col": "SMILES",
        "maximization": False,
    },
    "kinase": {
        "csv": "enamine10k.csv.gz",
        "target": "score",
        "smiles_col": "SMILES",
        "maximization": False,
    },
    "laser": {
        "csv": "laser_multi10k.csv.gz",
        "target": "Fluorescence Oscillator Strength",
        "smiles_col": "SMILES",
        "maximization": True,
    },
    "pce": {
        "csv": "photovoltaics_pce10k.csv.gz",
        "target": "pce",
        "smiles_col": "SMILES",
        "maximization": True,
    },
    "photoswitch": {
        "csv": "photoswitches.csv.gz",
        "target": "Pi-Pi* Transition Wavelength",
        "smiles_col": "SMILES",
        "maximization": True,
    },
    "solvation": {
        "csv": "redox_mer_with_iupac.csv.gz",
        "target": "Gsol",
        "smiles_col": "SMILES",
        "maximization": False,
    },
}

API_LABELS = {
    "gpt": "GPT-4o",
    "gemini": "Gemini",
    "qwen3": "Qwen 3",
}


def load_clustering_labels(data_name, api_type):
    """
    Load all batch results for a given dataset and API, return dict: SMILES -> cluster_label.
    """
    if api_type == "gpt":
        folder = os.path.join(DATA_DIR, "llm_clustering", data_name)
    else:
        folder = os.path.join(DATA_DIR, "llm_clustering_api", api_type, data_name)

    if not os.path.isdir(folder):
        return None

    mole_cluster = {}
    batch_idx = 0
    while True:
        result_file = os.path.join(folder, f"{data_name}_{batch_idx}_result.npy")
        if not os.path.exists(result_file):
            break
        results = np.load(result_file)
        for smiles, label in results:
            try:
                clean_label = label.split("-")[-1] if "-" in label else label
                mole_cluster[smiles] = int(clean_label)
            except (ValueError, IndexError):
                mole_cluster[smiles] = -1
        batch_idx += 1

    if not mole_cluster:
        return None
    return mole_cluster


def load_dataset(data_name):
    """Load the raw dataset CSV."""
    info = DATASETS_INFO[data_name]
    df = pd.read_csv(os.path.join(DATA_DIR, info["csv"]))
    return df


def find_common_datasets():
    """Find datasets that have clustering results for at least 2 APIs."""
    available = {}
    for data_name in DATASETS_INFO:
        apis_with_data = []
        for api_type in ["gpt", "gemini", "qwen3"]:
            labels = load_clustering_labels(data_name, api_type)
            if labels is not None and len(labels) > 0:
                apis_with_data.append(api_type)
        if len(apis_with_data) >= 2:
            available[data_name] = apis_with_data
    return available


def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = True


def plot_cluster_distributions(all_labels, data_name):
    """
    Bar chart: number of molecules per cluster for each API.
    """
    apis = list(all_labels.keys())
    n_apis = len(apis)

    fig, axes = plt.subplots(1, n_apis, figsize=(5 * n_apis, 4), sharey=True)
    if n_apis == 1:
        axes = [axes]

    for idx, api_type in enumerate(apis):
        ax = axes[idx]
        labels = list(all_labels[api_type].values())
        counts = Counter(labels)
        clusters = sorted(counts.keys())
        values = [counts[c] for c in clusters]

        ax.bar([str(c) for c in clusters], values, color=sns.color_palette("Set2", len(clusters)))
        ax.set_title(f"{API_LABELS[api_type]}", fontweight='bold')
        ax.set_xlabel("Cluster ID")
        if idx == 0:
            ax.set_ylabel("Count")

    fig.suptitle(f"Cluster Size Distribution — {data_name}", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"api_cluster_distribution_{data_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


def plot_target_by_cluster(all_labels, df, data_name):
    """
    Violin plot: target value distribution per cluster for each API.
    """
    info = DATASETS_INFO[data_name]
    target_col = info["target"]
    smiles_col = info["smiles_col"]
    apis = list(all_labels.keys())
    n_apis = len(apis)

    fig, axes = plt.subplots(1, n_apis, figsize=(6 * n_apis, 4), sharey=True)
    if n_apis == 1:
        axes = [axes]

    for idx, api_type in enumerate(apis):
        ax = axes[idx]
        label_dict = all_labels[api_type]

        # Build a dataframe with target and cluster
        rows = []
        for _, row in df.iterrows():
            smi = row[smiles_col]
            if smi in label_dict:
                rows.append({"target": row[target_col], "cluster": label_dict[smi]})
        plot_df = pd.DataFrame(rows)

        if plot_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, alpha=0.5)
            continue

        sns.violinplot(data=plot_df, x="cluster", y="target", ax=ax,
                       order=sorted(plot_df["cluster"].unique()), cut=0, inner="box")

        # Mark the global optimum target value
        if info["maximization"]:
            opt_val = df[target_col].max()
        else:
            opt_val = df[target_col].min()
        ax.axhline(y=opt_val, color='red', linestyle='--', linewidth=1.5, label=f"Optimum: {opt_val:.2f}")

        ax.set_title(f"{API_LABELS[api_type]}", fontweight='bold')
        ax.set_xlabel("Cluster ID")
        if idx == 0:
            ax.set_ylabel(target_col)
        ax.legend(fontsize=10)

    fig.suptitle(f"Target Distribution by Cluster — {data_name}", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"api_target_by_cluster_{data_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


def plot_inter_api_agreement(all_labels, data_name):
    """
    Heatmap of Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
    between each pair of APIs.
    """
    apis = list(all_labels.keys())
    if len(apis) < 2:
        print(f"Need at least 2 APIs for agreement metrics on {data_name}, skipping.")
        return

    # Find common molecules across all APIs
    common_smiles = set(all_labels[apis[0]].keys())
    for api_type in apis[1:]:
        common_smiles &= set(all_labels[api_type].keys())
    common_smiles = sorted(common_smiles)

    if len(common_smiles) == 0:
        print(f"No common molecules found for {data_name}, skipping agreement plot.")
        return

    n = len(apis)
    ari_matrix = np.ones((n, n))
    nmi_matrix = np.ones((n, n))

    for i in range(n):
        labels_i = [all_labels[apis[i]][s] for s in common_smiles]
        for j in range(i + 1, n):
            labels_j = [all_labels[apis[j]][s] for s in common_smiles]
            ari = adjusted_rand_score(labels_i, labels_j)
            nmi = normalized_mutual_info_score(labels_i, labels_j)
            ari_matrix[i, j] = ari_matrix[j, i] = ari
            nmi_matrix[i, j] = nmi_matrix[j, i] = nmi

    api_names = [API_LABELS[a] for a in apis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.heatmap(ari_matrix, annot=True, fmt=".3f", xticklabels=api_names, yticklabels=api_names,
                cmap="YlOrRd", vmin=0, vmax=1, ax=ax1, square=True)
    ax1.set_title("Adjusted Rand Index (ARI)", fontweight='bold')

    sns.heatmap(nmi_matrix, annot=True, fmt=".3f", xticklabels=api_names, yticklabels=api_names,
                cmap="YlOrRd", vmin=0, vmax=1, ax=ax2, square=True)
    ax2.set_title("Normalized Mutual Information (NMI)", fontweight='bold')

    fig.suptitle(f"Inter-API Clustering Agreement — {data_name}", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"api_agreement_{data_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


def plot_rank_performance(all_labels, df, data_name):
    """
    Bar chart: for each API, rank clusters by mean target and show
    which cluster contains the global optimum molecule.
    """
    info = DATASETS_INFO[data_name]
    target_col = info["target"]
    smiles_col = info["smiles_col"]
    maximization = info["maximization"]
    apis = list(all_labels.keys())
    n_apis = len(apis)

    # Find optimal molecule
    if maximization:
        opt_idx = df[target_col].idxmax()
    else:
        opt_idx = df[target_col].idxmin()
    opt_smiles = df.loc[opt_idx, smiles_col]
    opt_target = df.loc[opt_idx, target_col]

    fig, axes = plt.subplots(1, n_apis, figsize=(6 * n_apis, 4), sharey=True)
    if n_apis == 1:
        axes = [axes]

    for idx, api_type in enumerate(apis):
        ax = axes[idx]
        label_dict = all_labels[api_type]

        # Build cluster -> [targets] mapping
        cluster_targets = {}
        for _, row in df.iterrows():
            smi = row[smiles_col]
            if smi in label_dict:
                c = label_dict[smi]
                cluster_targets.setdefault(c, []).append(row[target_col])

        # Compute mean target per cluster, rank them
        cluster_means = {c: np.mean(vals) for c, vals in cluster_targets.items()}
        sorted_clusters = sorted(cluster_means.keys(),
                                 key=lambda c: cluster_means[c],
                                 reverse=maximization)

        bar_labels = [str(c) for c in sorted_clusters]
        bar_values = [cluster_means[c] for c in sorted_clusters]

        # Color the bar containing the optimum
        opt_cluster = label_dict.get(opt_smiles, None)
        colors = ['#e74c3c' if c == opt_cluster else '#3498db' for c in sorted_clusters]

        ax.bar(bar_labels, bar_values, color=colors)
        ax.set_title(f"{API_LABELS[api_type]}", fontweight='bold')
        ax.set_xlabel("Cluster (ranked)")
        if idx == 0:
            ax.set_ylabel(f"Mean {target_col}")

        # Mark optimum cluster rank
        if opt_cluster in sorted_clusters:
            rank = sorted_clusters.index(opt_cluster) + 1
            ax.text(0.95, 0.95, f"Opt in rank {rank}/{len(sorted_clusters)}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    fig.suptitle(f"Cluster Rank by Mean Target — {data_name}\n(red = cluster containing global optimum)",
                 fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fname = f"api_rank_performance_{data_name}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


def print_summary_table(all_labels, df, data_name):
    """Print a summary table of clustering and rank metrics."""
    info = DATASETS_INFO[data_name]
    target_col = info["target"]
    smiles_col = info["smiles_col"]
    maximization = info["maximization"]

    if maximization:
        opt_idx = df[target_col].idxmax()
    else:
        opt_idx = df[target_col].idxmin()
    opt_smiles = df.loc[opt_idx, smiles_col]
    opt_target = df.loc[opt_idx, target_col]

    print(f"\n{'='*70}")
    print(f"  Summary for {data_name} (optimum {target_col} = {opt_target:.4f})")
    print(f"{'='*70}")
    header = f"{'API':<12} {'#Clusters':<12} {'Opt Cluster':<14} {'Opt Rank':<10} {'Cluster Sizes'}"
    print(header)
    print("-" * len(header))

    for api_type, label_dict in all_labels.items():
        clusters = set(label_dict.values())
        n_clusters = len(clusters)

        # Cluster sizes
        counts = Counter(label_dict.values())
        sizes_str = ", ".join(f"{c}:{counts[c]}" for c in sorted(counts.keys()))

        # Rank of optimal cluster
        cluster_targets = {}
        for _, row in df.iterrows():
            smi = row[smiles_col]
            if smi in label_dict:
                c = label_dict[smi]
                cluster_targets.setdefault(c, []).append(row[target_col])

        cluster_means = {c: np.mean(vals) for c, vals in cluster_targets.items()}
        sorted_clusters = sorted(cluster_means.keys(),
                                 key=lambda c: cluster_means[c],
                                 reverse=maximization)

        opt_cluster = label_dict.get(opt_smiles, "N/A")
        if opt_cluster in sorted_clusters:
            opt_rank = sorted_clusters.index(opt_cluster) + 1
        else:
            opt_rank = "N/A"

        print(f"{API_LABELS[api_type]:<12} {n_clusters:<12} {str(opt_cluster):<14} {str(opt_rank):<10} {sizes_str}")


def main():
    set_plot_style()

    available = find_common_datasets()
    if not available:
        print("No datasets found with clustering results from at least 2 APIs.")
        return

    print(f"Datasets with multi-API clustering data: {list(available.keys())}")

    for data_name, apis in available.items():
        print(f"\nProcessing {data_name} with APIs: {[API_LABELS[a] for a in apis]}")

        df = load_dataset(data_name)
        all_labels = {}
        for api_type in apis:
            labels = load_clustering_labels(data_name, api_type)
            if labels:
                all_labels[api_type] = labels

        if len(all_labels) < 2:
            print(f"  Skipping {data_name}: fewer than 2 APIs with valid labels.")
            continue

        print_summary_table(all_labels, df, data_name)
        plot_cluster_distributions(all_labels, data_name)
        plot_target_by_cluster(all_labels, df, data_name)
        plot_inter_api_agreement(all_labels, data_name)
        plot_rank_performance(all_labels, df, data_name)


if __name__ == "__main__":
    main()
