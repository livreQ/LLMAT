"""Analyze LLMAT runtime scaling w.r.t. candidate pool size."""

import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

seeds = list(range(15))

DATASET_SIZES = {
    "photoswitch": 392,
    "redox-mer": 1407,
    "solvation": 1407,
    "kinase": 10449,
    "laser": 10000,
    "pce": 10000,
}

f_models = ["t5-base-chem", "molformer", "t5-base", "gpt2-medium", "gpt2-large", "llama-2-7b"]
prompt_type = "just-smiles"
feature_reduction = "average"

# Hidden/embedding dimension of each feature extractor's output representation.
FEATURE_MODEL_DIMS = {
    "t5-base-chem": {"dim": 768, "label": "T5-Base (Chem)"},
    "t5-base": {"dim": 768, "label": "T5-Base"},
    "molformer": {"dim": 768, "label": "MoLFormer-XL"},
    "gpt2-medium": {"dim": 1024, "label": "GPT2-Medium"},
    "gpt2-large": {"dim": 1280, "label": "GPT2-Large"},
    "llama-2-7b": {"dim": 4096, "label": "Llama-2-7B"},
}


def _get_feature_name(f_model):
    if f_model not in ["fingerprints", "molformer", "gin", "gcn", "schnet"]:
        return f"{f_model}-{prompt_type}-{feature_reduction}"
    return f_model


def load_timing(data_name, f_model, algo, seed):
    feat_name = _get_feature_name(f_model)
    if algo == "llmat":
        base = f"results/{data_name}/{feat_name}/llmat/kmeans"
        suffix = f"10_ei_0.5_0.005_3_{seed}"
    else:
        base = f"results/{data_name}/{feat_name}/lapeft"
        suffix = f"10_ts_all_layer_{algo}_{seed}"

    paths = {
        "total": f"{base}/trace_timing_{suffix}.npy",
        "train": f"{base}/timing_train_{suffix}.npy",
        "preds": f"{base}/timing_preds_{suffix}.npy",
    }
    result = {}
    for key, path in paths.items():
        if os.path.exists(path):
            result[key] = np.load(path)
        else:
            return None
    return result


def collect_timing_data():
    rows = []
    for algo in ["llmat", "gp", "laplace"]:
        for data_name in DATASET_SIZES:
            pool_size = DATASET_SIZES[data_name]
            for f_model in f_models:
                for seed in seeds:
                    t = load_timing(data_name, f_model, algo, seed)
                    if t is not None:
                        n_rounds = min(len(t["train"]), len(t["preds"]))
                        if n_rounds < 10:
                            continue
                        total_nonzero = t["total"][t["total"] > 0]
                        rows.append({
                            "algo": algo,
                            "dataset": data_name,
                            "pool_size": pool_size,
                            "f_model": f_model,
                            "f_model_dim": FEATURE_MODEL_DIMS[f_model]["dim"],
                            "seed": seed,
                            "n_rounds": n_rounds,
                            "train_per_round": np.mean(t["train"][:n_rounds]),
                            "preds_per_round": np.mean(t["preds"][:n_rounds]),
                            "total_per_round": np.mean(total_nonzero) if len(total_nonzero) > 0 else 0,
                        })
    return pd.DataFrame(rows)


def print_scaling_table(df, f_models_filter=("t5-base-chem", "molformer")):
    algo_names = {"llmat": "LLMAT", "gp": "GP", "laplace": "Laplace"}
    df = df[df["f_model"].isin(f_models_filter)]
    print("\n" + "=" * 120)
    print(f"Runtime Scaling: Average Time per BO Round (seconds) — grouped by candidate pool size "
          f"(f_model: {', '.join(f_models_filter)})")
    print("=" * 120)
    header = (f"{'Algorithm':<10} | {'Dataset':<12} | {'Pool Size':>10} | "
              f"{'Train (s/round)':>18} | {'Predict (s/round)':>18} | {'Total (s/round)':>18} | {'#runs':>5}")
    print(header)
    print("-" * 120)

    for algo in ["llmat", "gp", "laplace"]:
        sub = df[df["algo"] == algo]
        for data_name in ["photoswitch", "redox-mer", "solvation", "kinase", "laser", "pce"]:
            d = sub[sub["dataset"] == data_name]
            if len(d) == 0:
                continue
            train_m = d["train_per_round"].mean()
            train_s = d["train_per_round"].sem()
            preds_m = d["preds_per_round"].mean()
            preds_s = d["preds_per_round"].sem()
            total_m = d["total_per_round"].mean()
            total_s = d["total_per_round"].sem()
            print(f"{algo_names[algo]:<10} | {data_name:<12} | {DATASET_SIZES[data_name]:>10,} | "
                  f"{train_m:>7.3f} ± {train_s:>5.3f}   | {preds_m:>7.3f} ± {preds_s:>5.3f}   | "
                  f"{total_m:>7.3f} ± {total_s:>5.3f}   | {len(d):>5}")
        print("-" * 120)

    # Per-algorithm average across the selected f_models, grouped by pool size
    print("\n" + "=" * 90)
    print(f"Summary: Mean time per round averaged across {', '.join(f_models_filter)} and seeds")
    print("=" * 90)
    print(f"{'Algorithm':<10} | {'Pool Size':>10} | {'Train (s)':>12} | {'Predict (s)':>12} | {'Total (s)':>12}")
    print("-" * 90)
    for algo in ["llmat", "gp", "laplace"]:
        sub = df[df["algo"] == algo]
        for pool_size in sorted(sub["pool_size"].unique()):
            d = sub[sub["pool_size"] == pool_size]
            print(f"{algo_names[algo]:<10} | {pool_size:>10,} | "
                  f"{d['train_per_round'].mean():>10.3f}   | "
                  f"{d['preds_per_round'].mean():>10.3f}   | "
                  f"{d['total_per_round'].mean():>10.3f}")
        print("-" * 90)


def print_fmodel_scaling_table(df):
    algo_names = {"llmat": "LLMAT", "gp": "GP", "laplace": "Laplace"}
    fmodels_by_dim = sorted(f_models, key=lambda m: FEATURE_MODEL_DIMS[m]["dim"])

    print("\n" + "=" * 120)
    print("Runtime Scaling: Average Time per BO Round (seconds) — grouped by feature model (sorted by feature dim)")
    print("=" * 120)
    header = (f"{'Algorithm':<10} | {'Feature Model':<16} | {'Dim':>5} | "
              f"{'Train (s/round)':>18} | {'Predict (s/round)':>18} | {'Total (s/round)':>18} | {'#runs':>5}")
    print(header)
    print("-" * 120)

    for algo in ["llmat", "gp", "laplace"]:
        sub = df[df["algo"] == algo]
        for f_model in fmodels_by_dim:
            d = sub[sub["f_model"] == f_model]
            if len(d) == 0:
                continue
            info = FEATURE_MODEL_DIMS[f_model]
            train_m, train_s = d["train_per_round"].mean(), d["train_per_round"].sem()
            preds_m, preds_s = d["preds_per_round"].mean(), d["preds_per_round"].sem()
            total_m, total_s = d["total_per_round"].mean(), d["total_per_round"].sem()
            print(f"{algo_names[algo]:<10} | {info['label']:<16} | {info['dim']:>5} | "
                  f"{train_m:>7.3f} ± {train_s:>5.3f}   | {preds_m:>7.3f} ± {preds_s:>5.3f}   | "
                  f"{total_m:>7.3f} ± {total_s:>5.3f}   | {len(d):>5}")
        print("-" * 120)


def plot_scaling(df, save_path="results/figs/scaling/"):
    os.makedirs(save_path, exist_ok=True)
    algo_names = {"llmat": "LLMAT", "gp": "GP", "laplace": "Laplace"}
    algo_colors = {"llmat": "#e74c3c", "gp": "#3498db", "laplace": "#2ecc71"}

    # Aggregate: mean across f_models and seeds per (algo, pool_size)
    agg = df.groupby(["algo", "pool_size"]).agg(
        train_mean=("train_per_round", "mean"),
        train_sem=("train_per_round", "sem"),
        preds_mean=("preds_per_round", "mean"),
        preds_sem=("preds_per_round", "sem"),
        total_mean=("total_per_round", "mean"),
        total_sem=("total_per_round", "sem"),
    ).reset_index()

    # --- Plot 1: Total time per round vs pool size ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ["llmat", "gp", "laplace"]:
        sub = agg[agg["algo"] == algo].sort_values("pool_size")
        ax.errorbar(sub["pool_size"], sub["total_mean"], yerr=sub["total_sem"],
                     marker="o", label=algo_names[algo], color=algo_colors[algo],
                     linewidth=2, markersize=7, capsize=4)
    ax.set_xlabel("Candidate Pool Size (N)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time per BO Round (s)", fontsize=14, fontweight="bold")
    ax.set_title("Runtime Scaling: Total Time per Round", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xscale("log")
    ax.tick_params(labelsize=12)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + "total_time_vs_pool_size.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # --- Plot 2: Train vs Predict breakdown (stacked/side-by-side) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for i, algo in enumerate(["llmat", "gp", "laplace"]):
        ax = axes[i]
        sub = agg[agg["algo"] == algo].sort_values("pool_size")
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, sub["train_mean"], w, yerr=sub["train_sem"],
               label="Train", color=algo_colors[algo], alpha=0.7, capsize=3)
        ax.bar(x + w/2, sub["preds_mean"], w, yerr=sub["preds_sem"],
               label="Predict", color=algo_colors[algo], alpha=0.35, capsize=3,
               hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(s):,}" for s in sub["pool_size"]], rotation=30, fontsize=10)
        ax.set_xlabel("Pool Size", fontsize=12)
        ax.set_title(algo_names[algo], fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=11)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Time per Round (s)", fontsize=13, fontweight="bold")
    plt.suptitle("Train vs. Predict Time Breakdown", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path + "train_vs_predict_breakdown.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # --- Plot 3: Per-candidate prediction cost (verifies linearity) ---
    agg["pred_per_candidate_ms"] = agg["preds_mean"] / agg["pool_size"] * 1000
    agg["pred_per_candidate_sem"] = agg["preds_sem"] / agg["pool_size"] * 1000

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ["llmat", "gp", "laplace"]:
        sub = agg[agg["algo"] == algo].sort_values("pool_size")
        ax.errorbar(sub["pool_size"], sub["pred_per_candidate_ms"],
                     yerr=sub["pred_per_candidate_sem"],
                     marker="s", label=algo_names[algo], color=algo_colors[algo],
                     linewidth=2, markersize=7, capsize=4)

    ax.set_xlabel("Candidate Pool Size (N)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Prediction Cost per Candidate (ms)", fontsize=14, fontweight="bold")
    ax.set_title("Per-Candidate Prediction Cost (constant $\\Rightarrow$ linear scaling)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xscale("log")
    ax.tick_params(labelsize=12)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + "per_candidate_cost.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # --- Plot 4: Extrapolation using per-candidate cost ---
    # Use median per-candidate cost from pools >= 1000 (skip tiny photoswitch)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    extrapolate_sizes = np.array([1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6])

    for algo in ["llmat", "gp", "laplace"]:
        sub = agg[(agg["algo"] == algo) & (agg["pool_size"] >= 1000)]
        if len(sub) == 0:
            continue
        cost_per_cand = (sub["preds_mean"] / sub["pool_size"]).median()
        train_cost = sub["train_mean"].median()

        y_pred = cost_per_cand * extrapolate_sizes
        y_total = train_cost + y_pred

        ax.plot(extrapolate_sizes, y_total, marker="o", label=algo_names[algo],
                color=algo_colors[algo], linewidth=2, markersize=5)

        # Print the constants used
        print(f"{algo_names[algo]}: train={train_cost:.3f}s (fixed) + "
              f"pred={cost_per_cand*1000:.4f}ms × N")

    ax.set_xlabel("Candidate Pool Size (N)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Estimated Time per Round (s)", fontsize=14, fontweight="bold")
    ax.set_title("Extrapolated Runtime: $T = T_{train} + c \\times N$",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.axhline(y=60, color="gray", linestyle=":", alpha=0.5)
    ax.text(extrapolate_sizes[-1], 60, "1 min", ha="right", va="bottom", fontsize=10, color="gray")
    ax.axhline(y=3600, color="gray", linestyle=":", alpha=0.5)
    ax.text(extrapolate_sizes[-1], 3600, "1 hour", ha="right", va="bottom", fontsize=10, color="gray")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + "extrapolated_scaling.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_fmodel_scaling(df, save_path="results/figs/scaling/"):
    os.makedirs(save_path, exist_ok=True)
    algo_names = {"llmat": "LLMAT", "gp": "GP", "laplace": "Laplace"}
    algo_colors = {"llmat": "#e74c3c", "gp": "#3498db", "laplace": "#2ecc71"}
    fmodels_by_dim = sorted(f_models, key=lambda m: FEATURE_MODEL_DIMS[m]["dim"])

    # Aggregate: mean across datasets/seeds per (algo, f_model)
    agg = df.groupby(["algo", "f_model"]).agg(
        train_mean=("train_per_round", "mean"),
        train_sem=("train_per_round", "sem"),
        preds_mean=("preds_per_round", "mean"),
        preds_sem=("preds_per_round", "sem"),
        total_mean=("total_per_round", "mean"),
        total_sem=("total_per_round", "sem"),
    ).reset_index()
    agg["dim"] = agg["f_model"].map(lambda m: FEATURE_MODEL_DIMS[m]["dim"])

    # --- Plot: Total time per round vs feature dimension ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ["llmat", "gp", "laplace"]:
        sub = agg[agg["algo"] == algo].sort_values("dim")
        if len(sub) == 0:
            continue
        ax.errorbar(sub["dim"], sub["total_mean"], yerr=sub["total_sem"],
                     marker="o", label=algo_names[algo], color=algo_colors[algo],
                     linewidth=2, markersize=7, capsize=4)
    ax.set_xlabel("Feature Dimension", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time per BO Round (s)", fontsize=14, fontweight="bold")
    ax.set_title("Runtime Scaling: Total Time vs. Feature Dimension", fontsize=13, fontweight="bold")
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + "total_time_vs_fmodel_dim.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # --- Plot: Train vs Predict breakdown per feature model, faceted by algo ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for i, algo in enumerate(["llmat", "gp", "laplace"]):
        ax = axes[i]
        sub = agg[agg["algo"] == algo].set_index("f_model").reindex(fmodels_by_dim).dropna(how="all")
        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, sub["train_mean"], w, yerr=sub["train_sem"],
               label="Train", color=algo_colors[algo], alpha=0.7, capsize=3)
        ax.bar(x + w/2, sub["preds_mean"], w, yerr=sub["preds_sem"],
               label="Predict", color=algo_colors[algo], alpha=0.35, capsize=3,
               hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{FEATURE_MODEL_DIMS[m]['label']}\n(dim={FEATURE_MODEL_DIMS[m]['dim']})" for m in sub.index],
            rotation=30, ha="right", fontsize=9,
        )
        ax.set_title(algo_names[algo], fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=10)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Time per Round (s)", fontsize=13, fontweight="bold")
    plt.suptitle("Train vs. Predict Time by Feature Model", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path + "train_vs_predict_by_fmodel.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_fmodel_pool_scaling(df, save_path="results/figs/scaling/"):
    """Mean time per BO round vs. candidate pool size, one curve per feature model."""
    os.makedirs(save_path, exist_ok=True)
    algo_names = {"llmat": "LLMAT", "gp": "GP", "laplace": "Laplace"}
    fmodels_by_dim = sorted(f_models, key=lambda m: FEATURE_MODEL_DIMS[m]["dim"])
    cmap = plt.cm.viridis(np.linspace(0, 1, len(fmodels_by_dim)))
    fmodel_colors = {m: cmap[i] for i, m in enumerate(fmodels_by_dim)}

    agg = df.groupby(["algo", "f_model", "pool_size"]).agg(
        total_mean=("total_per_round", "mean"),
        total_sem=("total_per_round", "sem"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for i, algo in enumerate(["llmat", "gp", "laplace"]):
        ax = axes[i]
        sub_algo = agg[agg["algo"] == algo]
        for f_model in fmodels_by_dim:
            sub = sub_algo[sub_algo["f_model"] == f_model].sort_values("pool_size")
            if len(sub) == 0:
                continue
            info = FEATURE_MODEL_DIMS[f_model]
            ax.errorbar(sub["pool_size"], sub["total_mean"], yerr=sub["total_sem"],
                         marker="o", label=f"{info['label']} (dim={info['dim']})",
                         color=fmodel_colors[f_model], linewidth=2, markersize=6, capsize=3)
        ax.set_xlabel("Candidate Pool Size (N)", fontsize=12, fontweight="bold")
        ax.set_title(algo_names[algo], fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.tick_params(labelsize=10)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean Time per BO Round (s)", fontsize=13, fontweight="bold")
    axes[-1].legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.suptitle("Mean Time per BO Round vs. Pool Size, by Feature Model", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path + "time_vs_pool_size_by_fmodel.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_scaling_per_fmodel(df, save_path="results/figs/scaling/"):
    """One standalone pool-size scaling plot per feature model, with a curve per algorithm."""
    os.makedirs(save_path, exist_ok=True)
    algo_names = {"llmat": "LLMAT", "gp": "GP", "laplace": "Laplace"}
    algo_colors = {"llmat": "#e74c3c", "gp": "#3498db", "laplace": "#2ecc71"}

    for f_model in f_models:
        sub_model = df[df["f_model"] == f_model]
        if len(sub_model) == 0:
            continue
        agg = sub_model.groupby(["algo", "pool_size"]).agg(
            total_mean=("total_per_round", "mean"),
            total_sem=("total_per_round", "sem"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algo in ["llmat", "gp", "laplace"]:
            sub = agg[agg["algo"] == algo].sort_values("pool_size")
            if len(sub) == 0:
                continue
            ax.errorbar(sub["pool_size"], sub["total_mean"], yerr=sub["total_sem"],
                         marker="o", label=algo_names[algo], color=algo_colors[algo],
                         linewidth=2, markersize=7, capsize=4)
        info = FEATURE_MODEL_DIMS[f_model]
        ax.set_xlabel("Candidate Pool Size (N)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Mean Time per BO Round (s)", fontsize=14, fontweight="bold")
        ax.set_title(f"Runtime Scaling: {info['label']} (dim={info['dim']})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=12)
        ax.set_xscale("log")
        ax.tick_params(labelsize=12)
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.savefig(save_path + f"total_time_vs_pool_size_{f_model}.pdf", bbox_inches="tight", dpi=300)
        plt.show()


if __name__ == "__main__":
    print("Collecting timing data across all datasets, feature models, and algorithms...")
    df = collect_timing_data()
    print(f"Collected {len(df)} timing records.")
    print_scaling_table(df)
    print_fmodel_scaling_table(df)
    plot_scaling(df)
    plot_fmodel_scaling(df)
    plot_fmodel_pool_scaling(df)
    plot_scaling_per_fmodel(df)
