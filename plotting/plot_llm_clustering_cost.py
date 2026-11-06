#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
LLM Clustering Cost Analysis
=============================
Quantifies token consumption, API latency, and estimated cost of one-time
LLM clustering across datasets, then cross-references with BO performance
gain (LLM clustering vs K-means) to determine when the overhead is justified.

Usage:
    python src/utils/draw/plot_llm_clustering_cost.py
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from copy import deepcopy
import scipy.stats as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_NAMES   = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
DATASET_SIZES = {         # total molecules in each dataset (from CSV counts)
    "redox-mer":   1407,
    "solvation":   1407,
    "kinase":     10449,
    "laser":      10000,
    "pce":        10000,
    "photoswitch":  392,
}
BATCH_SIZE = 200

# API pricing  (USD per 1 000 000 tokens, 2025 rates)
API_PRICING = {
    "gpt": {
        "model":        "gpt-4o",
        "input_per_1M":  2.50,
        "output_per_1M": 10.00,
    },
    "gemini": {
        "model":        "gemini-2.5-pro",
        "input_per_1M":  1.25,
        "output_per_1M": 10.00,
    },
    "qwen3": {
        "model":        "qwen3-plus",
        "input_per_1M":  0.57,
        "output_per_1M": 1.13,
    },
}

# Measured inter-batch delay in prompt_cluster*.py (seconds)
INTER_BATCH_DELAY_S = 10.0
# Approximate LLM response time per batch (empirical, seconds)
API_RESPONSE_TIME_S = {
    "gpt":    15.0,
    "gemini": 20.0,
    "qwen3":  12.0,
}

# BO results root
RESULTS_ROOT  = "results"
F_MODEL       = "molformer"
PROMPT_TYPE   = "just-smiles"
FEATURE_REDU  = "average"
N_INIT        = 10
ACQF          = "ei"
GAMMA         = 0.5
ETA           = 0.005
TREE_DEPTH    = 3
SEEDS         = list(range(15))

# ---------------------------------------------------------------------------
# Prompt prefix lengths  (from PromptCluster.get_prefix_prompt)
# ---------------------------------------------------------------------------

def get_prefix_lengths() -> dict:
    """Return estimated prefix token count per dataset (chars / 4)."""
    try:
        from benchmarks.MAT.prompt_cluster import PromptCluster
        lengths = {}
        for ds in DATA_NAMES:
            pc = PromptCluster(ds, "SMILES")
            lengths[ds] = len(pc.get_prefix_prompt()) / 4.0
        return lengths
    except Exception:
        # Hardcoded fallback from earlier measurement
        return {
            "redox-mer":   171,
            "solvation":   120,
            "kinase":      188,
            "laser":       188,
            "pce":         175,
            "photoswitch": 240,
        }


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

SUFFIX_OVERHEAD = (
    len("**Respond strictly with the counter and numerical cluster labels only. "
        "Do not include any additional text.**\n"
        "Now please cluster the following molecules:\n") / 4.0
)

def estimate_smiles_tokens(data_name: str) -> float:
    """Average SMILES token length per molecule (chars/4) from stored JSON."""
    for api in ("gpt", "gemini", "qwen3"):
        folder = f"benchmarks/MAT/data/llm_clustering{'_api/' + api if api != 'gpt' else '/' }{data_name}"
        jfile  = f"{folder}/{data_name}_0_data.json"
        if os.path.exists(jfile):
            with open(jfile) as f:
                d = json.load(f)
            avg_chars = np.mean([len(s) for s, _ in d])
            return avg_chars / 4.0
    # fallback: average SMILES length ~40 chars
    return 10.0


def compute_token_stats() -> pd.DataFrame:
    """
    For each (dataset, api) compute:
      n_batches, input_tokens_total, output_tokens_total
    """
    prefix_lengths = get_prefix_lengths()
    rows = []

    for ds in DATA_NAMES:
        n_mols    = DATASET_SIZES[ds]
        n_batches = math.ceil(n_mols / BATCH_SIZE)
        prefix_tok = prefix_lengths[ds]
        smiles_tok  = estimate_smiles_tokens(ds)

        # Per-batch: prefix + numbering (~3 chars each) + SMILES tokens + suffix overhead
        numbering_tok = (BATCH_SIZE * 3) / 4.0     # "123. \n" ≈ 3 chars
        mol_tok_per_batch = BATCH_SIZE * (smiles_tok + numbering_tok / BATCH_SIZE)
        input_per_batch   = prefix_tok + SUFFIX_OVERHEAD + mol_tok_per_batch
        output_per_batch  = BATCH_SIZE * 3          # "123 \n" ≈ 3 tokens per molecule

        for api in ("gpt", "gemini", "qwen3"):
            rows.append({
                "dataset":       ds,
                "api":           api,
                "n_molecules":   n_mols,
                "n_batches":     n_batches,
                "input_tokens":  input_per_batch * n_batches,
                "output_tokens": output_per_batch * n_batches,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def compute_costs(token_df: pd.DataFrame) -> pd.DataFrame:
    df = token_df.copy()
    costs_input, costs_output, costs_total = [], [], []
    for _, row in df.iterrows():
        pricing = API_PRICING[row["api"]]
        c_in  = row["input_tokens"]  / 1e6 * pricing["input_per_1M"]
        c_out = row["output_tokens"] / 1e6 * pricing["output_per_1M"]
        costs_input.append(c_in)
        costs_output.append(c_out)
        costs_total.append(c_in + c_out)
    df["cost_input_usd"]  = costs_input
    df["cost_output_usd"] = costs_output
    df["cost_total_usd"]  = costs_total
    return df


# ---------------------------------------------------------------------------
# Latency estimation
# ---------------------------------------------------------------------------

def compute_latency(token_df: pd.DataFrame) -> pd.DataFrame:
    df = token_df.copy()
    latencies = []
    for _, row in df.iterrows():
        api = row["api"]
        n   = row["n_batches"]
        lat = n * (INTER_BATCH_DELAY_S + API_RESPONSE_TIME_S[api])
        latencies.append(lat / 60.0)   # minutes
    df["latency_min"] = latencies
    return df


# ---------------------------------------------------------------------------
# BO performance gain: LLM clustering vs K-means
# ---------------------------------------------------------------------------

def load_trace(data_name, clustering_type, seed, f_model=F_MODEL):
    """
    Load trace_best_y for a given configuration.
    Path layout: results/{data}/{f_model}/llmat/{clustering}/trace_best_y_{suffix}.npy
    Suffix: {N_INIT}_{ACQF}_{GAMMA}_{ETA}_{TREE_DEPTH}[_{P_VAL}]_{seed}
    LLM clustering uses p_val=0.01; kmeans uses p_val=0.
    """
    base = f"{RESULTS_ROOT}/{data_name}/{f_model}/llmat/{clustering_type}"
    if clustering_type == "llms":
        suffix = f"{N_INIT}_{ACQF}_{GAMMA}_{ETA}_{TREE_DEPTH}_0.01_{seed}"
    else:
        suffix = f"{N_INIT}_{ACQF}_{GAMMA}_{ETA}_{TREE_DEPTH}_{seed}"

    path = f"{base}/trace_best_y_{suffix}.npy"
    if not os.path.exists(path):
        return None
    return np.load(path)


def compute_gap(trace_by, ground_truth_opt):
    gap = np.zeros_like(trace_by, dtype=float)
    y0  = trace_by[1]
    for j in range(1, len(trace_by)):
        denom = ground_truth_opt - y0
        gap[j] = np.nan_to_num((trace_by[j] - y0) / denom, nan=1.0) if abs(denom) > 1e-8 else 1.0
    return gap


def load_ground_truth_opt(data_name):
    try:
        from benchmarks.MAT.material_benchmarks import MATBench
        mb = MATBench(
            data_name=data_name,
            run_subset_only=False,
            feature_type=F_MODEL,
            finetuning=False,
            iupac=False,
            prompt_type=PROMPT_TYPE,
            randseed=665,
            clustering_type="kmeans",
            feature_reduction=FEATURE_REDU,
        )
        return mb.ground_truth_opt
    except Exception:
        return None


def auc_gap(gap_curve: np.ndarray, t_start: int = 1, t_end: int = 100) -> float:
    """Area under the GAP curve over rounds t_start..t_end (trapezoidal), in percentage points.
    Expressed as 100 × (normalised AUC) so values are on a 0–100 scale."""
    curve = gap_curve[t_start:t_end + 1]
    return float(np.trapz(curve) / len(curve)) * 100.0   # → percentage points


def compute_gap_deltas() -> pd.DataFrame:
    """
    For each dataset compute mean AUC-GAP for kmeans and llms and their delta.
    AUC-GAP = normalised area under GAP(t) for t = 1..100 (trapezoidal).
    Returns a DataFrame with columns: dataset, auc_kmeans, auc_llms, gap_delta, n_seeds
    """
    rows = []
    for ds in DATA_NAMES:
        opt = load_ground_truth_opt(ds)
        kmeans_aucs, llms_aucs = [], []
        for seed in SEEDS:
            for ct, lst in [("kmeans", kmeans_aucs), ("llms", llms_aucs)]:
                tr = load_trace(ds, ct, seed)
                if tr is not None and opt is not None:
                    g = compute_gap(tr, opt)
                    lst.append(auc_gap(g))   # area under GAP curve

        if kmeans_aucs and llms_aucs:
            n = min(len(kmeans_aucs), len(llms_aucs))
            deltas = np.array(llms_aucs[:n]) - np.array(kmeans_aucs[:n])
            rows.append({
                "dataset":    ds,
                "auc_kmeans": np.mean(kmeans_aucs),
                "auc_llms":   np.mean(llms_aucs),
                "gap_delta":  float(np.mean(deltas)),
                "sem_delta":  float(st.sem(deltas)),
                "n_seeds":    n,
            })
        else:
            rows.append({
                "dataset": ds, "auc_kmeans": None,
                "auc_llms": None, "gap_delta": None,
                "sem_delta": None, "n_seeds": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 13, "axes.titlesize": 13,
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "lines.linewidth": 2, "grid.linestyle": "--", "grid.linewidth": 0.5,
})

API_COLORS  = {"gpt": "#4C72B0", "gemini": "#DD8452", "qwen3": "#55A868"}
API_LABELS  = {"gpt": "GPT-4o", "gemini": "Gemini-2.5-Pro", "qwen3": "Qwen3-Plus"}
DS_LABELS   = {
    "redox-mer":   "Redox-Mer",
    "solvation":   "Solvation",
    "kinase":      "Kinase",
    "laser":       "Laser",
    "pce":         "PCE",
    "photoswitch": "Photoswitch",
}
OUT_DIR = "./results/figs/llm_cost/"


def plot_token_consumption(df: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, col, title in zip(
        axes,
        ["input_tokens", "output_tokens"],
        ["Input Tokens per Dataset", "Output Tokens per Dataset"],
    ):
        pivot = df.pivot(index="dataset", columns="api", values=col) / 1e3  # → K tokens
        pivot.index = [DS_LABELS[d] for d in pivot.index]
        pivot.columns = [API_LABELS[a] for a in pivot.columns]
        pivot.plot(kind="bar", ax=ax, color=[API_COLORS[a] for a in df["api"].unique()],
                   width=0.6, legend=ax == axes[0])
        ax.set_title(title)
        ax.set_ylabel("Tokens (K)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        if ax == axes[0]:
            ax.legend(title="API", fontsize=11, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(path + "token_consumption.pdf", bbox_inches="tight")
    plt.savefig(path + "token_consumption.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}token_consumption.pdf")


def plot_cost_breakdown(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 4))

    ds_order = DATA_NAMES
    x = np.arange(len(ds_order))
    width = 0.22

    for i, api in enumerate(["gpt", "gemini", "qwen3"]):
        sub = df[df["api"] == api].set_index("dataset").reindex(ds_order)
        offset = (i - 1) * width
        bars_in  = ax.bar(x + offset, sub["cost_input_usd"],  width,
                          label=f"{API_LABELS[api]} (input)",
                          color=API_COLORS[api], alpha=0.9)
        bars_out = ax.bar(x + offset, sub["cost_output_usd"], width,
                          bottom=sub["cost_input_usd"],
                          label=f"{API_LABELS[api]} (output)",
                          color=API_COLORS[api], alpha=0.45, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[d] for d in ds_order], rotation=30, ha="right")
    ax.set_ylabel("Estimated Cost (USD)")
    ax.set_title("One-Time LLM Clustering Cost per Dataset")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.3f"))
    ax.legend(fontsize=9, ncol=3, loc="upper left")

    plt.tight_layout()
    plt.savefig(path + "cost_breakdown.pdf", bbox_inches="tight")
    plt.savefig(path + "cost_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}cost_breakdown.pdf")


def plot_latency(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7, 4))

    pivot = df.pivot(index="dataset", columns="api", values="latency_min")
    pivot.index = [DS_LABELS[d] for d in pivot.index]
    pivot.columns = [API_LABELS[a] for a in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=[API_COLORS[a] for a in ["gpt","gemini","qwen3"]],
               width=0.6)
    ax.set_ylabel("Wall-Clock Latency (min)")
    ax.set_title("Estimated LLM Clustering Latency (one-time)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="API", fontsize=11, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(path + "latency.pdf", bbox_inches="tight")
    plt.savefig(path + "latency.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}latency.pdf")


def plot_cost_vs_gain(cost_df: pd.DataFrame, gap_df: pd.DataFrame, path: str):
    """Scatter: x = GPT cost, y = Δ AUC-GAP (llm - kmeans)."""
    merged = cost_df[cost_df["api"] == "gpt"].merge(gap_df, on="dataset")
    merged = merged.dropna(subset=["gap_delta"])

    if merged.empty:
        print("No BO results found — skipping cost-vs-gain plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in merged.iterrows():
        ax.errorbar(
            row["cost_total_usd"], row["gap_delta"],
            yerr=row["sem_delta"] if row["sem_delta"] else 0,
            fmt="o", ms=9, capsize=4, color=API_COLORS["gpt"],
        )
        ax.annotate(DS_LABELS[row["dataset"]],
                    (row["cost_total_usd"], row["gap_delta"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=10)

    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("GPT-4o Clustering Cost (USD, one-time)")
    ax.set_ylabel("Δ AUC-GAP (pp)  (LLM − KMeans)")
    ax.set_title("Cost vs. BO Performance Gain from LLM Clustering")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

    plt.tight_layout()
    plt.savefig(path + "cost_vs_gain.pdf", bbox_inches="tight")
    plt.savefig(path + "cost_vs_gain.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}cost_vs_gain.pdf")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _worthwhile(delta) -> str:
    if delta is None or np.isnan(delta):
        return "N/A"
    if delta > 2.0:
        return "Yes"
    if delta > 0:
        return "Marginal"
    return "No"


def _fmt_n(n: int) -> str:
    """Format integer with thin-space thousands separator, e.g. 10 449."""
    s = f"{n:,}".replace(",", " ")   # Unicode thin space
    return s


def print_summary_table(cost_df: pd.DataFrame, gap_df: pd.DataFrame):
    cost_gpt    = cost_df[cost_df["api"] == "gpt"].set_index("dataset")
    cost_gemini = cost_df[cost_df["api"] == "gemini"].set_index("dataset")
    cost_qwen   = cost_df[cost_df["api"] == "qwen3"].set_index("dataset")
    gap_idx     = gap_df.set_index("dataset")

    HDR = (f"{'Dataset':<13}  {'Molecules':>10}  {'GPT cost':>9}  "
           f"{'Gemini cost':>12}  {'Qwen3 cost':>11}  "
           f"{'Latency':>9}  {'Δ AUC-GAP (pp)':>15}  {'Worthwhile?'}")
    SEP = "-" * len(HDR)

    print()
    print(HDR)
    print(SEP)

    total_gpt = total_gem = total_qw = total_lat = 0.0
    for ds in DATA_NAMES:
        n    = DATASET_SIZES[ds]
        cg   = cost_gpt.loc[ds, "cost_total_usd"]
        cgem = cost_gemini.loc[ds, "cost_total_usd"]
        cq   = cost_qwen.loc[ds, "cost_total_usd"]
        lat  = cost_gpt.loc[ds, "latency_min"]
        total_gpt += cg; total_gem += cgem; total_qw += cq; total_lat += lat

        delta = gap_idx.loc[ds, "gap_delta"] if ds in gap_idx.index else None
        if delta is not None and not (isinstance(delta, float) and np.isnan(delta)):
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.3f}"
        else:
            delta_str = "N/A"

        lat_str = f"{lat:.1f} min"

        print(f"{DS_LABELS[ds]:<13}  {_fmt_n(n):>10}  ${cg:>7.3f}  "
              f"${cgem:>10.3f}  ${cq:>9.3f}  "
              f"{lat_str:>9}  {delta_str:>15}  {_worthwhile(delta)}")

    print(SEP)
    print(f"{'Total':<13}  {'':>10}  ${total_gpt:>7.2f}  "
          f"${total_gem:>10.2f}  ${total_qw:>9.2f}  "
          f"{total_lat:.0f} min")
    print()
    print("Δ AUC-GAP (pp) = 100 × [AUC(GAP, LLM) − AUC(GAP, K-means)] / T,  T=1..100.")
    print("Worthwhile: Δ > 2 pp  |  Marginal: 0 < Δ ≤ 2 pp  |  No: Δ ≤ 0")
    print()


def save_latex_table(cost_df: pd.DataFrame, gap_df: pd.DataFrame, path: str):
    """Save the summary as a LaTeX booktabs table."""
    cost_gpt    = cost_df[cost_df["api"] == "gpt"].set_index("dataset")
    cost_gemini = cost_df[cost_df["api"] == "gemini"].set_index("dataset")
    cost_qwen   = cost_df[cost_df["api"] == "qwen3"].set_index("dataset")
    gap_idx     = gap_df.set_index("dataset")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLM clustering cost, latency, and BO gain ($\Delta$ AUC-GAP, in percentage points) per dataset.}",
        r"\label{tab:llm_cost}",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Dataset & $N$ & GPT-4o & Gemini & Qwen3 & Latency & $\Delta$ AUC-GAP (pp) & Worth? \\",
        r"\midrule",
    ]

    total_gpt = total_gem = total_qw = total_lat = 0.0
    for ds in DATA_NAMES:
        n    = DATASET_SIZES[ds]
        cg   = cost_gpt.loc[ds, "cost_total_usd"]
        cgem = cost_gemini.loc[ds, "cost_total_usd"]
        cq   = cost_qwen.loc[ds, "cost_total_usd"]
        lat  = cost_gpt.loc[ds, "latency_min"]
        total_gpt += cg; total_gem += cgem; total_qw += cq; total_lat += lat

        delta = gap_idx.loc[ds, "gap_delta"] if ds in gap_idx.index else None
        if delta is not None and not (isinstance(delta, float) and np.isnan(delta)):
            sign = "+" if delta >= 0 else ""
            delta_str = f"${sign}{delta:.3f}$"
            worth = r"\textbf{Yes}" if delta > 2.0 else ("Marginal" if delta > 0 else "No")
        else:
            delta_str = "--"
            worth = "--"

        lines.append(
            f"{DS_LABELS[ds]} & {n:,} & \\${cg:.3f} & \\${cgem:.3f} & "
            f"\\${cq:.3f} & {lat:.1f}\\,min & {delta_str} & {worth} \\\\"
        )

    lines += [
        r"\midrule",
        f"Total & & \\${total_gpt:.2f} & \\${total_gem:.2f} & \\${total_qw:.2f} & "
        f"{total_lat:.0f}\\,min & & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    fpath = path + "llm_clustering_cost.tex"
    with open(fpath, "w") as f:
        f.write(tex)
    print(f"Saved: {fpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Computing token statistics...")
    token_df = compute_token_stats()

    print("Computing costs...")
    cost_df = compute_costs(token_df)

    print("Computing latencies...")
    cost_df = compute_latency(cost_df)

    print("Loading BO results for GAP delta...")
    gap_df = compute_gap_deltas()

    print_summary_table(cost_df, gap_df)
    save_latex_table(cost_df, gap_df, OUT_DIR)

    print("Plotting...")
    plot_token_consumption(cost_df, OUT_DIR)
    plot_cost_breakdown(cost_df, OUT_DIR)
    plot_latency(cost_df, OUT_DIR)
    plot_cost_vs_gain(cost_df, gap_df, OUT_DIR)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
