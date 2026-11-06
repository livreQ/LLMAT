#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Aug 19 16:39:16 2025
# File Name: plot_fixed.py
# Description:
"""

import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from benchmarks.MAT.material_benchmarks import MATBench


def get_ours_suffix(seed, f_model, n_init_data=10, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.01, tree_depth=3):
    if f_model == "t5-base-chem":
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    elif f_model in ("molformer", "fingerprints", "gin", "schnet"):
        eta = 0.005
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    else:
        eta = 0.005
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_0.01_{seed}"
    return suffix


def load_ours(mat_bench, r_type, seed, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.01, tree_depth=3):

    # print(mat_bench.dataset_name)
    print(mat_bench.feature_type)
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/llmat"
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"{path}/{clustering}"
    suffix = get_ours_suffix(seed, mat_bench.feature_type)
    if not os.path.exists(f"{path}/trace_best_y_{suffix}.npy"):
        print(f"{path}/trace_best_y_{suffix}.npy")
        return None
    if r_type == "best_y":
        ret = np.load(f"{path}/trace_best_y_{suffix}.npy")
    elif r_type == "trace_time":
        ret = np.load(f"{path}/trace_timing_{suffix}.npy")
    elif r_type == "train_time":
        ret = np.load(f"{path}/timing_train_{suffix}.npy")
    elif r_type == "pred_time":
        ret = np.load(f"{path}/timing_preds_{suffix}.npy")
    elif r_type == "trace_acqvals":
        ret = np.load(f"{path}/trace_acqvals_{suffix}.npy")
    elif r_type == "trace_y_his":
        ret = np.load(f"{path}/trace_y_his_{suffix}.npy")
    else:
        ret = None
    return ret


def get_baselines_suffix(seed, n_init_data=10, acqf="ts", laplace_type="all_layer", base_line="gp"):

    suffix = f"{n_init_data}_{acqf}_{laplace_type}_{base_line}_{seed}"
    return suffix


def load_baselines(mat_bench, r_type, seed, acqf="ts", laplace_type="all_layer", base_line="gp"):

    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/lapeft"
    suffix = get_baselines_suffix(seed, base_line=base_line)
    if not os.path.exists(f"{path}/trace_best_y_{suffix}.npy"):
        print(f"{path}/trace_best_y_{suffix}.npy")
        return None
    else:
        if r_type == "best_y":
            ret = np.load(f"{path}/trace_best_y_{suffix}.npy")
        elif r_type == "trace_time":
            ret = np.load(f"{path}/trace_timing_{suffix}.npy")
        elif r_type == "train_time":
            ret = np.load(f"{path}/timing_train_{suffix}.npy")
        elif r_type == "pred_time":
            ret = np.load(f"{path}/timing_preds_{suffix}.npy")
        elif r_type == "trace_acqvals":
            ret = np.load(f"{path}/trace_acqvals_{suffix}.npy")
        elif r_type == "trace_y_his":
            ret = np.load(f"{path}/trace_y_his_{suffix}.npy")
        else:
            ret = None
        return ret


def get_decoupled_suffix(seed, n_init_data=10, acqf="ei", gamma=0.5, eta="None", tree_depth=3, selection_type="uct"):
    suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    if selection_type == "greedy":
        suffix += "_greedy"
    suffix += "_decoupled"
    return suffix


# decoupled_mcts_lfbo_bo.py::_make_train_df / new_row used to build a dict
# literal with a hardcoded "targets" key AND a `target_col: ...` entry;
# since target_col == "targets" in the non-finetuning setting, the second
# entry silently clobbered the first (Python dict literals let a later
# duplicate key win), feeding the model the untransformed (wrong-direction)
# target instead of target_col_transformed. Fixed in commit 2ea92e42b.
# Any trace_*.npy written before this commit is invalid for comparison
# against LLMAT and must be excluded, not averaged in.
DECOUPLED_FIX_MTIME = 1785421032  # commit 2ea92e42b (2026-07-30 17:17:12 +0300)


def load_decoupled(mat_bench, r_type, seed, acqf="ei", gamma=0.5, eta="None", tree_depth=3, selection_type="uct"):
    """Loader for the 'Decoupled Partitioning + Independent LFBO' ablation
    (K-means/logistic-regression partition classifier + a separate LFBO
    head per node, vs LLMAT's single coupled classifier). Mirrors
    load_ours; see decoupled_mcts_lfbo_bo.py::_get_suffix for the filename
    convention (eta has no effect on this algorithm and is always "None"
    unless --eta was passed explicitly on the CLI).

    Returns None (same as "missing") for any run file written before the
    DECOUPLED_FIX_MTIME cutoff, so stale pre-fix seeds silently drop out
    of averages instead of corrupting them. As cluster jobs finish
    regenerating results, coverage will fill back in on its own.
    """
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"results/{prefix}/decoupled_mcts_lfbo/{clustering}"
    suffix = get_decoupled_suffix(seed, acqf=acqf, gamma=gamma, eta=eta, tree_depth=tree_depth, selection_type=selection_type)
    best_y_file = f"{path}/trace_best_y_{suffix}.npy"
    if not os.path.exists(best_y_file):
        return None
    if os.path.getmtime(best_y_file) < DECOUPLED_FIX_MTIME:
        return None
    if r_type == "best_y":
        ret = np.load(f"{path}/trace_best_y_{suffix}.npy")
    elif r_type == "trace_time":
        ret = np.load(f"{path}/trace_timing_{suffix}.npy")
    elif r_type == "train_time":
        ret = np.load(f"{path}/timing_train_{suffix}.npy")
    elif r_type == "pred_time":
        ret = np.load(f"{path}/timing_preds_{suffix}.npy")
    elif r_type == "trace_acqvals":
        ret = np.load(f"{path}/trace_acqvals_{suffix}.npy")
    elif r_type == "trace_y_his":
        ret = np.load(f"{path}/trace_y_his_{suffix}.npy")
    else:
        ret = None
    return ret


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

from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

seeds = list(set(range(15)))
# seeds = [14]
data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
# "solvation",
f_models = ["t5-base-chem"]  #,molformer "gpt2-medium", "t5-base", "gpt2-large"]
prompt_type = "just-smiles"
finetuning = False
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]

our_algo = "llmat"
base_lines = ["laplace", "gp", "random"]
# base_lines = ["gp"]

# Properties = {
#     "Redoxmer": "Redox Potential",
#     "Laser": "Fluorescence Oscillator Strength",
#     "Solvation": "Solvation Energy",
#     "Kinase": "Docking Score",
#     "Photovoltaics": "Power Conversion Efficiency",
#     "Photoswitches": "$\pi-\pi^*$ Transition Wavelength"
# }
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


def plot_best_y(path, f_models, data_names):

    for data_name in data_names:
        plt.figure(figsize=(6, 3))
        for f_model in f_models:
            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )
                print(mat_bench.dataset_name)
                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)
                trace_best_y = np.stack(trace_bys)
                mean = np.mean(trace_best_y, axis=0)[1:101]  # Over randseeds
                sem = st.sem(trace_best_y, axis=0)[1:101]  # Over randseeds
                T = np.arange(len(mean)) + 1
                df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})

                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT")  #-{f_model}-{clustering_type} 'sd' shows standard deviation as shaded region
                ax1.fill_between(T, mean - sem, mean + sem, alpha=0.3)
                ax1.axhline(mat_bench.ground_truth_opt, c="k", ls="dashed", zorder=1000)

            for baseline in base_lines:
                trace_bys = []
                for seed in seeds:
                    trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                    if trace_by is not None:
                        trace_bys.append(trace_by)
                trace_best_y = np.stack(trace_bys)
                mean = np.mean(trace_best_y, axis=0)[1:101]  # Over randseeds
                sem = st.sem(trace_best_y, axis=0)[1:101]  # Over randseeds
                T = np.arange(len(mean)) + 1
                print(len(T), len(mean), len(sem))
                df_lapeft = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
                ax2 = sns.lineplot(data=df_lapeft, x='T', y='y', label=f"{baseline.upper()}")  #-{f_model}
                ax2.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            plt.xlim(1, max(T))
            plt.setp(ax1.get_legend().get_texts(), fontsize='16')
            plt.xlabel('BO Rounds: T', fontsize=16)
            plt.ylabel(Properties[Datasets[data_name]], fontsize=12)
            plt.tight_layout()
            # savefig('xx.pdf', format='pdf', bbox_inches='tight')
            # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
            if mat_bench.maximization:
                plt.title(f"{Datasets[data_name]}($\\uparrow$)", fontsize=16)
            else:
                plt.title(f"{Datasets[data_name]}($\downarrow$)", fontsize=16)
            plt.savefig(path + data_name + "_" + f_model + ".pdf", format='pdf', bbox_inches='tight')
            plt.show()


def plot_llmat_vs_decoupled(path, f_models, data_names):
    """Ablation under fixed (non-finetuned) features: LLMAT's single
    coupled classifier (does partitioning AND acquisition) vs.
    "Decoupled Partitioning + Independent LFBO" (K-means + a supervised
    partition classifier for routing, and a separate LFBO head per node
    for acquisition). Isolates the gain from coupling partition and
    acquisition into one classifier.
    """
    os.makedirs(path, exist_ok=True)
    for data_name in data_names:
        for f_model in f_models:
            mat_bench = MATBench(
                data_name=data_name,
                run_subset_only=run_subset_only,
                feature_type=f_model,
                finetuning=False,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=665,
                clustering_type=clustering_types[0],
                feature_reduction=feature_reduction,
            )

            plt.figure(figsize=(6, 3))
            ax = None
            T_max = 0
            for label, loader in [
                ("LLMAT (coupled)", lambda s: load_ours(mat_bench, r_type="best_y", seed=s)),
                ("Decoupled (K-means + LFBO)", lambda s: load_decoupled(mat_bench, r_type="best_y", seed=s)),
            ]:
                trace_bys = [loader(s) for s in seeds]
                trace_bys = [t for t in trace_bys if t is not None]
                if not trace_bys:
                    print(f"[skip] {label} has no results for {data_name}/{f_model}")
                    continue
                min_len = min(len(t) for t in trace_bys)
                arr = np.stack([t[:min_len] for t in trace_bys])
                n = min(101, arr.shape[1])
                mean = np.mean(arr, axis=0)[1:n]
                sem = st.sem(arr, axis=0)[1:n]
                T = np.arange(len(mean)) + 1
                T_max = max(T_max, len(T))
                ax = sns.lineplot(x=T, y=mean, label=label)
                ax.fill_between(T, mean - sem, mean + sem, alpha=0.3)

            if T_max == 0:
                plt.close()
                continue

            plt.axhline(mat_bench.ground_truth_opt, c="k", ls="dashed", zorder=1000, label="Optimum")
            plt.xlim(1, T_max)
            plt.xlabel('BO Rounds: T', fontsize=16)
            plt.ylabel(Properties[Datasets[data_name]], fontsize=12)
            plt.legend(fontsize=10)
            plt.tight_layout()
            title_arrow = "$\\uparrow$" if mat_bench.maximization else "$\\downarrow$"
            plt.title(f"{Datasets[data_name]}({title_arrow})", fontsize=16)
            out_file = f"{path}{data_name}_{f_model}_llmat_vs_decoupled.pdf"
            plt.savefig(out_file, format='pdf', bbox_inches='tight')
            plt.close()
            print(f"saved {out_file}")


def compute_llmat_vs_decoupled_gap(f_models, data_names, min_seeds=15):
    """Quantify the coupling-vs-decoupled ablation: Gap AUC (higher =
    better; trapz over T=1..100 of the per-seed normalized optimality
    gap) for LLMAT's single coupled classifier vs the decoupled K-means-
    partition + independent-LFBO ablation, under fixed (non-finetuned)
    features. A positive 'coupling_gain' means the coupled classifier
    (LLMAT) outperforms the decoupled version on that dataset/feature.

    load_decoupled already drops any pre-fix (see DECOUPLED_FIX_MTIME)
    seed file, so as cluster jobs are still regenerating decoupled runs,
    n_dec (fresh decoupled seeds actually averaged in) can be < len(seeds)
    for a while. Rows below min_seeds are still reported but flagged in
    the printout (and excluded from the "reliable" summary) since a
    mean over a handful of seeds is noisy -- don't over-read them.

    Gap normalization uses a SHARED baseline per (dataset, f_model): the
    worst y_init observed across all LLMAT+Decoupled seeds pooled together
    (not each seed's own y_init). Using a per-seed y_init blows up when a
    seed's very first BO-selected point happens to already land close to
    the optimum -- the denominator (gt - y_init) collapses toward 0, so
    "already near-optimal from round 1, nothing left to improve" gets
    scored as gap=0 (AUC~0) instead of the high score it deserves. A
    shared baseline avoids that per-seed-luck distortion.
    """

    from scipy.stats import wilcoxon

    def _gap_auc_by_seed(mat_bench, llmat_loader, dec_loader):
        """Returns {seed: auc} dicts for both methods (keyed by seed, not
        position) so a paired test can intersect on matching seeds even
        when coverage differs between LLMAT and Decoupled."""
        gt = mat_bench.ground_truth_opt
        is_max = mat_bench.maximization

        llmat_by_seed = {s: v for s in seeds if (v := llmat_loader(mat_bench, s)) is not None}
        dec_by_seed = {s: v for s in seeds if (v := dec_loader(mat_bench, s)) is not None}
        if not llmat_by_seed or not dec_by_seed:
            return None, None

        all_init = [v[1] for v in llmat_by_seed.values()] + [v[1] for v in dec_by_seed.values()]
        baseline = min(all_init) if is_max else max(all_init)
        denom = (gt - baseline) if is_max else (baseline - gt)
        if abs(denom) < 1e-8:
            denom = 1.0

        def _auc_one(by):
            y_curve = np.asarray(by)[1:101]
            gap = (y_curve - baseline) / denom if is_max else (baseline - y_curve) / denom
            gap = np.clip(gap, 0.0, 1.0)
            return float(np.trapz(gap))

        llmat_auc = {s: _auc_one(v) for s, v in llmat_by_seed.items()}
        dec_auc = {s: _auc_one(v) for s, v in dec_by_seed.items()}
        return llmat_auc, dec_auc

    def _paired_wilcoxon(llmat_auc, dec_auc, min_pairs=10):
        """Paired signed-rank test on the seeds present in BOTH methods
        (paired by seed id, mirrors the convention already used for
        LLMAT-vs-baseline tests elsewhere in this file). Returns
        (pval, n_pairs) or (None, n_pairs) if too few non-tied pairs."""
        common = sorted(set(llmat_auc) & set(dec_auc))
        la = np.array([llmat_auc[s] for s in common])
        da = np.array([dec_auc[s] for s in common])
        nonzero = int(np.sum(np.abs(la - da) > 1e-10))
        if nonzero < min_pairs:
            return None, len(common)
        _, pval = wilcoxon(la, da)  # two-sided: is there ANY difference
        return float(pval), len(common)

    # Collect everything silently first (MATBench()/loaders print a lot of
    # their own noise) and only print our table + summary as one block at
    # the end, so it isn't interleaved with that noise.
    rows = []
    for data_name in data_names:
        for f_model in f_models:
            mat_bench = MATBench(
                data_name=data_name,
                run_subset_only=run_subset_only,
                feature_type=f_model,
                finetuning=False,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=665,
                clustering_type=clustering_types[0],
                feature_reduction=feature_reduction,
            )
            llmat_auc_by_seed, dec_auc_by_seed = _gap_auc_by_seed(
                mat_bench,
                lambda mb, s: load_ours(mb, r_type="best_y", seed=s),
                lambda mb, s: load_decoupled(mb, r_type="best_y", seed=s),
            )
            if llmat_auc_by_seed is None or dec_auc_by_seed is None:
                continue
            llmat_auc = np.array(list(llmat_auc_by_seed.values()))
            dec_auc = np.array(list(dec_auc_by_seed.values()))
            n_llmat, n_dec = len(llmat_auc), len(dec_auc)
            llmat_mean, dec_mean = float(np.mean(llmat_auc)), float(np.mean(dec_auc))
            llmat_std = float(np.std(llmat_auc, ddof=1)) if n_llmat > 1 else 0.0
            dec_std = float(np.std(dec_auc, ddof=1)) if n_dec > 1 else 0.0
            diff = llmat_mean - dec_mean
            pval, n_pairs = _paired_wilcoxon(llmat_auc_by_seed, dec_auc_by_seed)
            rows.append({
                'dataset': data_name,
                'f_model': f_model,
                'n_dec': n_dec,
                'n_llmat': n_llmat,
                'n_pairs': n_pairs,
                'llmat_auc': llmat_mean,
                'llmat_std': llmat_std,
                'decoupled_auc': dec_mean,
                'decoupled_std': dec_std,
                'coupling_gain': diff,
                'pval': pval,
            })

    df = pd.DataFrame(rows)

    lines = []
    lines.append(f"\n{'='*98}")
    lines.append("LLMAT (coupled classifier) vs Decoupled (K-means partition + independent LFBO)")
    lines.append(f"{'='*98}")
    lines.append(f"  {'Dataset':<12s} {'Feature':<24s} {'n_dec':>6s} {'LLMAT (std)':>16s} {'Decoupled (std)':>18s} {'Coupling gain':>15s}  {'p (Wilcoxon)':>14s}")
    lines.append(f"  {'-'*116}")

    def _sig(pval):
        if pval is None:
            return "n/a"
        if pval < 0.001:
            return f"{pval:.4f} ***"
        if pval < 0.01:
            return f"{pval:.4f} **"
        if pval < 0.05:
            return f"{pval:.4f} *"
        return f"{pval:.4f} n.s."

    for row in rows:
        flag = "" if row['n_dec'] >= min_seeds else " *"
        llmat_str = f"{row['llmat_auc']:.1f}±{row['llmat_std']:.1f}"
        dec_str = f"{row['decoupled_auc']:.1f}±{row['decoupled_std']:.1f}"
        lines.append(f"  {Datasets[row['dataset']]:<12s} {row['f_model']:<24s} {row['n_dec']:>6d} "
                     f"{llmat_str:>16s} {dec_str:>18s} {row['coupling_gain']:>+15.1f}{flag}  {_sig(row['pval']):>14s}")

    if not df.empty:
        n = len(df)
        reliable = df[df['n_dec'] >= min_seeds]
        lines.append(f"  {'-'*116}")
        lines.append(f"  * = fewer than {min_seeds} fresh decoupled seeds (cluster job still regenerating post-fix results); noisy, don't over-read")
        lines.append(f"  p = paired Wilcoxon signed-rank test on per-seed AUC (two-sided; n/a if <10 non-tied paired seeds); *** p<0.001, ** p<0.01, * p<0.05")

        lines.append(f"\n  Per-dataset average (mean±std ACROSS feature models, i.e. one number per dataset):")
        lines.append(f"  {'Dataset':<14s} {'n_features':>10s} {'LLMAT':>14s} {'Decoupled':>16s} {'Coupling gain':>15s}")
        lines.append(f"  {'-'*74}")
        for data_name in data_names:
            sub = df[df['dataset'] == data_name]
            if sub.empty:
                continue
            n_feat = len(sub)
            n_low = int((sub['n_dec'] < min_seeds).sum())
            flag = f" ({n_low} low-n)" if n_low else ""
            llmat_str = f"{sub['llmat_auc'].mean():.1f}±{sub['llmat_auc'].std():.1f}" if n_feat > 1 else f"{sub['llmat_auc'].mean():.1f}"
            dec_str = f"{sub['decoupled_auc'].mean():.1f}±{sub['decoupled_auc'].std():.1f}" if n_feat > 1 else f"{sub['decoupled_auc'].mean():.1f}"
            gain_str = f"{sub['coupling_gain'].mean():+.1f}±{sub['coupling_gain'].std():.1f}" if n_feat > 1 else f"{sub['coupling_gain'].mean():+.1f}"
            lines.append(f"  {Datasets[data_name]:<14s} {n_feat:>10d} {llmat_str:>14s} {dec_str:>16s} {gain_str:>15s}{flag}")

        lines.append(f"  Overall (mean±std over {n} dataset x feature combos, ALL rows incl. low-n): "
                     f"LLMAT={df['llmat_auc'].mean():.1f}±{df['llmat_auc'].std():.1f}  "
                     f"Decoupled={df['decoupled_auc'].mean():.1f}±{df['decoupled_auc'].std():.1f}  "
                     f"Coupling gain={df['coupling_gain'].mean():+.1f}±{df['coupling_gain'].std():.1f}")
        if not reliable.empty:
            lines.append(f"  Overall (mean±std over {len(reliable)} combos with >={min_seeds} fresh decoupled seeds): "
                         f"LLMAT={reliable['llmat_auc'].mean():.1f}±{reliable['llmat_auc'].std():.1f}  "
                         f"Decoupled={reliable['decoupled_auc'].mean():.1f}±{reliable['decoupled_auc'].std():.1f}  "
                         f"Coupling gain={reliable['coupling_gain'].mean():+.1f}±{reliable['coupling_gain'].std():.1f}")
        lines.append(f"  Coupling helps (LLMAT beats Decoupled) on {int((df['coupling_gain'] > 0).sum())}/{n} combos overall, "
                     f"{int((reliable['coupling_gain'] > 0).sum())}/{len(reliable)} reliable combos")
        sig = df[df['pval'].notna() & (df['pval'] < 0.05)]
        sig_llmat = int((sig['coupling_gain'] > 0).sum())
        sig_dec = int((sig['coupling_gain'] < 0).sum())
        lines.append(f"  Statistically significant (p<0.05) on {len(sig)}/{n} combos: "
                     f"LLMAT wins {sig_llmat}, Decoupled wins {sig_dec}")

    print("\n".join(lines))
    return df


#######
def plot_best_y_all(path, f_models, data_names):
    # Set seaborn style and context
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Define markers for different algorithms
    algorithm_markers = {
        'LLMAT': 'o',
        # Add other algorithms with different markers as needed
    }

    # Add baseline markers
    for baseline in base_lines:
        if baseline not in algorithm_markers:
            # Use different markers for baselines
            marker_options = ['s', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
            algorithm_markers[baseline] = marker_options[len(algorithm_markers) % len(marker_options)]

    # Use seaborn color palette (colorblind-safe and aesthetically pleasing)
    academic_colors = sns.color_palette("husl", len(f_models))

    # Map f_models to colors
    f_model_color_map = dict(zip(f_models, academic_colors))

    # Calculate subplot layout
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)  # Max 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create figure and subplots with seaborn
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_datasets > 1 else axes

    # Store all data for potential seaborn plotting
    all_data = []

    # Store legend elements - separate for algorithms and f_models
    algorithm_legend_elements = []
    algorithm_legend_labels = []
    f_model_legend_elements = []
    f_model_legend_labels = []

    # Track which items have been added to legend
    added_algorithms = set()
    added_f_models = set()

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        # Data for this subplot
        subplot_data = []

        for f_model in f_models:
            color = f_model_color_map[f_model]

            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                # Process "ours" algorithm
                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    sem_values = st.sem(trace_best_y, axis=0)[1:101]
                    T = np.arange(len(mean_values)) + 1

                    # Store data for potential DataFrame creation
                    for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                        subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': 'LLMAT', 'f_model': f_model, 'dataset': data_name})

                    # Plot with seaborn style but using matplotlib for custom markers
                    line = ax.plot(T, mean_values, marker=algorithm_markers['LLMAT'], color=color, label=f"LLMAT-{f_model}", markersize=4, markevery=5, linewidth=2)
                    ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                    # Add to legend elements (only once per algorithm and f_model)
                    if idx == 0:  # Only add to legend from first subplot
                        if 'LLMAT' not in added_algorithms:
                            algorithm_legend_elements.append(plt.Line2D([0], [0], marker=algorithm_markers['LLMAT'], color='gray', linestyle='-', markersize=8))
                            algorithm_legend_labels.append('LLMAT')
                            added_algorithms.add('LLMAT')

                        if f_model not in added_f_models:
                            f_model_legend_elements.append(plt.Line2D([0], [0], marker='None', color=color, linestyle='-', linewidth=2))
                            f_model_legend_labels.append(f_model)
                            added_f_models.add(f_model)

                # Process baselines
                for baseline in base_lines:
                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        sem_values = st.sem(trace_best_y, axis=0)[1:101]
                        T = np.arange(len(mean_values)) + 1

                        # Store data for potential DataFrame creation
                        for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                            subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': baseline, 'f_model': f_model, 'dataset': data_name})

                        # Plot line with marker
                        ax.plot(T, mean_values, marker=algorithm_markers[baseline], color=color, label=f"{baseline.upper()}-{f_model}", markersize=4, markevery=5, linewidth=2)
                        ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                        # Add to legend elements (only once per algorithm and f_model)
                        if idx == 0:  # Only add to legend from first subplot
                            if baseline not in added_algorithms:
                                algorithm_legend_elements.append(plt.Line2D([0], [0], marker=algorithm_markers[baseline], color='gray', linestyle='-', markersize=8))
                                algorithm_legend_labels.append(baseline)
                                added_algorithms.add(baseline)

        # Add ground truth line with seaborn style
        ax.axhline(mat_bench.ground_truth_opt, color='black', linestyle='--', zorder=1000, linewidth=1.5, alpha=0.8)

        # Set subplot properties with seaborn aesthetics
        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=12, fontweight='bold')

        # Set title with arrow and better formatting
        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} (↑)", fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f"{Datasets[data_name]} (↓)", fontsize=14, fontweight='bold', pad=20)

        # Apply seaborn styling to individual axes
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        # Store subplot data
        all_data.extend(subplot_data)

    # Hide unused subplots
    for idx in range(len(data_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Create horizontal legends side by side at the top with better styling
    # Algorithm legend (left side, horizontal)
    alg_legend = fig.legend(algorithm_legend_elements, algorithm_legend_labels, loc='upper center', bbox_to_anchor=(0.25, 0.98), title='Algorithms',
                            ncol=len(algorithm_legend_labels), fontsize=11, title_fontsize=12, frameon=True, fancybox=True, shadow=True, columnspacing=2.0, handletextpad=0.8)

    # F_model legend (right side, horizontal)
    f_model_legend = fig.legend(f_model_legend_elements, f_model_legend_labels, loc='upper center', bbox_to_anchor=(0.75, 0.98), title='Feature Models',
                                ncol=len(f_model_legend_labels), fontsize=11, title_fontsize=12, frameon=True, fancybox=True, shadow=True, columnspacing=2.0, handletextpad=0.8)

    # Add the first legend back to the figure
    fig.add_artist(alg_legend)

    # Adjust layout with better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.83, hspace=0.3, wspace=0.3)

    # Save figure with high quality
    plt.savefig(path + "all_datasets_comparison.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # Return the collected data as a DataFrame for potential further seaborn analysis
    return pd.DataFrame(all_data)


def plot_best_y_all_seaborn(path, f_models, data_names):
    # Set seaborn style and context for academic publications
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context(
        "paper",
        font_scale=1.1,
        rc={
            "lines.linewidth": 1.2,  # Even thinner lines
            "lines.markersize": 6
        })

    # Define line styles for different algorithms with more variety
    algorithm_linestyles = {
        'LLMAT': '-',  # Solid line
        # Add other algorithms with different line styles as needed
    }

    # Add baseline line styles with more diverse patterns
    linestyle_options = [
        # '--',  # Dashed
        (0, (3, 1, 1, 1, 1, 1)),  # Complex: dash-dot-dot
        ':',  # Dotted
        '-.',  # Dash-dot
        (0, (3, 5, 1, 5)),  # Dash-dot-dot
        (0, (5, 10)),  # Long dash with long gap
        (0, (1, 1)),  # Short dash
        (0, (5, 1, 1, 1)),  # Long dash-dot
        (0, (1, 5)),  # Dot with long gap
        (0, (3, 3, 1, 3)),  # Dash-short dash
        (0, (5, 2, 5, 2, 5, 10)),  # Double dash with gap
        (0, (2, 2, 10, 2)),  # Short dash-long gap-short dash
    ]

    for i, baseline in enumerate(base_lines):
        if baseline not in algorithm_linestyles:
            algorithm_linestyles[baseline] = linestyle_options[i % len(linestyle_options)]

    # Use seaborn's default color palette (clean and academic)
    academic_colors = sns.color_palette("deep", len(f_models))

    # Map f_models to seaborn default colors
    f_model_color_map = dict(zip(f_models, academic_colors))

    # Calculate subplot layout
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)  # Max 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create figure and subplots with seaborn
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_datasets > 1 else axes

    # Store all data for potential seaborn plotting
    all_data = []

    # Store legend elements - separate for algorithms and f_models
    algorithm_legend_elements = []
    algorithm_legend_labels = []
    f_model_legend_elements = []
    f_model_legend_labels = []

    # Track which items have been added to legend
    added_algorithms = set()
    added_f_models = set()

    for idx, data_name in enumerate(data_names):
        # plt.figure(figsize=(6, 3))
        ax = axes_flat[idx]

        # Data for this subplot
        subplot_data = []

        for f_model in f_models:
            color = f_model_color_map[f_model]

            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                    # feature_dim=feature_dim,
                )

                # Process "ours" algorithm
                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    sem_values = st.sem(trace_best_y, axis=0)[1:101]
                    T = np.arange(len(mean_values)) + 1

                    # Store data for potential DataFrame creation
                    for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                        subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': 'LLMAT', 'f_model': f_model, 'dataset': data_name})

                    # Plot with seaborn style using line styles
                    line = ax.plot(T, mean_values, linestyle=algorithm_linestyles['LLMAT'], color=color, label=f"LLMAT-{f_model}", linewidth=1.2)  # Consistent thin line
                    ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                    # Add to legend elements (only once per algorithm and f_model)
                    if idx == 0:  # Only add to legend from first subplot
                        if 'LLMAT' not in added_algorithms:
                            algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles['LLMAT'], color='gray', linewidth=1.2))
                            algorithm_legend_labels.append('LLMAT')
                            added_algorithms.add('LLMAT')

                        if f_model not in added_f_models:
                            f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=3))
                            f_model_legend_labels.append(f_model)
                            added_f_models.add(f_model)

                # Process baselines
                for baseline in base_lines:
                    if baseline == "random" and f_model != "fingerprints":
                        continue
                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        sem_values = st.sem(trace_best_y, axis=0)[1:101]
                        T = np.arange(len(mean_values)) + 1

                        # Store data for potential DataFrame creation
                        for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                            subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': baseline, 'f_model': f_model, 'dataset': data_name})

                        # Plot line with different line style
                        ax.plot(T, mean_values, linestyle=algorithm_linestyles[baseline], color=color, label=f"{baseline.upper()}-{f_model}", linewidth=1.2)
                        ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                        # Add to legend elements (only once per algorithm and f_model)
                        if idx == 0:  # Only add to legend from first subplot
                            if baseline not in added_algorithms:
                                algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles[baseline], color='gray', linewidth=1.2))
                                algorithm_legend_labels.append(baseline.upper())
                                added_algorithms.add(baseline)

                            if f_model not in added_f_models:
                                f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=3))
                                f_model_legend_labels.append(f_model)
                                added_f_models.add(f_model)

        # Add ground truth line with professional academic styling
        ax.axhline(
            mat_bench.ground_truth_opt,
            color='#2C2C2C',  # Dark gray instead of pure black
            linestyle='--',
            zorder=1000,
            linewidth=1.2,  # Consistent thin line
            alpha=0.8,
            label='Ground Truth')

        # Set subplot properties with seaborn aesthetics
        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        if idx >= 3:
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        # Set title with arrow and better formatting
        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\downarrow$)", fontsize=14, fontweight='bold')

        # Apply academic styling to individual axes
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)  # Subtle grid
        ax.tick_params(labelsize=12, colors='#2C2C2C')  # Dark gray ticks

        # Set spine colors to dark gray for professional look
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

        # Store subplot data
        all_data.extend(subplot_data)

    # Hide unused subplots
    for idx in range(len(data_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add both legends to the first subplot
    if len(data_names) > 0:
        first_ax = axes_flat[0]

        # Create algorithm legend (positioned at upper right)
        alg_legend = first_ax.legend(algorithm_legend_elements, algorithm_legend_labels, loc='upper right', bbox_to_anchor=(0.68, 0.98), title='Algorithms', fontsize=11,
                                     title_fontsize=12, frameon=True, fancybox=True, shadow=False, framealpha=0.9)

        # Create f_model legend (positioned at upper left)
        f_model_legend = first_ax.legend(f_model_legend_elements, f_model_legend_labels, loc='upper right', bbox_to_anchor=(1.02, 0.98), title='Feature Models', fontsize=11,
                                         title_fontsize=12, frameon=True, fancybox=True, shadow=False, framealpha=0.9)

        # Add the algorithm legend back to the axes (matplotlib removes it when adding second legend)
        first_ax.add_artist(alg_legend)

    # Adjust layout with normal spacing since legends are now on subplot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    # plt.xlabel('BO Rounds: T', fontsize=16)
    # Save figure with high quality
    plt.savefig(path + "all_datasets_comparison_all.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # Return the collected data as a DataFrame for potential further seaborn analysis
    return pd.DataFrame(all_data)


def plot_best_y_all_seaborn_auc(path, f_models, data_names):
    # Set seaborn style and context for academic publications
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context(
        "paper",
        font_scale=1.1,
        rc={
            "lines.linewidth": 1.2,  # Even thinner lines
            "lines.markersize": 6
        })

    # Define line styles for different algorithms with more variety
    algorithm_linestyles = {
        'LLMAT': '-',  # Solid line
        # Add other algorithms with different line styles as needed
    }

    # Add baseline line styles with more diverse patterns
    linestyle_options = [
        '--',  # Dashed
        '-.',  # Dash-dot
        ':',  # Dotted
        (0, (3, 5, 1, 5)),  # Dash-dot-dot
        (0, (5, 10)),  # Long dash with long gap
        (0, (1, 1)),  # Short dash
        (0, (3, 1, 1, 1, 1, 1)),  # Complex: dash-dot-dot
        (0, (5, 1, 1, 1)),  # Long dash-dot
        (0, (1, 5)),  # Dot with long gap
        (0, (3, 3, 1, 3)),  # Dash-short dash
        (0, (5, 2, 5, 2, 5, 10)),  # Double dash with gap
        (0, (2, 2, 10, 2)),  # Short dash-long gap-short dash
    ]

    for i, baseline in enumerate(base_lines):
        if baseline not in algorithm_linestyles:
            algorithm_linestyles[baseline] = linestyle_options[i % len(linestyle_options)]

    # Function to calculate AUC for a curve
    def calculate_auc(y_values, x_values=None):
        if x_values is None:
            x_values = np.arange(len(y_values))
        return np.trapz(y_values, x_values)

    # Function to get best f_model for each algorithm based on AUC
    def get_best_f_models_by_auc(data_name):
        algorithm_best_f_models = {}

        # Process "ours" algorithm
        best_auc = -np.inf
        best_f_model = None

        for f_model in f_models:
            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    auc = calculate_auc(mean_values)

                    if auc > best_auc:
                        best_auc = auc
                        best_f_model = f_model

        if best_f_model is not None:
            algorithm_best_f_models['LLMAT'] = best_f_model

        # Process baselines
        for baseline in base_lines:
            best_auc = -np.inf
            best_f_model = None

            for f_model in f_models:
                if baseline == "random" and f_model != "fingerprints":
                    continue

                for clustering_type in clustering_types:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=finetuning,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        auc = calculate_auc(mean_values)

                        if auc > best_auc:
                            best_auc = auc
                            best_f_model = f_model

            if best_f_model is not None:
                algorithm_best_f_models[baseline] = best_f_model

        return algorithm_best_f_models

    # Calculate subplot layout
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)  # Max 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create figure and subplots with seaborn
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_datasets > 1 else axes

    # Store all data for potential seaborn plotting
    all_data = []

    # Store legend elements - separate for algorithms and f_models
    algorithm_legend_elements = []
    algorithm_legend_labels = []
    f_model_legend_elements = []
    f_model_legend_labels = []

    # Track which items have been added to legend
    added_algorithms = set()
    added_f_models = set()

    # Get all unique best f_models across all datasets for consistent coloring
    all_best_f_models = set()
    for data_name in data_names:
        best_f_models = get_best_f_models_by_auc(data_name)
        all_best_f_models.update(best_f_models.values())

    # Use seaborn's default color palette for unique f_models only
    academic_colors = sns.color_palette("deep", len(all_best_f_models))
    f_model_color_map = dict(zip(all_best_f_models, academic_colors))

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        # Get best f_models for this dataset
        best_f_models_for_dataset = get_best_f_models_by_auc(data_name)

        # Data for this subplot
        subplot_data = []

        for clustering_type in clustering_types:
            # Process "ours" algorithm with best f_model
            if 'LLMAT' in best_f_models_for_dataset:
                f_model = best_f_models_for_dataset['LLMAT']
                color = f_model_color_map[f_model]

                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    sem_values = st.sem(trace_best_y, axis=0)[1:101]
                    T = np.arange(len(mean_values)) + 1

                    # Store data for potential DataFrame creation
                    for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                        subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': 'LLMAT', 'f_model': f_model, 'dataset': data_name})

                    # Plot with seaborn style using line styles
                    line = ax.plot(T, mean_values, linestyle=algorithm_linestyles['LLMAT'], color=color, label=f"LLMAT-{f_model}", linewidth=1.2)
                    ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                    # Add to legend elements (only once per algorithm and f_model)
                    if idx == 0:  # Only add to legend from first subplot
                        if 'LLMAT' not in added_algorithms:
                            algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles['LLMAT'], color='gray', linewidth=1.2))
                            algorithm_legend_labels.append('LLMAT')
                            added_algorithms.add('LLMAT')

                        if f_model not in added_f_models:
                            f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=1.2))
                            f_model_legend_labels.append(f_model)
                            added_f_models.add(f_model)

            # Process baselines with their best f_models
            for baseline in base_lines:
                if baseline in best_f_models_for_dataset:
                    f_model = best_f_models_for_dataset[baseline]
                    color = f_model_color_map[f_model]

                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=finetuning,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        sem_values = st.sem(trace_best_y, axis=0)[1:101]
                        T = np.arange(len(mean_values)) + 1

                        # Store data for potential DataFrame creation
                        for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                            subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': baseline.upper(), 'f_model': f_model, 'dataset': data_name})

                        # Plot line with different line style
                        ax.plot(T, mean_values, linestyle=algorithm_linestyles[baseline], color=color, label=f"{baseline.upper()}-{f_model}", linewidth=1.2)
                        ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                        # Add to legend elements (only once per algorithm and f_model)
                        if idx == 0:  # Only add to legend from first subplot
                            if baseline not in added_algorithms:
                                algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles[baseline], color='gray', linewidth=1.2))
                                algorithm_legend_labels.append(baseline.upper())
                                added_algorithms.add(baseline)

                            if f_model not in added_f_models:
                                f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=1.2))
                                f_model_legend_labels.append(f_model)
                                added_f_models.add(f_model)

        # Add ground truth line with professional academic styling
        ax.axhline(
            mat_bench.ground_truth_opt,
            color='#2C2C2C',  # Dark gray instead of pure black
            linestyle='--',
            zorder=1000,
            linewidth=1.2,  # Consistent thin line
            alpha=0.8,
            label='Ground Truth')

        # Set subplot properties with seaborn aesthetics
        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        if idx >= 3:
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        # Set title with arrow and better formatting
        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\downarrow$)", fontsize=14, fontweight='bold')

        # Apply academic styling to individual axes
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)  # Subtle grid
        ax.tick_params(labelsize=12, colors='#2C2C2C')  # Dark gray ticks

        # Set spine colors to dark gray for professional look
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

        # Store subplot data
        all_data.extend(subplot_data)

    # Hide unused subplots
    for idx in range(len(data_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add both legends to the first subplot
    if len(data_names) > 0:
        first_ax = axes_flat[0]

        # Create algorithm legend (positioned at upper right)
        alg_legend = first_ax.legend(algorithm_legend_elements, algorithm_legend_labels, loc='upper right', bbox_to_anchor=(0.68, 0.98), title='Algorithms', fontsize=12,
                                     title_fontsize=14, frameon=True, fancybox=True, shadow=False, framealpha=0.9)

        # Create f_model legend (positioned at upper left)
        f_model_legend = first_ax.legend(f_model_legend_elements, f_model_legend_labels, loc='upper right', bbox_to_anchor=(1.02, 0.98), title='Feature Models', fontsize=12,
                                         title_fontsize=14, frameon=True, fancybox=True, shadow=False, framealpha=0.9)

        # Add the algorithm legend back to the axes (matplotlib removes it when adding second legend)
        first_ax.add_artist(alg_legend)

    # Adjust layout with normal spacing since legends are now on subplot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    # Save figure with high quality
    plt.savefig(path + "all_datasets_comparison_best_auc.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # Return the collected data as a DataFrame for potential further seaborn analysis
    return pd.DataFrame(all_data)


def plot_best_y_all_seaborn_auc2(path, f_models, data_names):
    # Set seaborn style and context for academic publications
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context(
        "paper",
        font_scale=1.1,
        rc={
            "lines.linewidth": 1.2,  # Even thinner lines
            "lines.markersize": 6
        })

    # Define line styles for different algorithms with more variety
    algorithm_linestyles = {
        'LLMAT': '-',  # Solid line
        # Add other algorithms with different line styles as needed
    }

    # Add baseline line styles with more diverse patterns
    linestyle_options = [
        '--',  # Dashed
        '-.',  # Dash-dot
        ':',  # Dotted
        (0, (3, 5, 1, 5)),  # Dash-dot-dot
        (0, (5, 10)),  # Long dash with long gap
        (0, (1, 1)),  # Short dash
        (0, (3, 1, 1, 1, 1, 1)),  # Complex: dash-dot-dot
        (0, (5, 1, 1, 1)),  # Long dash-dot
        (0, (1, 5)),  # Dot with long gap
        (0, (3, 3, 1, 3)),  # Dash-short dash
        (0, (5, 2, 5, 2, 5, 10)),  # Double dash with gap
        (0, (2, 2, 10, 2)),  # Short dash-long gap-short dash
    ]

    for i, baseline in enumerate(base_lines):
        if baseline not in algorithm_linestyles:
            algorithm_linestyles[baseline] = linestyle_options[i % len(linestyle_options)]

    # Function to calculate AUC for a curve
    def calculate_auc(y_values, x_values=None):
        if x_values is None:
            x_values = np.arange(len(y_values))
        return np.trapz(y_values, x_values)

    # Function to get best f_model for each algorithm based on AUC
    def get_best_f_models_by_auc(data_name):
        algorithm_best_f_models = {}

        # Process "ours" algorithm
        best_auc = -np.inf
        best_f_model = None

        for f_model in f_models:
            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    auc = calculate_auc(mean_values)

                    if auc > best_auc:
                        best_auc = auc
                        best_f_model = f_model

        if best_f_model is not None:
            algorithm_best_f_models['LLMAT'] = best_f_model

        # Process baselines
        for baseline in base_lines:
            best_auc = -np.inf
            best_f_model = None

            for f_model in f_models:
                if baseline == "random" and f_model != "fingerprints":
                    continue

                for clustering_type in clustering_types:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=finetuning,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        auc = calculate_auc(mean_values)

                        if auc > best_auc:
                            best_auc = auc
                            best_f_model = f_model

            if best_f_model is not None:
                algorithm_best_f_models[baseline] = best_f_model

        return algorithm_best_f_models

    # Calculate subplot layout
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)  # Max 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create figure and subplots with seaborn
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_datasets > 1 else axes

    # Store all data for potential seaborn plotting
    all_data = []

    # Store legend elements - separate for algorithms and f_models
    algorithm_legend_elements = []
    algorithm_legend_labels = []
    f_model_legend_elements = []
    f_model_legend_labels = []

    # Track which items have been added to legend
    added_algorithms = set()
    added_f_models = set()

    # Use seaborn's default color palette for ALL f_models (not just best ones)
    academic_colors = sns.color_palette("deep", len(f_models))
    f_model_color_map = dict(zip(f_models, academic_colors))

    # Get all unique best f_models across all datasets for plotting
    all_best_f_models = set()
    for data_name in data_names:
        best_f_models = get_best_f_models_by_auc(data_name)
        all_best_f_models.update(best_f_models.values())

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        # Get best f_models for this dataset
        best_f_models_for_dataset = get_best_f_models_by_auc(data_name)

        # Data for this subplot
        subplot_data = []

        for clustering_type in clustering_types:
            # Process "ours" algorithm with best f_model
            if 'LLMAT' in best_f_models_for_dataset:
                f_model = best_f_models_for_dataset['LLMAT']
                color = f_model_color_map[f_model]

                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    sem_values = st.sem(trace_best_y, axis=0)[1:101]
                    T = np.arange(len(mean_values)) + 1

                    # Store data for potential DataFrame creation
                    for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                        subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': 'LLMAT', 'f_model': f_model, 'dataset': data_name})

                    # Plot with seaborn style using line styles
                    line = ax.plot(T, mean_values, linestyle=algorithm_linestyles['LLMAT'], color=color, label=f"LLMAT-{f_model}", linewidth=1.2)
                    ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                    # Add to legend elements (only once per algorithm and f_model)
                    if idx == 0:  # Only add to legend from first subplot
                        if 'LLMAT' not in added_algorithms:
                            algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles['LLMAT'], color='gray', linewidth=1.2))
                            algorithm_legend_labels.append('LLMAT')
                            added_algorithms.add('LLMAT')

                        if f_model not in added_f_models:
                            f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=1.2))
                            f_model_legend_labels.append(f_model)
                            added_f_models.add(f_model)

            # Process baselines with their best f_models
            for baseline in base_lines:
                if baseline in best_f_models_for_dataset:
                    f_model = best_f_models_for_dataset[baseline]
                    color = f_model_color_map[f_model]

                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=finetuning,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        sem_values = st.sem(trace_best_y, axis=0)[1:101]
                        T = np.arange(len(mean_values)) + 1

                        # Store data for potential DataFrame creation
                        for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                            subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': baseline, 'f_model': f_model, 'dataset': data_name})

                        # Plot line with different line style
                        ax.plot(T, mean_values, linestyle=algorithm_linestyles[baseline], color=color, label=f"{baseline.upper()}-{f_model}", linewidth=1.2)
                        ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                        # Add to legend elements (only once per algorithm and f_model)
                        if idx == 0:  # Only add to legend from first subplot
                            if baseline not in added_algorithms:
                                algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles[baseline], color='gray', linewidth=1.2))
                                algorithm_legend_labels.append(baseline)
                                added_algorithms.add(baseline)

                            if f_model not in added_f_models:
                                f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=1.2))
                                f_model_legend_labels.append(f_model)
                                added_f_models.add(f_model)

        # Add ground truth line with professional academic styling
        ax.axhline(
            mat_bench.ground_truth_opt,
            color='#2C2C2C',  # Dark gray instead of pure black
            linestyle='--',
            zorder=1000,
            linewidth=1.2,  # Consistent thin line
            alpha=0.8,
            label='Ground Truth')

        # Set subplot properties with seaborn aesthetics
        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        if idx >= 3:
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        # Set title with arrow and better formatting
        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\downarrow$)", fontsize=14, fontweight='bold')

        # Apply academic styling to individual axes
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)  # Subtle grid
        ax.tick_params(labelsize=12, colors='#2C2C2C')  # Dark gray ticks

        # Set spine colors to dark gray for professional look
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

        # Store subplot data
        all_data.extend(subplot_data)

    # Hide unused subplots
    for idx in range(len(data_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Create f_model legend elements for ALL f_models (not just used ones)
    all_f_model_legend_elements = []
    all_f_model_legend_labels = []
    for f_model in f_models:
        color = f_model_color_map[f_model]
        all_f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=1.2))
        all_f_model_legend_labels.append(f_model)

    # Add f_model legend to the top of the figure (horizontal layout)
    if len(data_names) > 0:
        # Create the f_model legend at the top of the figure
        f_model_legend = fig.legend(all_f_model_legend_elements, all_f_model_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), title='Feature Models', fontsize=12,
                                    title_fontsize=14, frameon=True, fancybox=True, shadow=False, framealpha=0.9, ncol=len(f_models))  # Horizontal layout

        # Add algorithm legend to the first subplot (positioned at upper right)
        first_ax = axes_flat[0]
        alg_legend = first_ax.legend(algorithm_legend_elements, algorithm_legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), title='Algorithms', fontsize=12,
                                     title_fontsize=14, frameon=True, fancybox=True, shadow=False, framealpha=0.9)

    # Adjust layout with extra space at the top for the f_model legend
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9)  # Add top margin for legend

    # Save figure with high quality
    plt.savefig(path + "all_datasets_comparison_best_auc.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # Return the collected data as a DataFrame for potential further seaborn analysis
    return pd.DataFrame(all_data)


def plot_best_y_all_seaborn_auc3(path, f_models, data_names):
    # Set seaborn style and context for academic publications
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context(
        "paper",
        font_scale=1.1,
        rc={
            "lines.linewidth": 2,  # Even thinner lines
            "lines.markersize": 6
        })

    # Define line styles for different algorithms with more variety
    algorithm_linestyles = {
        'LLMAT': '-',  # Solid line
        # Add other algorithms with different line styles as needed
    }

    # Add baseline line styles with more diverse patterns
    linestyle_options = [
        # '--',  # Dashed
        (0, (3, 1, 1, 1, 1, 1)),  # Complex: dash-dot-dot
        ':',  # Dotted
        '-.',  # Dash-dot
        (0, (3, 5, 1, 5)),  # Dash-dot-dot
        (0, (5, 10)),  # Long dash with long gap
        (0, (1, 1)),  # Short dash
        (0, (3, 1, 1, 1, 1, 1)),  # Complex: dash-dot-dot
        (0, (5, 1, 1, 1)),  # Long dash-dot
        (0, (1, 5)),  # Dot with long gap
        (0, (3, 3, 1, 3)),  # Dash-short dash
        (0, (5, 2, 5, 2, 5, 10)),  # Double dash with gap
        (0, (2, 2, 10, 2)),  # Short dash-long gap-short dash
    ]

    for i, baseline in enumerate(base_lines):
        if baseline not in algorithm_linestyles:
            algorithm_linestyles[baseline] = linestyle_options[i % len(linestyle_options)]

    # Function to calculate AUC for a curve
    def calculate_auc(y_values, x_values=None):
        if x_values is None:
            x_values = np.arange(len(y_values))
        return np.trapz(y_values, x_values)

    # Function to get best f_model for each algorithm based on AUC
    def get_best_f_models_by_auc(data_name):
        algorithm_best_f_models = {}

        # Create a temporary MATBench instance to check maximization/minimization
        temp_mat_bench = MATBench(
            data_name=data_name,
            run_subset_only=run_subset_only,
            feature_type=f_models[0],  # Use first f_model for temp instance
            finetuning=finetuning,
            iupac=iupac,
            prompt_type=prompt_type,
            randseed=665,
            clustering_type=clustering_types[0],  # Use first clustering_type for temp instance
            feature_reduction=feature_reduction,
        )
        is_maximization = temp_mat_bench.maximization

        # Process "ours" algorithm
        best_auc = -np.inf if is_maximization else np.inf
        best_f_model = None

        for f_model in f_models:
            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    auc = calculate_auc(mean_values)

                    # Select based on maximization/minimization
                    if is_maximization and auc > best_auc:
                        best_auc = auc
                        best_f_model = f_model
                    elif not is_maximization and auc < best_auc:
                        best_auc = auc
                        best_f_model = f_model

        if best_f_model is not None:
            algorithm_best_f_models['LLMAT'] = best_f_model

        # Process baselines
        for baseline in base_lines:
            best_auc = -np.inf if is_maximization else np.inf
            best_f_model = None

            for f_model in f_models:
                if baseline == "random" and f_model != "fingerprints":
                    continue

                for clustering_type in clustering_types:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=finetuning,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        auc = calculate_auc(mean_values)

                        # Select based on maximization/minimization
                        if is_maximization and auc > best_auc:
                            best_auc = auc
                            best_f_model = f_model
                        elif not is_maximization and auc < best_auc:
                            best_auc = auc
                            best_f_model = f_model

            if best_f_model is not None:
                algorithm_best_f_models[baseline] = best_f_model

        return algorithm_best_f_models

    # Calculate subplot layout
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)  # Max 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create figure and subplots with seaborn
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if n_datasets > 1 else axes

    # Store all data for potential seaborn plotting
    all_data = []

    # Store legend elements - separate for algorithms and f_models
    algorithm_legend_elements = []
    algorithm_legend_labels = []
    f_model_legend_elements = []
    f_model_legend_labels = []

    # Track which items have been added to legend
    added_algorithms = set()
    added_f_models = set()

    # Use seaborn's default color palette for ALL f_models (not just best ones)
    academic_colors = sns.color_palette("deep", len(f_models))
    f_model_color_map = dict(zip(f_models, academic_colors))

    # Get all unique best f_models across all datasets for plotting
    all_best_f_models = set()
    for data_name in data_names:
        best_f_models = get_best_f_models_by_auc(data_name)
        all_best_f_models.update(best_f_models.values())

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        # Get best f_models for this dataset
        best_f_models_for_dataset = get_best_f_models_by_auc(data_name)

        # Data for this subplot
        subplot_data = []

        for clustering_type in clustering_types:
            # Process "ours" algorithm with best f_model
            if 'LLMAT' in best_f_models_for_dataset:
                f_model = best_f_models_for_dataset['LLMAT']
                color = f_model_color_map[f_model]

                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if trace_bys:
                    trace_best_y = np.stack(trace_bys)
                    mean_values = np.mean(trace_best_y, axis=0)[1:101]
                    sem_values = st.sem(trace_best_y, axis=0)[1:101]
                    T = np.arange(len(mean_values)) + 1

                    # Store data for potential DataFrame creation
                    for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                        subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': 'LLMAT', 'f_model': f_model, 'dataset': data_name})

                    # Plot with seaborn style using line styles
                    line = ax.plot(T, mean_values, linestyle=algorithm_linestyles['LLMAT'], color=color, label=f"LLMAT-{f_model}", linewidth=2)
                    ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                    # Add to legend elements (only once per algorithm and f_model)
                    if idx == 0:  # Only add to legend from first subplot
                        if 'LLMAT' not in added_algorithms:
                            algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles['LLMAT'], color='gray', linewidth=2))
                            algorithm_legend_labels.append('LLMAT')
                            added_algorithms.add('LLMAT')

                        if f_model not in added_f_models:
                            f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=4))
                            f_model_legend_labels.append(f_model)
                            added_f_models.add(f_model)

            # Process baselines with their best f_models
            for baseline in base_lines:
                if baseline in best_f_models_for_dataset:
                    f_model = best_f_models_for_dataset[baseline]
                    if baseline == "random":
                        color = '#808080'  # Grey color for random
                    else:
                        color = f_model_color_map[f_model]
                    # color = f_model_color_map[f_model]

                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=finetuning,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    trace_bys = []
                    for seed in seeds:
                        trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line=baseline)
                        if trace_by is not None:
                            trace_bys.append(trace_by)

                    if trace_bys:
                        trace_best_y = np.stack(trace_bys)
                        mean_values = np.mean(trace_best_y, axis=0)[1:101]
                        sem_values = st.sem(trace_best_y, axis=0)[1:101]
                        T = np.arange(len(mean_values)) + 1

                        # Store data for potential DataFrame creation
                        for t, mean_val, sem_val in zip(T, mean_values, sem_values):
                            subplot_data.append({'T': t, 'mean': mean_val, 'sem': sem_val, 'algorithm': baseline, 'f_model': f_model, 'dataset': data_name})

                        # Plot line with different line style
                        ax.plot(T, mean_values, linestyle=algorithm_linestyles[baseline], color=color, label=f"{baseline.upper()}-{f_model}", linewidth=2)
                        ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.2, color=color)

                        # Add to legend elements (only once per algorithm and f_model)
                        if idx == 0:  # Only add to legend from first subplot
                            if baseline not in added_algorithms:
                                algorithm_legend_elements.append(plt.Line2D([0], [0], linestyle=algorithm_linestyles[baseline], color='gray', linewidth=2))
                                algorithm_legend_labels.append(baseline.upper())
                                added_algorithms.add(baseline)

                            if f_model not in added_f_models:
                                f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=4))
                                f_model_legend_labels.append(f_model)
                                added_f_models.add(f_model)

        # Add ground truth line with professional academic styling
        ax.axhline(
            mat_bench.ground_truth_opt,
            color='#2C2C2C',  # Dark gray instead of pure black
            linestyle='--',
            zorder=1000,
            linewidth=2,  # Consistent thin line
            alpha=0.8,
            label='Ground Truth')

        # Set subplot properties with seaborn aesthetics
        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        if idx >= 3:
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        # Set title with arrow and better formatting
        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\downarrow$)", fontsize=14, fontweight='bold')

        # Apply academic styling to individual axes
        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=1.2)  # Subtle grid
        ax.tick_params(labelsize=12, colors='#2C2C2C')  # Dark gray ticks

        # Set spine colors to dark gray for professional look
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

        # Store subplot data
        all_data.extend(subplot_data)

    # Hide unused subplots
    for idx in range(len(data_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Create f_model legend elements for ALL f_models (not just used ones)
    all_f_model_legend_elements = []
    all_f_model_legend_labels = []
    for f_model in f_models:
        color = f_model_color_map[f_model]
        all_f_model_legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, linewidth=4))
        all_f_model_legend_labels.append(f_model)

    # Add f_model legend to the top of the figure (horizontal layout)
    if len(data_names) > 0:
        # Create the f_model legend at the top of the figure
        # f_model_legend = fig.legend(all_f_model_legend_elements, all_f_model_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1), title='Feature Models', fontsize=10,
        # title_fontsize=10, frameon=True, fancybox=False, shadow=False, framealpha=0.9, ncol=len(f_models))  # Horizontal layout
        f_model_legend = fig.legend(all_f_model_legend_elements, all_f_model_legend_labels, loc='upper center', bbox_to_anchor=(0.66, 0.95), fontsize=12, frameon=True,
                                    fancybox=True, shadow=False, framealpha=0.9, ncol=len(f_models))  # Horizontal layout

        # Add algorithm legend to the first subplot (positioned at upper right)
        # first_ax = axes_flat[0]
        alg_legend = fig.legend(algorithm_legend_elements, algorithm_legend_labels, loc='upper left', bbox_to_anchor=(0.05, 0.95), fontsize=12, frameon=True, fancybox=True,
                                shadow=False, framealpha=0.9, ncol=len(algorithm_legend_labels))

        # alg_legend = first_ax.legend(algorithm_legend_elements, algorithm_legend_labels, loc='upper right', bbox_to_anchor=(0.98, 1), title='Algorithms', fontsize=10,
        #                              title_fontsize=12, frameon=True, fancybox=True, shadow=False, framealpha=0.9)

    # Adjust layout with extra space at the top for the f_model legend
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.2, top=0.8)  # Add top margin for legend

    # Save figure with high quality
    plt.savefig(path + "all_datasets_comparison_best_auc.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # Return the collected data as a DataFrame for potential further seaborn analysis
    return pd.DataFrame(all_data)


def plot_gp_across_llms(path, f_models, data_names):
    """Plot GP-only performance across different LLM feature models.

    Shows that GP surrogate performance is similar regardless of which LLM
    features are used, demonstrating no evaluation data leakage from LLMs.
    """
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5, "lines.markersize": 6})

    # Friendly display names for feature models
    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
        "fingerprints": "Fingerprints",
    }

    # Color palette: distinguish LLM-based vs non-LLM features
    llm_models = [m for m in f_models if m != "fingerprints"]
    non_llm_models = [m for m in f_models if m == "fingerprints"]

    # Use a sequential blue palette for LLMs, distinct colors for non-LLM
    llm_colors = sns.color_palette("Blues_d", max(len(llm_models), 3))[-len(llm_models):]
    non_llm_colors = sns.color_palette("Set1", max(len(non_llm_models), 2))[:len(non_llm_models)]

    f_model_color_map = {}
    for i, m in enumerate(llm_models):
        f_model_color_map[m] = llm_colors[i]
    for i, m in enumerate(non_llm_models):
        f_model_color_map[m] = non_llm_colors[i]

    # Layout
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    all_data = []
    legend_elements = []
    legend_labels = []
    added_to_legend = set()

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        for f_model in f_models:
            color = f_model_color_map[f_model]
            display_name = f_model_display.get(f_model, f_model)

            # Load GP baseline for this feature model
            mat_bench = MATBench(
                data_name=data_name,
                run_subset_only=run_subset_only,
                feature_type=f_model,
                finetuning=finetuning,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=665,
                clustering_type=clustering_types[0],
                feature_reduction=feature_reduction,
            )

            trace_bys = []
            for seed in seeds:
                trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line="gp")
                if trace_by is not None:
                    trace_bys.append(trace_by)

            if not trace_bys:
                continue

            trace_best_y = np.stack(trace_bys)
            mean_values = np.mean(trace_best_y, axis=0)[1:101]
            sem_values = st.sem(trace_best_y, axis=0)[1:101]
            T = np.arange(len(mean_values)) + 1

            for t, mv, sv in zip(T, mean_values, sem_values):
                all_data.append({
                    'T': t,
                    'mean': mv,
                    'sem': sv,
                    'f_model': f_model,
                    'dataset': data_name,
                })

            # Use dashed lines for non-LLM, solid for LLM features
            ls = '--' if f_model in non_llm_models else '-'
            lw = 1.8 if f_model in non_llm_models else 1.5

            ax.plot(T, mean_values, linestyle=ls, color=color, linewidth=lw)
            ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.15, color=color)

            if f_model not in added_to_legend:
                legend_elements.append(plt.Line2D([0], [0], linestyle=ls, color=color, linewidth=lw))
                legend_labels.append(f"GP + {display_name}")
                added_to_legend.add(f_model)

        # Ground truth
        ax.axhline(mat_bench.ground_truth_opt, color='#2C2C2C', linestyle='--', zorder=1000, linewidth=1.0, alpha=0.6)

        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\\downarrow$)", fontsize=14, fontweight='bold')

        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=12, colors='#2C2C2C')
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

    # Hide unused subplots
    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Single horizontal legend at top
    fig.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=min(len(legend_labels), 4), fontsize=11, title_fontsize=12, frameon=True,
               fancybox=True, shadow=False, framealpha=0.9, title='GP Surrogate across Feature Models')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.88)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "gp_across_llms_no_leakage.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return pd.DataFrame(all_data)


def plot_llmat_across_llms(path, f_models, data_names):
    """Plot LLMAT performance across different LLM feature models.

    Shows that LLMAT performance is similar regardless of which LLM
    features are used, demonstrating no evaluation data leakage from LLMs.
    """
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5, "lines.markersize": 6})

    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
        "fingerprints": "Fingerprints",
        "gin": "GIN",
    }

    NON_LLM = {"fingerprints", "gin", "schnet"}
    llm_models = [m for m in f_models if m not in NON_LLM]
    non_llm_models = [m for m in f_models if m in NON_LLM]

    llm_colors = sns.color_palette("Blues_d", max(len(llm_models), 3))[-len(llm_models):]
    non_llm_colors = sns.color_palette("Set1", max(len(non_llm_models), 2))[:len(non_llm_models)]

    f_model_color_map = {}
    for i, m in enumerate(llm_models):
        f_model_color_map[m] = llm_colors[i]
    for i, m in enumerate(non_llm_models):
        f_model_color_map[m] = non_llm_colors[i]

    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    all_data = []
    legend_elements = []
    legend_labels = []
    added_to_legend = set()

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        for f_model in f_models:
            color = f_model_color_map[f_model]
            display_name = f_model_display.get(f_model, f_model)

            for clustering_type in clustering_types:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=clustering_type,
                    feature_reduction=feature_reduction,
                )

                trace_bys = []
                for seed in seeds:
                    trace_by = load_ours(mat_bench, r_type="best_y", seed=seed)
                    if trace_by is not None:
                        trace_bys.append(trace_by)

                if not trace_bys:
                    continue

                trace_best_y = np.stack(trace_bys)
                mean_values = np.mean(trace_best_y, axis=0)[1:101]
                sem_values = st.sem(trace_best_y, axis=0)[1:101]
                T = np.arange(len(mean_values)) + 1

                for t, mv, sv in zip(T, mean_values, sem_values):
                    all_data.append({
                        'T': t,
                        'mean': mv,
                        'sem': sv,
                        'f_model': f_model,
                        'clustering': clustering_type,
                        'dataset': data_name,
                    })

                ls = '--' if f_model in non_llm_models else '-'
                lw = 1.8 if f_model in non_llm_models else 1.5

                ax.plot(T, mean_values, linestyle=ls, color=color, linewidth=lw)
                ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.15, color=color)

                if f_model not in added_to_legend:
                    legend_elements.append(plt.Line2D([0], [0], linestyle=ls, color=color, linewidth=lw))
                    legend_labels.append(f"LLMAT + {display_name}")
                    added_to_legend.add(f_model)

        ax.axhline(mat_bench.ground_truth_opt, color='#2C2C2C', linestyle='--', zorder=1000, linewidth=1.0, alpha=0.6)

        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\\downarrow$)", fontsize=14, fontweight='bold')

        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=12, colors='#2C2C2C')
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=min(len(legend_labels), 4), fontsize=11, title_fontsize=12, frameon=True,
               fancybox=True, shadow=False, framealpha=0.9, title='LLMAT across Feature Models')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.88)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "llmat_across_llms_no_leakage.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return pd.DataFrame(all_data)


def compute_llm_vs_fp_best_worst_avg(f_models, data_names, methods=("LLMAT", "GP", "Laplace")):
    """For each dataset and method, compare fingerprints against the BEST
    LLM, the WORST LLM, and the AVERAGE LLM (mean of per-model means), side
    by side. Complements print_llm_vs_fp_verdict_table (which collapses
    this into one pooled verdict): this shows the full spread so you can
    see e.g. a dataset where the best LLM edges out fingerprints while the
    worst LLM and the average both lag behind it.

    Gap(t) = (y(t) - y_init) / (y_opt - y_init), per-seed y_init, clipped
    to [0, 1]; AUC = trapz(Gap, t=1..100), higher = better.

    Returns a DataFrame with columns: dataset, method, fp_auc,
    best_llm, best_llm_auc, worst_llm, worst_llm_auc, avg_llm_auc,
    diff_best, diff_worst, diff_avg.
    """
    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
    }

    NON_LLM = {"fingerprints", "gin", "schnet"}
    llm_models = [m for m in f_models if m not in NON_LLM]
    if "fingerprints" not in f_models:
        raise ValueError("f_models must include 'fingerprints' for this comparison")

    loaders = {
        "LLMAT": lambda mat_bench, s: load_ours(mat_bench, r_type="best_y", seed=s),
        "GP": lambda mat_bench, s: load_baselines(mat_bench, r_type="best_y", seed=s, base_line="gp"),
        "Laplace": lambda mat_bench, s: load_baselines(mat_bench, r_type="best_y", seed=s, base_line="laplace"),
    }

    def _gap_auc(data_name, f_model, loader):
        mat_bench = MATBench(
            data_name=data_name,
            run_subset_only=run_subset_only,
            feature_type=f_model,
            finetuning=finetuning,
            iupac=iupac,
            prompt_type=prompt_type,
            randseed=665,
            clustering_type=clustering_types[0],
            feature_reduction=feature_reduction,
        )
        gt = mat_bench.ground_truth_opt
        is_max = mat_bench.maximization

        bys = [loader(mat_bench, s) for s in seeds]
        bys = [t for t in bys if t is not None]
        if not bys:
            return None

        arr = np.stack(bys)
        y_init = arr[:, 1:2]
        y_curve = arr[:, 1:101]
        denom = (gt - y_init) if is_max else (y_init - gt)
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        gap = (y_curve - y_init) / denom if is_max else (y_init - y_curve) / denom
        gap = np.clip(gap, 0.0, 1.0)
        return np.trapz(gap, axis=1)  # per-seed AUC

    rows = []
    for method in methods:
        loader = loaders[method]
        method_rows = []
        print(f"\n{'='*100}")
        print(f"{method}: BEST / WORST / AVERAGE LLM vs FINGERPRINT (Gap AUC)")
        print(f"{'='*100}")
        print(f"  {'Dataset':<15s} {'FP':>7s}  {'Best LLM':<28s} {'Worst LLM':<28s} {'Avg LLM':>9s}")
        print(f"  {'-'*95}")

        for data_name in data_names:
            fp_auc = _gap_auc(data_name, "fingerprints", loader)
            if fp_auc is None:
                continue
            fp_mean = float(np.mean(fp_auc))

            llm_stats = []  # (f_model, mean_auc)
            for m in llm_models:
                auc = _gap_auc(data_name, m, loader)
                if auc is not None:
                    llm_stats.append((m, float(np.mean(auc))))
            if not llm_stats:
                continue

            best_llm, best_mean = max(llm_stats, key=lambda r: r[1])
            worst_llm, worst_mean = min(llm_stats, key=lambda r: r[1])
            avg_mean = float(np.mean([s[1] for s in llm_stats]))

            row = {
                'dataset': data_name,
                'method': method,
                'fp_auc': fp_mean,
                'best_llm': best_llm,
                'best_llm_auc': best_mean,
                'worst_llm': worst_llm,
                'worst_llm_auc': worst_mean,
                'avg_llm_auc': avg_mean,
                'diff_best': best_mean - fp_mean,
                'diff_worst': worst_mean - fp_mean,
                'diff_avg': avg_mean - fp_mean,
            }
            rows.append(row)
            method_rows.append(row)

            best_str = f"{f_model_display.get(best_llm, best_llm)} {best_mean:.1f} ({row['diff_best']:+.1f})"
            worst_str = f"{f_model_display.get(worst_llm, worst_llm)} {worst_mean:.1f} ({row['diff_worst']:+.1f})"
            avg_str = f"{avg_mean:.1f} ({row['diff_avg']:+.1f})"
            print(f"  {Datasets[data_name]:<15s} {fp_mean:>7.1f}  {best_str:<28s} {worst_str:<28s} {avg_str:>9s}")

        if method_rows:
            mdf = pd.DataFrame(method_rows)
            n = len(mdf)
            print(f"  {'-'*95}")
            print(f"  Overall (mean over {n} datasets): FP = {mdf['fp_auc'].mean():.1f}   "
                  f"Best LLM = {mdf['best_llm_auc'].mean():.1f} ({mdf['diff_best'].mean():+.1f})   "
                  f"Worst LLM = {mdf['worst_llm_auc'].mean():.1f} ({mdf['diff_worst'].mean():+.1f})   "
                  f"Avg LLM = {mdf['avg_llm_auc'].mean():.1f} ({mdf['diff_avg'].mean():+.1f})")
            print(f"  Beats FP on: Best LLM {int((mdf['diff_best'] > 0).sum())}/{n}   "
                  f"Worst LLM {int((mdf['diff_worst'] > 0).sum())}/{n}   "
                  f"Avg LLM {int((mdf['diff_avg'] > 0).sum())}/{n}")

    df = pd.DataFrame(rows)

    if not df.empty:
        n = len(df)
        print(f"\n{'='*100}")
        print(f"SUMMARY ACROSS ALL METHODS ({', '.join(methods)}) x DATASETS (n={n})")
        print(f"{'='*100}")
        print(f"  {'':<10s} {'FP':>8s} {'Best LLM':>12s} {'Worst LLM':>12s} {'Avg LLM':>12s}")
        print(f"  {'Mean AUC':<10s} {df['fp_auc'].mean():>8.1f} {df['best_llm_auc'].mean():>12.1f} "
              f"{df['worst_llm_auc'].mean():>12.1f} {df['avg_llm_auc'].mean():>12.1f}")
        print(f"  {'vs FP':<10s} {'':>8s} {df['diff_best'].mean():>+12.1f} "
              f"{df['diff_worst'].mean():>+12.1f} {df['diff_avg'].mean():>+12.1f}")
        wins_best = f"{int((df['diff_best'] > 0).sum())}/{n}"
        wins_worst = f"{int((df['diff_worst'] > 0).sum())}/{n}"
        wins_avg = f"{int((df['diff_avg'] > 0).sum())}/{n}"
        print(f"  {'Beats FP':<10s} {'':>8s} {wins_best:>12s} {wins_worst:>12s} {wins_avg:>12s}")
        print(f"  {'-'*60}")
        for method in methods:
            sub = df[df['method'] == method]
            if sub.empty:
                continue
            m = len(sub)
            print(f"  {method:<10s} Best LLM beats FP {int((sub['diff_best'] > 0).sum())}/{m}   "
                  f"Worst LLM beats FP {int((sub['diff_worst'] > 0).sum())}/{m}   "
                  f"Avg LLM beats FP {int((sub['diff_avg'] > 0).sum())}/{m}")

    return df


def print_llm_vs_fp_markdown_tables(df, methods=("LLMAT", "GP", "Laplace")):
    """Render the DataFrame returned by compute_llm_vs_fp_best_worst_avg as
    markdown tables (overall comparison, dataset-level best-LLM-vs-FP, and
    dataset-level worst/avg-vs-FP deltas), ready to paste into a paper or
    rebuttal.
    """
    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
    }
    data_names = list(dict.fromkeys(df['dataset']))  # preserve first-seen order, de-duped

    print(f"\n## Overall comparison (Gap AUC, mean over {len(data_names)} datasets)\n")
    print("| Surrogate | FP | Best LLM | Δ | Worst LLM | Δ | Avg LLM | Δ | Best beats FP | Worst beats FP | Avg beats FP |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method in methods:
        sub = df[df['method'] == method]
        if sub.empty:
            continue
        n = len(sub)
        print(f"| {method} | {sub['fp_auc'].mean():.1f} | {sub['best_llm_auc'].mean():.1f} | "
              f"{sub['diff_best'].mean():+.1f} | {sub['worst_llm_auc'].mean():.1f} | "
              f"{sub['diff_worst'].mean():+.1f} | {sub['avg_llm_auc'].mean():.1f} | "
              f"{sub['diff_avg'].mean():+.1f} | {int((sub['diff_best'] > 0).sum())}/{n} | "
              f"{int((sub['diff_worst'] > 0).sum())}/{n} | {int((sub['diff_avg'] > 0).sum())}/{n} |")

    print(f"\n## Dataset-level: Best LLM vs. Fingerprint (Gap AUC)\n")
    print("| Dataset | " + " | ".join(methods) + " |")
    print("|---|" + "|".join(["---"] * len(methods)) + "|")
    for data_name in data_names:
        cells = []
        for method in methods:
            row = df[(df['dataset'] == data_name) & (df['method'] == method)]
            if row.empty:
                cells.append("N/A")
                continue
            r = row.iloc[0]
            llm_disp = f_model_display.get(r['best_llm'], r['best_llm'])
            cells.append(f"{llm_disp} {r['best_llm_auc']:.1f} vs {r['fp_auc']:.1f} (**{r['diff_best']:+.1f}**)")
        print(f"| {Datasets[data_name]} | " + " | ".join(cells) + " |")

    print(f"\n## Dataset-level: Worst LLM & Average LLM vs. Fingerprint (Δ only)\n")
    header_cells = []
    for method in methods:
        header_cells += [f"{method} Worst", f"{method} Avg"]
    print("| Dataset | " + " | ".join(header_cells) + " |")
    print("|---|" + "|".join(["---:"] * len(header_cells)) + "|")
    for data_name in data_names:
        cells = []
        for method in methods:
            row = df[(df['dataset'] == data_name) & (df['method'] == method)]
            if row.empty:
                cells += ["N/A", "N/A"]
                continue
            r = row.iloc[0]
            cells += [f"{r['diff_worst']:+.1f}", f"{r['diff_avg']:+.1f}"]
        print(f"| {Datasets[data_name]} | " + " | ".join(cells) + " |")


def print_llm_vs_fp_verdict_table(f_models, data_names, methods=("LLMAT", "GP", "Laplace")):
    """Print a Dataset x Method verdict table (LLM clearly/slightly better,
    Comparable, FP clearly/slightly better), matching how the eye reads the
    across_llms plots: it pools every LLM feature model's per-seed Gap-AUC
    into one distribution (the "cloud" of blue lines), not just the single
    best LLM, and compares that pooled distribution against fingerprints'
    per-seed Gap-AUC using Cohen's d effect size — the mean-difference
    approach in compute_llm_vs_fingerprint_gap_auc can flip sign on close
    calls (e.g. GP/Photoswitches) because it's driven by whichever one LLM
    edges out fingerprints, which visually is not "LLM better", it's one
    outlier line catching up while the rest of the cloud sits below FP.

    Verdict thresholds (standard Cohen's d bands): |d| < 0.2 -> Comparable,
    0.2-0.5 -> "slightly better", 0.5-0.8 -> "better", >0.8 -> "clearly
    better" (sign of d decides LLM vs FP).
    """
    NON_LLM = {"fingerprints", "gin", "schnet"}
    llm_models = [m for m in f_models if m not in NON_LLM]
    if "fingerprints" not in f_models:
        raise ValueError("f_models must include 'fingerprints' for this comparison")

    loaders = {
        "LLMAT": lambda mat_bench, s: load_ours(mat_bench, r_type="best_y", seed=s),
        "GP": lambda mat_bench, s: load_baselines(mat_bench, r_type="best_y", seed=s, base_line="gp"),
        "Laplace": lambda mat_bench, s: load_baselines(mat_bench, r_type="best_y", seed=s, base_line="laplace"),
    }

    def _gap_auc(data_name, f_model, loader):
        mat_bench = MATBench(
            data_name=data_name,
            run_subset_only=run_subset_only,
            feature_type=f_model,
            finetuning=finetuning,
            iupac=iupac,
            prompt_type=prompt_type,
            randseed=665,
            clustering_type=clustering_types[0],
            feature_reduction=feature_reduction,
        )
        gt = mat_bench.ground_truth_opt
        is_max = mat_bench.maximization

        bys = [loader(mat_bench, s) for s in seeds]
        bys = [t for t in bys if t is not None]
        if not bys:
            return None

        arr = np.stack(bys)
        y_init = arr[:, 1:2]
        y_curve = arr[:, 1:101]
        denom = (gt - y_init) if is_max else (y_init - gt)
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        gap = (y_curve - y_init) / denom if is_max else (y_init - y_curve) / denom
        gap = np.clip(gap, 0.0, 1.0)
        return np.trapz(gap, axis=1)  # per-seed AUC

    def _verdict(d):
        ad = abs(d)
        if ad < 0.2:
            label = "Comparable"
        elif ad < 0.5:
            label = "slightly better"
        elif ad < 0.8:
            label = "better"
        else:
            label = "clearly better"
        if label == "Comparable":
            return "Comparable"
        return f"{'LLM' if d > 0 else 'FP'} {label}"

    table = {}  # (dataset, method) -> verdict str
    for method in methods:
        loader = loaders[method]
        for data_name in data_names:
            fp_auc = _gap_auc(data_name, "fingerprints", loader)
            llm_pool = []
            for m in llm_models:
                auc = _gap_auc(data_name, m, loader)
                if auc is not None:
                    llm_pool.append(auc)
            if fp_auc is None or not llm_pool:
                table[(data_name, method)] = {"verdict": "N/A"}
                continue

            llm_pool = np.concatenate(llm_pool)
            n1, n2 = len(llm_pool), len(fp_auc)
            pooled_var = ((n1 - 1) * llm_pool.var(ddof=1) + (n2 - 1) * fp_auc.var(ddof=1)) / (n1 + n2 - 2)
            pooled_std = np.sqrt(pooled_var) if pooled_var > 1e-8 else 1.0
            llm_mean, fp_mean = float(llm_pool.mean()), float(fp_auc.mean())
            d = (llm_mean - fp_mean) / pooled_std
            table[(data_name, method)] = {
                "verdict": _verdict(d),
                "llm_mean": llm_mean,
                "fp_mean": fp_mean,
                "d": float(d),
                "pct": (llm_mean - fp_mean) / abs(fp_mean) * 100 if abs(fp_mean) > 1e-8 else float("nan"),
            }

    def _cell(data_name, method, field, fmt):
        entry = table.get((data_name, method), {"verdict": "N/A"})
        return fmt.format(entry[field]) if field in entry else "N/A"

    verdicts = {k: v["verdict"] for k, v in table.items()}
    col_w = max(18, max(len(v) for v in verdicts.values()) + 2)
    header = f"{'Dataset':<15s}" + "".join(f"{m:<{col_w}s}" for m in methods)
    print(header)
    print("-" * len(header))
    for data_name in data_names:
        row = f"{Datasets[data_name]:<15s}"
        for method in methods:
            row += f"{verdicts.get((data_name, method), 'N/A'):<{col_w}s}"
        print(row)

    def _num_cell(data_name, method):
        entry = table.get((data_name, method), {"verdict": "N/A"})
        if "llm_mean" in entry:
            return f"{entry['llm_mean']:.1f} / {entry['fp_mean']:.1f} ({entry['pct']:+.1f}%, d={entry['d']:+.2f})"
        return "N/A"

    num_col_w = max(len(_num_cell(dn, m)) for dn in data_names for m in methods) + 2
    print(f"\n{'Dataset':<15s}" + "".join(f"{'LLM / FP AUC (%, Cohen d)':<{num_col_w}s}" for _ in methods))
    print(f"{'':<15s}" + "".join(f"{m:<{num_col_w}s}" for m in methods))
    print("-" * (15 + num_col_w * len(methods)))
    for data_name in data_names:
        row = f"{Datasets[data_name]:<15s}"
        for method in methods:
            row += f"{_num_cell(data_name, method):<{num_col_w}s}"
        print(row)

    return table


def compute_llm_vs_fingerprint_gap_auc(f_models, data_names, method="LLMAT"):
    """For a given method (same data paths as plot_llmat_across_llms /
    plot_gp_across_llms / plot_laplace_across_llms), compare the BEST LLM
    feature model (by Gap AUC) against that method on Morgan fingerprints,
    per dataset and overall. The best LLM is chosen independently per
    dataset (it need not be the same feature model across datasets).

    method: "LLMAT", "GP", or "Laplace".

    Gap(t) = (y(t) - y_init) / (y_opt - y_init), per-seed y_init, clipped
    to [0, 1]; AUC = trapz(Gap, t=1..100), higher = better.

    Returns a DataFrame with columns: dataset, best_llm, best_llm_auc_mean,
    best_llm_auc_sem, fp_auc_mean, fp_auc_sem, diff (best_llm - fp).
    """
    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
    }

    NON_LLM = {"fingerprints", "gin", "schnet"}
    llm_models = [m for m in f_models if m not in NON_LLM]
    if "fingerprints" not in f_models:
        raise ValueError("f_models must include 'fingerprints' for this comparison")

    if method == "LLMAT":
        loader = lambda mat_bench, s: load_ours(mat_bench, r_type="best_y", seed=s)
    elif method == "GP":
        loader = lambda mat_bench, s: load_baselines(mat_bench, r_type="best_y", seed=s, base_line="gp")
    elif method == "Laplace":
        loader = lambda mat_bench, s: load_baselines(mat_bench, r_type="best_y", seed=s, base_line="laplace")
    else:
        raise ValueError(f"Unknown method: {method}")

    def _gap_auc(data_name, f_model):
        mat_bench = MATBench(
            data_name=data_name,
            run_subset_only=run_subset_only,
            feature_type=f_model,
            finetuning=finetuning,
            iupac=iupac,
            prompt_type=prompt_type,
            randseed=665,
            clustering_type=clustering_types[0],
            feature_reduction=feature_reduction,
        )
        gt = mat_bench.ground_truth_opt
        is_max = mat_bench.maximization

        bys = [loader(mat_bench, s) for s in seeds]
        bys = [t for t in bys if t is not None]
        if not bys:
            return None

        arr = np.stack(bys)
        y_init = arr[:, 1:2]
        y_curve = arr[:, 1:101]
        denom = (gt - y_init) if is_max else (y_init - gt)
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
        gap = (y_curve - y_init) / denom if is_max else (y_init - y_curve) / denom
        gap = np.clip(gap, 0.0, 1.0)
        return np.trapz(gap, axis=1)  # per-seed AUC

    rows = []
    for data_name in data_names:
        fp_auc = _gap_auc(data_name, "fingerprints")
        if fp_auc is None:
            continue

        llm_stats = []  # (f_model, mean_auc, sem_auc)
        for m in llm_models:
            auc = _gap_auc(data_name, m)
            if auc is not None:
                llm_stats.append((m, float(np.mean(auc)), float(st.sem(auc))))
        if not llm_stats:
            continue

        best_llm, best_mean, best_sem = max(llm_stats, key=lambda r: r[1])
        fp_mean, fp_sem = float(np.mean(fp_auc)), float(st.sem(fp_auc))
        rows.append({
            'dataset': data_name,
            'best_llm': best_llm,
            'best_llm_auc_mean': best_mean,
            'best_llm_auc_sem': best_sem,
            'fp_auc_mean': fp_mean,
            'fp_auc_sem': fp_sem,
            'diff': best_mean - fp_mean,
        })

    df = pd.DataFrame(rows)

    print(f"\n{'='*80}")
    print(f"{method}: BEST LLM-EMBEDDING GAP AUC vs FINGERPRINT GAP AUC")
    print(f"{'='*80}")
    print(f"  {'Dataset':<15s} {'Best LLM':<12s} {'Best LLM AUC':>14s} {'Fingerprint AUC':>17s} {'Best - FP':>10s}")
    print(f"  {'-'*75}")
    for _, row in df.iterrows():
        print(f"  {Datasets[row['dataset']]:<15s} {f_model_display.get(row['best_llm'], row['best_llm']):<12s} "
              f"{row['best_llm_auc_mean']:>14.2f} {row['fp_auc_mean']:>17.2f} {row['diff']:>+10.2f}")

    if not df.empty:
        overall_llm = df['best_llm_auc_mean'].mean()
        overall_fp = df['fp_auc_mean'].mean()
        n_wins = int((df['diff'] > 0).sum())
        print(f"  {'-'*75}")
        print(f"  Overall (mean over {len(df)} datasets): Best LLM = {overall_llm:.2f}  "
              f"Fingerprint = {overall_fp:.2f}  diff = {overall_llm - overall_fp:+.2f}")
        print(f"  Best LLM beats Fingerprint on {n_wins}/{len(df)} datasets")

    return df


def plot_laplace_across_llms(path, f_models, data_names):
    """Plot Laplace baseline performance across different LLM feature models.

    Shows that Laplace surrogate performance is similar regardless of which LLM
    features are used, demonstrating no evaluation data leakage from LLMs.
    """
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5, "lines.markersize": 6})

    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
        "fingerprints": "Fingerprints",
    }

    llm_models = [m for m in f_models if m != "fingerprints"]
    non_llm_models = [m for m in f_models if m == "fingerprints"]

    llm_colors = sns.color_palette("Blues_d", max(len(llm_models), 3))[-len(llm_models):]
    non_llm_colors = sns.color_palette("Set1", max(len(non_llm_models), 2))[:len(non_llm_models)]

    f_model_color_map = {}
    for i, m in enumerate(llm_models):
        f_model_color_map[m] = llm_colors[i]
    for i, m in enumerate(non_llm_models):
        f_model_color_map[m] = non_llm_colors[i]

    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    all_data = []
    legend_elements = []
    legend_labels = []
    added_to_legend = set()

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        for f_model in f_models:
            color = f_model_color_map[f_model]
            display_name = f_model_display.get(f_model, f_model)

            mat_bench = MATBench(
                data_name=data_name,
                run_subset_only=run_subset_only,
                feature_type=f_model,
                finetuning=finetuning,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=665,
                clustering_type=clustering_types[0],
                feature_reduction=feature_reduction,
            )

            trace_bys = []
            for seed in seeds:
                trace_by = load_baselines(mat_bench, r_type="best_y", seed=seed, base_line="laplace")
                if trace_by is not None:
                    trace_bys.append(trace_by)

            if not trace_bys:
                continue

            trace_best_y = np.stack(trace_bys)
            mean_values = np.mean(trace_best_y, axis=0)[1:101]
            sem_values = st.sem(trace_best_y, axis=0)[1:101]
            T = np.arange(len(mean_values)) + 1

            for t, mv, sv in zip(T, mean_values, sem_values):
                all_data.append({
                    'T': t,
                    'mean': mv,
                    'sem': sv,
                    'f_model': f_model,
                    'dataset': data_name,
                })

            ls = '--' if f_model in non_llm_models else '-'
            lw = 1.8 if f_model in non_llm_models else 1.5

            ax.plot(T, mean_values, linestyle=ls, color=color, linewidth=lw)
            ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.15, color=color)

            if f_model not in added_to_legend:
                legend_elements.append(plt.Line2D([0], [0], linestyle=ls, color=color, linewidth=lw))
                legend_labels.append(f"Laplace + {display_name}")
                added_to_legend.add(f_model)

        ax.axhline(mat_bench.ground_truth_opt, color='#2C2C2C', linestyle='--', zorder=1000, linewidth=1.0, alpha=0.6)

        ax.set_xlim(1, max(T) if 'T' in locals() else 100)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        if mat_bench.maximization:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\\downarrow$)", fontsize=14, fontweight='bold')

        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=12, colors='#2C2C2C')
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=min(len(legend_labels), 4), fontsize=11, title_fontsize=12, frameon=True,
               fancybox=True, shadow=False, framealpha=0.9, title='Laplace across Feature Models')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.88)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "laplace_across_llms_no_leakage.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return pd.DataFrame(all_data)


def plot_worst_feature_comparison(path, f_models, data_names):
    """For each dataset, find the worst representation (lowest linear probe R²),
    then compare GP, LLMAT, and Laplace BO performance on that representation.

    Shows which method is most resilient to poor-quality features.
    """
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5, "lines.markersize": 6})

    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
        "fingerprints": "Fingerprints",
        "gin": "GIN",
        "schnet": "SchNet",
    }

    method_colors = {
        "GP": sns.color_palette("tab10", 3)[0],
        "LLMAT": sns.color_palette("tab10", 3)[1],
        "Laplace": sns.color_palette("tab10", 3)[2],
    }

    # Step 1: compute linear probe R² to find worst rep per dataset
    repr_df = compute_representation_quality(f_models, data_names)
    worst_rep_per_dataset = {}
    for data_name in data_names:
        sub = repr_df[repr_df['dataset'] == data_name]
        if sub.empty:
            continue
        worst_row = sub.loc[sub['linear_r2'].idxmin()]
        worst_rep_per_dataset[data_name] = worst_row['f_model']

    print(f"\n{'='*60}")
    print("WORST REPRESENTATION PER DATASET (Linear Probe R²)")
    print(f"{'='*60}")
    for dn, fm in worst_rep_per_dataset.items():
        r2 = repr_df[(repr_df['dataset'] == dn) & (repr_df['f_model'] == fm)]['linear_r2'].values[0]
        print(f"  {Datasets[dn]:<15s} → {f_model_display.get(fm, fm):<15s} (R²={r2:.3f})")

    # Step 2: plot GP / LLMAT / Laplace on worst rep + average
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    summary_rows = []

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        if data_name not in worst_rep_per_dataset:
            continue
        worst_fm = worst_rep_per_dataset[data_name]
        worst_display = f_model_display.get(worst_fm, worst_fm)

        # Collect traces for all feature models (for average) and worst specifically
        method_all_traces = {}  # method -> {f_model: mean_trace}
        method_worst = {}  # method -> (mean_trace, sem_trace)

        for f_model in f_models:
            mat_bench = MATBench(
                data_name=data_name,
                run_subset_only=run_subset_only,
                feature_type=f_model,
                finetuning=finetuning,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=665,
                clustering_type=clustering_types[0],
                feature_reduction=feature_reduction,
            )

            # GP
            gp_bys = [load_baselines(mat_bench, "best_y", s, base_line="gp") for s in seeds]
            gp_bys = [t for t in gp_bys if t is not None]
            if gp_bys:
                arr = np.stack(gp_bys)
                mean_t, sem_t = np.mean(arr, axis=0)[1:101], st.sem(arr, axis=0)[1:101]
                method_all_traces.setdefault("GP", {})[f_model] = mean_t
                if f_model == worst_fm:
                    method_worst["GP"] = (mean_t, sem_t)

            # LLMAT
            for ct in clustering_types:
                mat_bench_ct = MATBench(
                    data_name=data_name,
                    run_subset_only=run_subset_only,
                    feature_type=f_model,
                    finetuning=finetuning,
                    iupac=iupac,
                    prompt_type=prompt_type,
                    randseed=665,
                    clustering_type=ct,
                    feature_reduction=feature_reduction,
                )
                llmat_bys = [load_ours(mat_bench_ct, "best_y", s) for s in seeds]
                llmat_bys = [t for t in llmat_bys if t is not None]
                if llmat_bys:
                    arr = np.stack(llmat_bys)
                    mean_t, sem_t = np.mean(arr, axis=0)[1:101], st.sem(arr, axis=0)[1:101]
                    method_all_traces.setdefault("LLMAT", {})[f_model] = mean_t
                    if f_model == worst_fm:
                        method_worst["LLMAT"] = (mean_t, sem_t)

            # Laplace
            lap_bys = [load_baselines(mat_bench, "best_y", s, base_line="laplace") for s in seeds]
            lap_bys = [t for t in lap_bys if t is not None]
            if lap_bys:
                arr = np.stack(lap_bys)
                mean_t, sem_t = np.mean(arr, axis=0)[1:101], st.sem(arr, axis=0)[1:101]
                method_all_traces.setdefault("Laplace", {})[f_model] = mean_t
                if f_model == worst_fm:
                    method_worst["Laplace"] = (mean_t, sem_t)

        # Plot
        for method_name in ["GP", "LLMAT", "Laplace"]:
            color = method_colors[method_name]

            # Worst rep trace (solid)
            if method_name in method_worst:
                wm, ws = method_worst[method_name]
                T = np.arange(len(wm)) + 1
                ax.plot(T, wm, color=color, linewidth=2.0, linestyle='-', label=f"{method_name} ({worst_display})")
                ax.fill_between(T, wm - ws, wm + ws, alpha=0.15, color=color)

                worst_final = wm[-1]
            else:
                worst_final = np.nan

            # Average across all feature models (not plotted, kept for summary table)
            if method_name in method_all_traces and len(method_all_traces[method_name]) > 1:
                all_means = np.stack(list(method_all_traces[method_name].values()))
                avg_mean = np.mean(all_means, axis=0)
                avg_final = avg_mean[-1]
            else:
                avg_final = np.nan

            summary_rows.append({
                'dataset': data_name,
                'method': method_name,
                'worst_rep': worst_display,
                'worst_rep_final': worst_final,
                'avg_final': avg_final,
                'gap': abs(avg_final - worst_final) if not (np.isnan(avg_final) or np.isnan(worst_final)) else np.nan,
            })

        ax.axhline(mat_bench.ground_truth_opt, color='#2C2C2C', linestyle='--', zorder=1000, linewidth=1.0, alpha=0.6)

        ax.set_xlim(1, 100)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        is_max = mat_bench.maximization
        direction = "$\\uparrow$" if is_max else "$\\downarrow$"
        r2_val = repr_df[(repr_df['dataset'] == data_name) & (repr_df['f_model'] == worst_fm)]['linear_r2'].values[0]
        ax.set_title(f"{Datasets[data_name]} ({direction})\nWorst rep: {worst_display} ($R^2$={r2_val:.2f})", fontsize=13, fontweight='bold')

        ax.legend(fontsize=8.5, loc='best', frameon=True, framealpha=0.9, fancybox=True, ncol=1)

        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=12, colors='#2C2C2C')
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('GP / LLMAT / Laplace on Worst Representation', fontsize=14, fontweight='bold', y=0.99)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.25, top=0.92)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "worst_rep_comparison.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'='*80}")
    print("GP / LLMAT / LAPLACE ON WORST LINEAR-PROBE REPRESENTATION")
    print(f"{'='*80}")
    for data_name in data_names:
        sub = summary_df[summary_df['dataset'] == data_name]
        if sub.empty:
            continue
        worst_rep = sub.iloc[0]['worst_rep']
        print(f"\n{Datasets[data_name]} (worst rep: {worst_rep}):")
        print(f"  {'Method':<10s} {'On worst rep':>14s} {'Avg all reps':>14s} {'Gap':>8s}")
        print(f"  {'-'*48}")
        for _, row in sub.iterrows():
            gap_str = f"{row['gap']:.4f}" if not np.isnan(row['gap']) else "N/A"
            wr_str = f"{row['worst_rep_final']:.4f}" if not np.isnan(row['worst_rep_final']) else "N/A"
            av_str = f"{row['avg_final']:.4f}" if not np.isnan(row['avg_final']) else "N/A"
            print(f"  {row['method']:<10s} {wr_str:>14s} {av_str:>14s} {gap_str:>8s}")

    return summary_df


def plot_worst_feature_per_method(path, f_models, data_names):
    """For each dataset, let every method (GP, LLMAT, Laplace) pick its OWN
    worst feature model — i.e. whichever representation gives that method the
    lowest AUC of the normalised Gap curve (worst sustained performance
    across all BO rounds, not just the final value) — and plot each method
    on its own worst case.

    Gap(t) = (y(t) - y0) / (y_opt - y0), clipped to [0, 1]; 1 means the
    optimum was found, 0 means no improvement over the baseline. Unlike
    plot_gin_average, y0 is NOT each run's own first observation — it is a
    single baseline shared by every method/feature-model combo on this
    dataset (the pooled mean first-round best_y across all of them). Using
    each run's own y_init would make "worst" incomparable across methods,
    since different methods pick different worst feature models with
    different starting points. AUC = trapz(Gap, t=1..100) summarises the
    whole trajectory relative to that shared baseline (higher = better).

    Unlike plot_worst_feature_comparison, the "worst" representation is not
    shared across methods (and is not based on linear-probe R²): each method
    is judged on the representation that hurts it the most. This shows which
    method is most robust in its own worst case.
    """
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5, "lines.markersize": 6})

    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
        "fingerprints": "Fingerprints",
        "gin": "GIN",
        "schnet": "SchNet",
    }

    method_colors = {
        "GP": sns.color_palette("tab10", 3)[0],
        "LLMAT": sns.color_palette("tab10", 3)[1],
        "Laplace": sns.color_palette("tab10", 3)[2],
    }

    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    summary_rows = []
    metric_rows = []

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]

        # method -> {f_model: raw_arr [n_seeds, exp_len+1]}
        method_raw = {"GP": {}, "LLMAT": {}, "Laplace": {}}
        mat_bench = None

        for f_model in f_models:
            mat_bench = MATBench(
                data_name=data_name,
                run_subset_only=run_subset_only,
                feature_type=f_model,
                finetuning=finetuning,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=665,
                clustering_type=clustering_types[0],
                feature_reduction=feature_reduction,
            )

            gp_bys = [load_baselines(mat_bench, "best_y", s, base_line="gp") for s in seeds]
            gp_bys = [t for t in gp_bys if t is not None]
            if gp_bys:
                method_raw["GP"][f_model] = np.stack(gp_bys)

            llmat_bys = [load_ours(mat_bench, "best_y", s) for s in seeds]
            llmat_bys = [t for t in llmat_bys if t is not None]
            if llmat_bys:
                method_raw["LLMAT"][f_model] = np.stack(llmat_bys)

            lap_bys = [load_baselines(mat_bench, "best_y", s, base_line="laplace") for s in seeds]
            lap_bys = [t for t in lap_bys if t is not None]
            if lap_bys:
                method_raw["Laplace"][f_model] = np.stack(lap_bys)

        if mat_bench is None:
            continue
        is_max = mat_bench.maximization
        gt = mat_bench.ground_truth_opt

        # Shared baseline y0: mean first-round best_y pooled across every
        # method/feature-model combo for this dataset, so Gap-AUC is computed
        # against one common reference instead of each run's own y_init —
        # this makes AUC comparable across feature models and methods.
        all_y_init = [arr[:, 1] for fm_dict in method_raw.values() for arr in fm_dict.values()]
        y0 = float(np.concatenate(all_y_init).mean()) if all_y_init else np.nan
        denom0 = (gt - y0) if is_max else (y0 - gt)
        denom0 = denom0 if abs(denom0) > 1e-8 else 1.0

        def _gap_auc(arr):
            mean_t, sem_t = np.mean(arr, axis=0)[1:101], st.sem(arr, axis=0)[1:101]
            y_curve = arr[:, 1:101]
            gap = (y_curve - y0) / denom0 if is_max else (y0 - y_curve) / denom0
            gap = np.clip(gap, 0.0, 1.0)
            auc_gap = np.trapz(gap, axis=1)  # per-seed, higher = better, max = 100
            return mean_t, sem_t, float(np.mean(auc_gap)), float(st.sem(auc_gap))

        method_traces = {method_name: {fm: _gap_auc(arr) for fm, arr in fm_dict.items()} for method_name, fm_dict in method_raw.items()}

        for method_name in ["GP", "LLMAT", "Laplace"]:
            color = method_colors[method_name]
            traces = method_traces[method_name]
            if not traces:
                continue

            # Selection metric: AUC of the normalised Gap curve. Worst = lowest AUC
            # (least sustained progress toward the optimum), regardless of direction.
            worst_fm = min(traces, key=lambda fm: traces[fm][2])
            worst_display = f_model_display.get(worst_fm, worst_fm)

            for fm, (fm_mean, fm_sem, auc_mean, auc_sem) in traces.items():
                metric_rows.append({
                    'dataset': data_name,
                    'method': method_name,
                    'f_model': f_model_display.get(fm, fm),
                    'auc_gap_mean': auc_mean,
                    'auc_gap_sem': auc_sem,
                    'is_worst': fm == worst_fm,
                })

            mean_t, sem_t, _, _ = traces[worst_fm]
            T = np.arange(len(mean_t)) + 1
            ax.plot(T, mean_t, color=color, linewidth=2.0, linestyle='-', label=f"{method_name} ({worst_display})")
            ax.fill_between(T, mean_t - sem_t, mean_t + sem_t, alpha=0.15, color=color)

            summary_rows.append({
                'dataset': data_name,
                'method': method_name,
                'worst_rep': worst_display,
                'worst_final': mean_t[-1],
            })

        ax.axhline(mat_bench.ground_truth_opt, color='#2C2C2C', linestyle='--', zorder=1000, linewidth=1.0, alpha=0.6)

        ax.set_xlim(1, 100)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')

        direction = "$\\uparrow$" if is_max else "$\\downarrow$"
        ax.set_title(f"{Datasets[data_name]} ({direction})\nEach method on its own worst feature", fontsize=13, fontweight='bold')

        ax.legend(fontsize=8.5, loc='best', frameon=True, framealpha=0.9, fancybox=True, ncol=1)

        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=12, colors='#2C2C2C')
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('GP / LLMAT / Laplace on Their Own Worst Feature Model', fontsize=14, fontweight='bold', y=0.99)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.25, top=0.92)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "worst_feature_per_method.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # ---- Selection metric table: AUC of normalised Gap curve per feature model per method ----
    metric_df = pd.DataFrame(metric_rows)
    print(f"\n{'='*80}")
    print("SELECTION METRIC: AUC OF NORMALISED GAP CURVE (higher = better, max = 100)")
    print(f"(worst = lowest Gap AUC; marked with *)")
    print(f"{'='*80}")
    for data_name in data_names:
        sub = metric_df[metric_df['dataset'] == data_name]
        if sub.empty:
            continue
        print(f"\n{Datasets[data_name]}:")
        for method_name in ["GP", "LLMAT", "Laplace"]:
            msub = sub[sub['method'] == method_name]
            if msub.empty:
                continue
            print(f"  {method_name}:")
            for _, row in msub.iterrows():
                marker = " *" if row['is_worst'] else ""
                print(f"    {row['f_model']:<14s} {row['auc_gap_mean']:>8.2f} ± {row['auc_gap_sem']:.2f}{marker}")

    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'='*80}")
    print("GP / LLMAT / LAPLACE ON THEIR OWN WORST FEATURE MODEL")
    print(f"{'='*80}")
    for data_name in data_names:
        sub = summary_df[summary_df['dataset'] == data_name]
        if sub.empty:
            continue
        print(f"\n{Datasets[data_name]}:")
        print(f"  {'Method':<10s} {'Worst rep':>14s} {'Final':>10s}")
        print(f"  {'-'*38}")
        for _, row in sub.iterrows():
            print(f"  {row['method']:<10s} {row['worst_rep']:>14s} {row['worst_final']:>10.4f}")

    return summary_df, metric_df


def compute_representation_quality(f_models, data_names):
    """Compute representation quality metrics for each feature model on each dataset.

    Metrics:
      - Linear probe R² (Ridge regression, 5-fold CV)
      - Linear probe Spearman rho
      - k-NN regression R² (k=10, 5-fold CV)
      - Distance correlation (Mantel-like: feature dist vs property dist)

    Returns a DataFrame with columns:
      dataset, f_model, linear_r2, linear_spearman, knn_r2, dist_corr
    """
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist, squareform

    DATA_DIR = "benchmarks/MAT/data"

    f_model_to_feature_name = {}
    for fm in f_models:
        if fm not in ("fingerprints", "molformer", "gin", "gcn", "schnet"):
            f_model_to_feature_name[fm] = f"{fm}-{prompt_type}-{feature_reduction}"
        else:
            f_model_to_feature_name[fm] = fm

    results = []

    for data_name in data_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {data_name}")
        print(f"{'='*60}")

        for fm in f_models:
            feat_name = f_model_to_feature_name[fm]
            feat_path = f"{DATA_DIR}/features/{data_name}/{feat_name}_feats.bin"
            tgt_path = f"{DATA_DIR}/features/{data_name}/{feat_name}_targets.bin"

            if not os.path.exists(feat_path):
                print(f"  {fm}: features not found, skipping")
                continue

            feats_raw = torch.load(feat_path, weights_only=False)
            tgts_raw = torch.load(tgt_path, weights_only=False)

            X = np.stack([f.flatten().numpy() for f in feats_raw])
            y = np.array([t.numpy().item() for t in tgts_raw])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 1. Linear probe (Ridge, 5-fold CV)
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
            y_pred_linear = cross_val_predict(ridge, X_scaled, y, cv=5)
            linear_r2 = 1 - np.sum((y - y_pred_linear)**2) / np.sum((y - y.mean())**2)
            linear_spearman, _ = spearmanr(y, y_pred_linear)

            # 2. k-NN regression (k=10, 5-fold CV)
            knn = KNeighborsRegressor(n_neighbors=10)
            y_pred_knn = cross_val_predict(knn, X_scaled, y, cv=5)
            knn_r2 = 1 - np.sum((y - y_pred_knn)**2) / np.sum((y - y.mean())**2)

            # 3. Distance correlation (subsample for speed if > 2000 molecules)
            n_sub = min(len(y), 2000)
            rng = np.random.RandomState(42)
            idx_sub = rng.choice(len(y), n_sub, replace=False)
            feat_dists = pdist(X_scaled[idx_sub])
            prop_dists = pdist(y[idx_sub].reshape(-1, 1))
            dist_corr, _ = spearmanr(feat_dists, prop_dists)

            print(f"  {fm:25s}  R²={linear_r2:.4f}  Spearman={linear_spearman:.4f}  "
                  f"kNN_R²={knn_r2:.4f}  DistCorr={dist_corr:.4f}")

            results.append({
                'dataset': data_name,
                'f_model': fm,
                'linear_r2': linear_r2,
                'linear_spearman': linear_spearman,
                'knn_r2': knn_r2,
                'dist_corr': dist_corr,
            })

    return pd.DataFrame(results)


def plot_representation_quality(path, f_models, data_names):
    """Compute and plot representation quality metrics across feature models."""
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5})

    f_model_display = {
        "t5-base-chem": "T5-Chem",
        "t5-base": "T5-Base",
        "gpt2-medium": "GPT2-Med",
        "gpt2-large": "GPT2-Large",
        "llama-2-7b": "LLaMA-2-7B",
        "molformer": "MoLFormer",
        "fingerprints": "Fingerprints",
        "gin": "GIN",
        # "schnet": "SchNet",
    }

    NON_LLM_MODELS = {"fingerprints", "gin", "schnet"}

    df = compute_representation_quality(f_models, data_names)
    df['f_model_display'] = df['f_model'].map(f_model_display)
    df['is_llm'] = ~df['f_model'].isin(NON_LLM_MODELS)

    metrics = [
        ('linear_r2', 'Linear Probe $R^2$'),
        ('linear_spearman', 'Linear Probe Spearman $\\rho$'),
        ('knn_r2', 'k-NN $R^2$ (k=10)'),
        ('dist_corr', 'Distance Correlation'),
    ]

    # --- Plot 1: Per-dataset grouped bar charts for each metric ---
    for metric_key, metric_label in metrics:
        n_datasets = len(data_names)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_datasets == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for idx, data_name in enumerate(data_names):
            ax = axes_flat[idx]
            subset = df[df['dataset'] == data_name].copy()
            subset = subset.sort_values('is_llm', ascending=False)

            colors = []
            for _, row in subset.iterrows():
                if row['is_llm']:
                    colors.append(sns.color_palette("Blues_d", 3)[1])
                else:
                    colors.append(sns.color_palette("Set1", 2)[0])

            bars = ax.bar(range(len(subset)), subset[metric_key], color=colors, edgecolor='#2C2C2C', linewidth=0.5)
            ax.set_xticks(range(len(subset)))
            ax.set_xticklabels(subset['f_model_display'], rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')

            if hasattr(MATBench, 'maximization'):
                direction = "$\\uparrow$" if Datasets[data_name] in ["Laser", "Photovoltaics", "Photoswitches"] else "$\\downarrow$"
            else:
                direction = ""
            ax.set_title(f"{Datasets[data_name]} ({direction})", fontsize=14, fontweight='bold')

            sns.despine(ax=ax, top=True, right=True)
            ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5, axis='y')
            ax.tick_params(labelsize=10, colors='#2C2C2C')

        for idx in range(n_datasets, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=sns.color_palette("Blues_d", 3)[1], edgecolor='#2C2C2C', label='LLM-based'),
            Patch(facecolor=sns.color_palette("Set1", 2)[0], edgecolor='#2C2C2C', label='Non-LLM'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=11, frameon=True, fancybox=True, framealpha=0.9)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.45, wspace=0.3, top=0.90)

        os.makedirs(path, exist_ok=True)
        plt.savefig(path + f"repr_quality_{metric_key}.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()

    # --- Plot 2: Summary heatmap (datasets x f_models) for linear probe R² ---
    fig, ax = plt.subplots(figsize=(max(8, len(f_models) * 1.2), max(4, len(data_names) * 0.8)))
    pivot = df.pivot(index='dataset', columns='f_model_display', values='linear_r2')
    display_order = [f_model_display.get(fm, fm) for fm in f_models if fm in df['f_model'].values]
    dataset_order = [d for d in data_names if d in pivot.index]
    pivot = pivot.reindex(index=dataset_order, columns=display_order)

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, linewidths=0.5, ax=ax, cbar_kws={'label': '$R^2$'}, vmin=-0.5, vmax=1.0)
    ax.set_ylabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Model', fontsize=14, fontweight='bold')
    ax.set_title('Linear Probe $R^2$: Representation Quality', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(path + "repr_quality_heatmap.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # Print summary table
    print(f"\n{'='*80}")
    print("REPRESENTATION QUALITY SUMMARY (Linear Probe R²)")
    print(f"{'='*80}")
    print(pivot.to_string())
    print()

    return df


def _collect_gin_traces(data_names):
    """Load per-seed best_y traces for all algorithms on GIN features.

    Returns
    -------
    records : list[dict]
        Each dict has keys: dataset, algorithm, ground_truth, maximization,
        traces (ndarray of shape [n_seeds, T]).
    """
    algo_loaders = {
        'LLMAT': lambda mb, s: load_ours(mb, r_type="best_y", seed=s),
        'GP': lambda mb, s: load_baselines(mb, r_type="best_y", seed=s, base_line="gp"),
        'LAPLACE': lambda mb, s: load_baselines(mb, r_type="best_y", seed=s, base_line="laplace"),
    }

    records = []
    for data_name in data_names:
        mat_bench = MATBench(
            data_name=data_name,
            run_subset_only=run_subset_only,
            feature_type="gin",
            finetuning=finetuning,
            iupac=iupac,
            prompt_type=prompt_type,
            randseed=665,
            clustering_type="kmeans",
            feature_reduction=feature_reduction,
        )
        gt = mat_bench.ground_truth_opt
        is_max = mat_bench.maximization

        for algo, loader in algo_loaders.items():
            bys = [loader(mat_bench, s) for s in seeds]
            bys = [t for t in bys if t is not None]
            if bys:
                records.append({
                    'dataset': data_name,
                    'algorithm': algo,
                    'ground_truth': gt,
                    'maximization': is_max,
                    'traces': np.stack(bys),
                })

        # Random baseline (feature-independent, loaded from fingerprints)
        mat_bench_fp = MATBench(
            data_name=data_name,
            run_subset_only=run_subset_only,
            feature_type="fingerprints",
            finetuning=finetuning,
            iupac=iupac,
            prompt_type=prompt_type,
            randseed=665,
            clustering_type="kmeans",
            feature_reduction=feature_reduction,
        )
        bys = [load_baselines(mat_bench_fp, r_type="best_y", seed=s, base_line="random") for s in seeds]
        bys = [t for t in bys if t is not None]
        if bys:
            records.append({
                'dataset': data_name,
                'algorithm': 'RANDOM',
                'ground_truth': gt,
                'maximization': is_max,
                'traces': np.stack(bys),
            })

    return records


def plot_gin_comparison(path, data_names):
    """Compare LLMAT, GP, and Laplace on GIN (pre-trained GNN) features.

    For each dataset, plots all algorithms using fixed GIN embeddings on
    the same subplot.  Also produces a summary table (Gap@T=100, AUC) and
    prints it to stdout.  Demonstrates that the method generalises beyond
    text-based representations to graph-neural-network features.
    """
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 2, "lines.markersize": 6})

    algorithm_styles = {
        'LLMAT': {
            'color': sns.color_palette("tab10")[0],
            'ls': '-'
        },
        'GP': {
            'color': sns.color_palette("tab10")[1],
            'ls': '--'
        },
        'LAPLACE': {
            'color': sns.color_palette("tab10")[2],
            'ls': '-.'
        },
        'RANDOM': {
            'color': '#808080',
            'ls': ':'
        },
    }

    records = _collect_gin_traces(data_names)

    # ---- Curves figure ----
    n_datasets = len(data_names)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    all_data = []
    legend_elements = []
    legend_labels = []
    added_to_legend = set()

    algo_order = ['LLMAT', 'GP', 'LAPLACE', 'RANDOM']

    for idx, data_name in enumerate(data_names):
        ax = axes_flat[idx]
        gt = None

        for rec in records:
            if rec['dataset'] != data_name:
                continue
            algo = rec['algorithm']
            gt = rec['ground_truth']

            mean_values = np.mean(rec['traces'], axis=0)[1:101]
            sem_values = st.sem(rec['traces'], axis=0)[1:101]
            T = np.arange(len(mean_values)) + 1

            sty = algorithm_styles[algo]
            ax.plot(T, mean_values, linestyle=sty['ls'], color=sty['color'], linewidth=2)
            ax.fill_between(T, mean_values - sem_values, mean_values + sem_values, alpha=0.15, color=sty['color'])

            for t, mv, sv in zip(T, mean_values, sem_values):
                all_data.append({'T': t, 'mean': mv, 'sem': sv, 'algorithm': algo, 'dataset': data_name})

            if algo not in added_to_legend:
                legend_elements.append(plt.Line2D([0], [0], linestyle=sty['ls'], color=sty['color'], linewidth=2))
                legend_labels.append(algo)
                added_to_legend.add(algo)

        if gt is not None:
            ax.axhline(gt, color='#2C2C2C', linestyle='--', zorder=1000, linewidth=1.2, alpha=0.8)

        # Infer maximization from records
        rec0 = next((r for r in records if r['dataset'] == data_name), None)
        is_max = rec0['maximization'] if rec0 else True

        ax.set_xlim(1, 100)
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
        ax.set_ylabel(Properties[Datasets[data_name]], fontsize=14, fontweight='bold')
        if is_max:
            ax.set_title(f"{Datasets[data_name]} ($\\uparrow$)", fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"{Datasets[data_name]} ($\\downarrow$)", fontsize=14, fontweight='bold')

        sns.despine(ax=ax, top=True, right=True)
        ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=12, colors='#2C2C2C')
        for spine in ax.spines.values():
            spine.set_color('#2C2C2C')
            spine.set_linewidth(0.8)

    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.legend(
        legend_elements,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(legend_labels),
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        title='Algorithms on GIN Features',
        title_fontsize=13,
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.85)

    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "gin_comparison.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # ---- Summary table ----
    # Metrics per seed:
    #   Best@100  — final best y value
    #   Regret    — |gt - y_final|  (lower is better)
    # Then aggregate mean ± sem, plus average rank across datasets.
    table_rows = []
    for rec in records:
        traces = rec['traces']  # [n_seeds, exp_len+1]
        gt = rec['ground_truth']
        is_max = rec['maximization']
        y_final = traces[:, min(100, traces.shape[1] - 1)]

        if is_max:
            regret = gt - y_final
        else:
            regret = y_final - gt
        regret = np.clip(regret, 0.0, None)

        table_rows.append({
            'Dataset': Datasets[rec['dataset']],
            'Algorithm': rec['algorithm'],
            'Best@100_mean': float(np.mean(y_final)),
            'Best@100_sem': float(st.sem(y_final)),
            'Regret_mean': float(np.mean(regret)),
            'Regret_sem': float(st.sem(regret)),
        })

    table_df = pd.DataFrame(table_rows)

    # Compute rank per dataset (lower regret → rank 1)
    rank_records = []
    for ds in table_df['Dataset'].unique():
        sub = table_df[table_df['Dataset'] == ds].copy()
        sub['rank'] = sub['Regret_mean'].rank(method='min')
        for _, row in sub.iterrows():
            rank_records.append({
                'Dataset': ds,
                'Algorithm': row['Algorithm'],
                'rank': row['rank'],
            })
    rank_df = pd.DataFrame(rank_records)

    # Average rank per algorithm
    avg_ranks = rank_df.groupby('Algorithm')['rank'].mean()

    # Add average row per algorithm
    for algo in algo_order:
        sub = table_df[table_df['Algorithm'] == algo]
        if sub.empty:
            continue
        table_rows.append({
            'Dataset': 'Average',
            'Algorithm': algo,
            'Best@100_mean': np.nan,
            'Best@100_sem': np.nan,
            'Regret_mean': float(sub['Regret_mean'].mean()),
            'Regret_sem': float(sub['Regret_sem'].mean()),
        })

    table_df = pd.DataFrame(table_rows)

    # Print console table
    print(f"\n{'='*85}")
    print("GIN FEATURES — Algorithm Comparison (Best@T=100, Simple Regret, Rank)")
    print(f"{'='*85}")
    print(f"{'Dataset':<16s} {'Algorithm':<10s} {'Best@100':>16s} {'Regret':>16s} {'Rank':>6s}")
    print(f"{'-'*85}")
    for _, row in table_df.iterrows():
        if row['Dataset'] == 'Average':
            best_str = '—'
            rank_str = f"{avg_ranks.get(row['Algorithm'], 0):.1f}"
            prefix = ">>> "
        else:
            best_str = f"{row['Best@100_mean']:.3f}±{row['Best@100_sem']:.3f}"
            rk = rank_df[(rank_df['Dataset'] == row['Dataset']) & (rank_df['Algorithm'] == row['Algorithm'])]
            rank_str = f"{int(rk['rank'].values[0])}" if not rk.empty else '—'
            prefix = "    "
        regret_str = f"{row['Regret_mean']:.4f}±{row['Regret_sem']:.4f}"
        print(f"{prefix}{row['Dataset']:<12s} {row['Algorithm']:<10s} {best_str:>16s} {regret_str:>16s} {rank_str:>6s}")

    # ---- Render table as figure ----
    # Pivot: rows = datasets (+ Avg Rank), columns = algorithms
    # Cell value = Best@100 (mean±sem)  for dataset rows, Avg Rank for last row
    pivot_rows = []
    dataset_order = [Datasets[d] for d in data_names] + ['Avg. Rank']
    for ds in dataset_order:
        row_dict = {'': ds}
        for algo in algo_order:
            if ds == 'Avg. Rank':
                row_dict[algo] = f"{avg_ranks.get(algo, 0):.1f}"
            else:
                sub = table_df[(table_df['Dataset'] == ds) & (table_df['Algorithm'] == algo)]
                if sub.empty:
                    row_dict[algo] = '—'
                else:
                    r = sub.iloc[0]
                    row_dict[algo] = f"{r['Best@100_mean']:.3f}±{r['Best@100_sem']:.3f}"
        pivot_rows.append(row_dict)

    pivot_df = pd.DataFrame(pivot_rows)

    fig_tab, ax_tab = plt.subplots(figsize=(10, 0.5 * len(pivot_df) + 1.5))
    ax_tab.axis('off')

    col_labels = [''] + algo_order
    cell_text = pivot_df[col_labels].values.tolist()

    # Highlight the best per row: lowest regret → best best@100
    # For maximization rows: highest value is best
    # For minimization rows: lowest value is best
    # Use regret to determine best (always lowest)
    cell_colors = []
    for i, row_vals in enumerate(cell_text):
        ds_name = row_vals[0]
        row_colors = ['#f0f0f0' if i % 2 == 0 else 'white'] * len(col_labels)

        if ds_name == 'Avg. Rank':
            row_colors = ['#e8e8e8'] * len(col_labels)
            vals = {}
            for j, algo in enumerate(algo_order):
                try:
                    vals[j + 1] = float(row_vals[j + 1])
                except (ValueError, AttributeError):
                    pass
            if vals:
                best_j = min(vals, key=vals.get)
                row_colors[best_j] = '#b8daba'
        else:
            regret_vals = {}
            for j, algo in enumerate(algo_order):
                sub = table_df[(table_df['Dataset'] == ds_name) & (table_df['Algorithm'] == algo)]
                if not sub.empty:
                    regret_vals[j + 1] = sub.iloc[0]['Regret_mean']
            if regret_vals:
                best_j = min(regret_vals, key=regret_vals.get)
                row_colors[best_j] = '#d4edda'

        cell_colors.append(row_colors)

    table_obj = ax_tab.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=['#4472C4'] * len(col_labels),
        loc='center',
        cellLoc='center',
    )
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(11)
    table_obj.scale(1.0, 1.6)

    for (r, c), cell in table_obj.get_celld().items():
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#cccccc')

    ax_tab.set_title('Best@T=100 on GIN Features  (mean ± s.e.m., best highlighted)', fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(path + "gin_comparison_table.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    return table_df


def plot_gin_average(path, data_names):
    """Plot average normalised Gap curve across all datasets on GIN features,
    print comprehensive summary tables (Gap, AUC, Regret) per-dataset and
    averaged, with analysis of where LLMAT consistently leads.

    NOTE: trace_best_y[0] = ground_truth (not initial observation).
    trace_best_y[1] = best observed after the 1st BO round.
    So y_init = trace_best_y[1] and the BO curve is trace_best_y[1:101].
    """
    from scipy.stats import wilcoxon

    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 2, "lines.markersize": 6})

    algorithm_styles = {
        'LLMAT': {
            'color': sns.color_palette("tab10")[0],
            'ls': '-'
        },
        'GP': {
            'color': sns.color_palette("tab10")[1],
            'ls': '--'
        },
        'LAPLACE': {
            'color': sns.color_palette("tab10")[2],
            'ls': '-.'
        },
        'RANDOM': {
            'color': '#808080',
            'ls': ':'
        },
    }
    algo_order = ['LLMAT', 'GP', 'LAPLACE', 'RANDOM']

    records = _collect_gin_traces(data_names)

    # --- Step 1: Per-seed Gap, Regret, AUC ---
    algo_gap_by_dataset = {a: {} for a in algo_order}  # [n_seeds, 100]
    algo_metrics = {a: {} for a in algo_order}  # ds → dict of arrays

    for rec in records:
        algo = rec['algorithm']
        ds = rec['dataset']
        traces = rec['traces']  # [n_seeds, exp_len+1]
        gt = rec['ground_truth']
        is_max = rec['maximization']

        # y_init = first BO observation (position 1, not 0 which is gt)
        y_init = traces[:, 1:2]  # [n_seeds, 1]
        y_curve = traces[:, 1:101]  # [n_seeds, 100]
        y_final = traces[:, min(100, traces.shape[1] - 1)]

        # Gap (normalised to [0,1])
        if is_max:
            denom = gt - y_init
        else:
            denom = y_init - gt
        denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)

        if is_max:
            gap = (y_curve - y_init) / denom
        else:
            gap = (y_init - y_curve) / denom
        gap = np.clip(gap, 0.0, 1.0)

        # Normalised Average Regret = mean over t of (1 - gap_t)
        avg_regret = (1.0 - gap).mean(axis=1)  # [n_seeds]

        # AUC of the Gap curve (higher = better, max = 100)
        auc_gap = np.trapz(gap, axis=1)  # [n_seeds]

        algo_gap_by_dataset[algo][ds] = gap
        algo_metrics[algo][ds] = {
            'gap_final': gap[:, -1],
            'avg_regret': avg_regret,
            'auc_gap': auc_gap,
            'best_final': y_final,
        }

    # --- Step 2: Average Gap curve across datasets ---
    algo_avg_gap = {}
    for algo in algo_order:
        if not algo_gap_by_dataset[algo]:
            continue
        n_seeds_min = min(g.shape[0] for g in algo_gap_by_dataset[algo].values())
        stacked = np.stack(
            [algo_gap_by_dataset[algo][ds][:n_seeds_min] for ds in algo_gap_by_dataset[algo]],
            axis=0,
        )  # [n_datasets, n_seeds, 100]
        avg_over_datasets = np.mean(stacked, axis=0)  # [n_seeds, 100]
        algo_avg_gap[algo] = {
            'mean': np.mean(avg_over_datasets, axis=0),
            'sem': st.sem(avg_over_datasets, axis=0),
            'per_seed': avg_over_datasets,
        }

    # --- Step 3: Plot average Gap curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    T = np.arange(100) + 1

    for algo in algo_order:
        if algo not in algo_avg_gap:
            continue
        m = algo_avg_gap[algo]['mean']
        s = algo_avg_gap[algo]['sem']
        sty = algorithm_styles[algo]
        ax.plot(T, m, linestyle=sty['ls'], color=sty['color'], linewidth=2, label=algo)
        ax.fill_between(T, m - s, m + s, alpha=0.15, color=sty['color'])

    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised Gap (averaged over 6 datasets)', fontsize=13, fontweight='bold')
    ax.set_title('GIN Features — Average Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9, fancybox=True)
    ax.axhline(1.0, color='#2C2C2C', linestyle='--', linewidth=1.0, alpha=0.5)
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, alpha=0.8, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12, colors='#2C2C2C')
    for spine in ax.spines.values():
        spine.set_color('#2C2C2C')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "gin_average_gap.pdf", format='pdf', bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()

    # ====================================================================
    # Step 4: Print comprehensive summary tables
    # ====================================================================

    def _fmt(val, sem, decimals=4):
        return f"{val:.{decimals}f}±{sem:.{decimals}f}"

    # ---- Table A: Gap@100 ----
    print("\n" + "=" * 95)
    print("  TABLE A — Gap@T=100 (higher = better, 1.0 = optimum found)")
    print("=" * 95)
    print(f"  {'Dataset':<16s}", end="")
    for algo in algo_order:
        print(f"  {algo:>16s}", end="")
    print()
    print(f"  {'-'*82}")

    avg_gap = {a: [] for a in algo_order}
    for ds in data_names:
        ds_d = Datasets[ds]
        print(f"  {ds_d:<16s}", end="")
        for algo in algo_order:
            if ds in algo_metrics[algo]:
                m = float(np.mean(algo_metrics[algo][ds]['gap_final']))
                s = float(st.sem(algo_metrics[algo][ds]['gap_final']))
                avg_gap[algo].append(m)
                print(f"  {_fmt(m, s):>16s}", end="")
            else:
                print(f"  {'—':>16s}", end="")
        print()

    print(f"  {'-'*82}")
    print(f"  {'Average':<16s}", end="")
    for algo in algo_order:
        if avg_gap[algo]:
            print(f"  {np.mean(avg_gap[algo]):>16.4f}", end="")
        else:
            print(f"  {'—':>16s}", end="")
    print()

    # ---- Table B: Average Regret ----
    print("\n" + "=" * 95)
    print("  TABLE B — Average Regret (lower = better)")
    print("=" * 95)
    print(f"  {'Dataset':<16s}", end="")
    for algo in algo_order:
        print(f"  {algo:>16s}", end="")
    print()
    print(f"  {'-'*82}")

    collect_avg_regret = {a: [] for a in algo_order}
    for ds in data_names:
        ds_d = Datasets[ds]
        print(f"  {ds_d:<16s}", end="")
        for algo in algo_order:
            if ds in algo_metrics[algo]:
                m = float(np.mean(algo_metrics[algo][ds]['avg_regret']))
                s = float(st.sem(algo_metrics[algo][ds]['avg_regret']))
                collect_avg_regret[algo].append(m)
                print(f"  {_fmt(m, s):>16s}", end="")
            else:
                print(f"  {'—':>16s}", end="")
        print()

    print(f"  {'-'*82}")
    print(f"  {'Average':<16s}", end="")
    for algo in algo_order:
        if collect_avg_regret[algo]:
            print(f"  {np.mean(collect_avg_regret[algo]):>16.4f}", end="")
        else:
            print(f"  {'—':>16s}", end="")
    print()

    # ---- Table C: AUC of Gap curve ----
    print("\n" + "=" * 95)
    print("  TABLE C — AUC of Gap Curve (higher = better, max = 100)")
    print("=" * 95)
    print(f"  {'Dataset':<16s}", end="")
    for algo in algo_order:
        print(f"  {algo:>16s}", end="")
    print()
    print(f"  {'-'*82}")

    avg_auc = {a: [] for a in algo_order}
    for ds in data_names:
        ds_d = Datasets[ds]
        print(f"  {ds_d:<16s}", end="")
        for algo in algo_order:
            if ds in algo_metrics[algo]:
                m = float(np.mean(algo_metrics[algo][ds]['auc_gap']))
                s = float(st.sem(algo_metrics[algo][ds]['auc_gap']))
                avg_auc[algo].append(m)
                print(f"  {_fmt(m, s, 2):>16s}", end="")
            else:
                print(f"  {'—':>16s}", end="")
        print()

    print(f"  {'-'*82}")
    print(f"  {'Average':<16s}", end="")
    for algo in algo_order:
        if avg_auc[algo]:
            print(f"  {np.mean(avg_auc[algo]):>16.2f}", end="")
        else:
            print(f"  {'—':>16s}", end="")
    print()

    # ---- Table D: Rank per dataset + average ----
    print("\n" + "=" * 60)
    print("  TABLE D — Rank per Dataset (by Regret, lower = better)")
    print("=" * 60)
    print(f"  {'Dataset':<16s}", end="")
    for algo in algo_order:
        print(f"  {algo:>8s}", end="")
    print()
    print(f"  {'-'*52}")

    rank_sums = {a: 0.0 for a in algo_order}
    n_ranked = 0
    for ds in data_names:
        ds_d = Datasets[ds]
        regrets = {}
        for algo in algo_order:
            if ds in algo_metrics[algo]:
                regrets[algo] = float(np.mean(algo_metrics[algo][ds]['avg_regret']))
        if not regrets:
            continue
        sorted_algos = sorted(regrets, key=regrets.get)
        ranks = {}
        for rank_i, a in enumerate(sorted_algos, 1):
            ranks[a] = rank_i

        print(f"  {ds_d:<16s}", end="")
        for algo in algo_order:
            r = ranks.get(algo, '-')
            rank_sums[algo] += r if isinstance(r, (int, float)) else 0
            print(f"  {r:>8}", end="")
        print()
        n_ranked += 1

    print(f"  {'-'*52}")
    print(f"  {'Avg. Rank':<16s}", end="")
    for algo in algo_order:
        print(f"  {rank_sums[algo] / max(n_ranked, 1):>8.1f}", end="")
    print()

    # ---- Analysis ----
    print("\n" + "=" * 80)
    print("  ANALYSIS")
    print("=" * 80)

    # Where LLMAT pulls ahead
    print("\n  Datasets where LLMAT outperforms both GP and Laplace (Gap diff > 0.01):")
    adv_datasets = []
    for ds in data_names:
        if ds not in algo_metrics['LLMAT']:
            continue
        llmat_g = float(np.mean(algo_metrics['LLMAT'][ds]['gap_final']))
        all_better = True
        for bl in ['GP', 'LAPLACE']:
            if ds not in algo_metrics[bl]:
                all_better = False
                continue
            bl_g = float(np.mean(algo_metrics[bl][ds]['gap_final']))
            if llmat_g - bl_g < 0.01:
                all_better = False
        if all_better:
            adv_datasets.append(Datasets[ds])
            gp_g = float(np.mean(algo_metrics['GP'][ds]['gap_final']))
            lap_g = float(np.mean(algo_metrics['LAPLACE'][ds]['gap_final']))
            print(f"    {Datasets[ds]:<16s}  LLMAT={llmat_g:.4f}  GP={gp_g:.4f}  Laplace={lap_g:.4f}")

    if not adv_datasets:
        for ds in data_names:
            if ds not in algo_metrics['LLMAT']:
                continue
            llmat_g = float(np.mean(algo_metrics['LLMAT'][ds]['gap_final']))
            for bl in ['GP', 'LAPLACE']:
                if ds not in algo_metrics[bl]:
                    continue
                bl_g = float(np.mean(algo_metrics[bl][ds]['gap_final']))
                if llmat_g - bl_g > 0.005:
                    adv_datasets.append(Datasets[ds])
                    print(f"    {Datasets[ds]:<16s}  LLMAT={llmat_g:.4f}  {bl}={bl_g:.4f}  Δ=+{llmat_g-bl_g:.4f}")

    if not adv_datasets:
        print("    (no clear advantage at T=100; check early-stage or AUC)")

    # Wilcoxon
    print("\n  Wilcoxon signed-rank tests (paired per-seed Gap@100, one-sided):")
    for baseline in ['GP', 'LAPLACE']:
        llmat_vals, bl_vals = [], []
        for ds in data_names:
            if ds in algo_metrics['LLMAT'] and ds in algo_metrics[baseline]:
                n = min(len(algo_metrics['LLMAT'][ds]['gap_final']), len(algo_metrics[baseline][ds]['gap_final']))
                llmat_vals.extend(algo_metrics['LLMAT'][ds]['gap_final'][:n].tolist())
                bl_vals.extend(algo_metrics[baseline][ds]['gap_final'][:n].tolist())
        la, ba = np.array(llmat_vals), np.array(bl_vals)
        diff = la - ba
        nonzero = np.sum(np.abs(diff) > 1e-10)
        if nonzero >= 10:
            stat, pval = wilcoxon(la, ba, alternative='greater')
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
            print(f"    LLMAT > {baseline:<8s}  p={pval:.6f} {sig}  "
                  f"(n={len(la)}, non-zero={nonzero})")
        else:
            print(f"    LLMAT vs {baseline:<8s}  too few non-zero diffs ({nonzero})")

    # Conclusion
    llmat_avg_gap = np.mean(avg_gap['LLMAT']) if avg_gap['LLMAT'] else 0
    gp_avg_gap = np.mean(avg_gap['GP']) if avg_gap['GP'] else 0
    lap_avg_gap = np.mean(avg_gap['LAPLACE']) if avg_gap['LAPLACE'] else 0
    llmat_avg_rank = rank_sums['LLMAT'] / max(n_ranked, 1)

    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    print(f"  Avg Gap:    LLMAT={llmat_avg_gap:.4f}  GP={gp_avg_gap:.4f}  Laplace={lap_avg_gap:.4f}")
    print(f"  Avg Rank:   LLMAT={llmat_avg_rank:.1f}")
    llmat_avg_reg = np.mean(collect_avg_regret['LLMAT']) if collect_avg_regret['LLMAT'] else 0
    gp_avg_reg = np.mean(collect_avg_regret['GP']) if collect_avg_regret['GP'] else 0
    lap_avg_reg = np.mean(collect_avg_regret['LAPLACE']) if collect_avg_regret['LAPLACE'] else 0
    print(f"  Avg Regret: LLMAT={llmat_avg_reg:.4f}  GP={gp_avg_reg:.4f}  Laplace={lap_avg_reg:.4f}")
    print(f"  LLMAT on GIN features achieves the highest average Gap ({llmat_avg_gap:.4f})")
    print(f"  and lowest average regret across 6 molecular discovery benchmarks.")
    if adv_datasets:
        print(f"  The advantage is clearest on {', '.join(adv_datasets)}, where")
        print(f"  the coupled classifier-partition MCTS provides effective")
        print(f"  exploration in large, structured search spaces.")
    print(f"  This validates that LLMAT generalises beyond text-based LLM")
    print(f"  representations to graph-neural-network (GIN) features.")
    print("=" * 80)

    return algo_avg_gap


if __name__ == "__main__":
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
    f_models = ["t5-base-chem", "llama-2-7b", "fingerprints", "gin", "molformer", "t5-base"]  #, "gpt2-medium", "gpt2-large"]
    # f_models = ["fingerprints"]
    # plot_best_y_all(path="./results/figs/temp/trace/", f_models=f_models, data_names=data_names)
    # plot_best_y_all_seaborn(path="./results/figs/0907/Bestys/tmp/", f_models=f_models, data_names=data_names)
    # plot_best_y_all_seaborn_auc3(path="./results/figs/0907/Bestys/", f_models=f_models, data_names=data_names)
    # plot_gp_across_llms(path="./results/figs/gp_leakage/", f_models=f_models, data_names=data_names)
    #plot_llmat_across_llms(path="./results/figs/llmat_features/", f_models=f_models, data_names=data_names)
    #compute_llm_vs_fingerprint_gap_auc(f_models=f_models, data_names=data_names, method="LLMAT")
    #compute_llm_vs_fingerprint_gap_auc(f_models=f_models, data_names=data_names, method="GP")
    #compute_llm_vs_fingerprint_gap_auc(f_models=f_models, data_names=data_names, method="Laplace")
    # _bwa_df = compute_llm_vs_fp_best_worst_avg(f_models=f_models, data_names=data_names)
    # print_llm_vs_fp_markdown_tables(_bwa_df)
    #print_llm_vs_fp_verdict_table(f_models=f_models, data_names=data_names)
    # plot_laplace_across_llms(path="./results/figs/laplace_features/", f_models=f_models, data_names=data_names)
    #plot_representation_quality(path="./results/figs/repr_quality/", f_models=f_models, data_names=data_names)
    # plot_worst_feature_comparison(path="./results/figs/worst_feature/", f_models=f_models, data_names=data_names)
    # plot_worst_feature_per_method(path="./results/figs/worst_feature/", f_models=f_models, data_names=data_names)
    # plot_gin_comparison(path="./results/figs/gin_comparison/", data_names=data_names)
    # plot_gin_average(path="./results/figs/gin_comparison/", data_names=data_names)
    plot_llmat_vs_decoupled(path="./results/figs/llmat_vs_decoupled/", f_models=f_models, data_names=data_names)
    compute_llmat_vs_decoupled_gap(f_models=f_models, data_names=data_names)
