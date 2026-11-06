from benchmarks.MAT.material_benchmarks import MATBench
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from plotting.load_results import load_baselines
import os

seeds = list(set(range(15)))

data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]
# "solvation",
f_models = ["molformer"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
prompt_type = "just-smiles"
finetuning = [True, False]
iupac = False
feature_reduction = "average"
run_subset_only = False
clustering_types = ["kmeans"]  #, "llms"]

our_algo = "llmat"
base_lines = ["laplace", "gp"]
tree_depths = [2, 3, 4, 5, 6]

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
plt.figure(figsize=(6, 3))


def get_ours_suffix(data_name, f_model, seed, n_init_data=10, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.005, tree_depth=3, finetune=True):

    if finetune:
        lora_lr = "3e-5"
        lora_reg = 0
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
    else:
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
    return suffix


def load_ours(mat_bench, f_model, r_type, seed, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.01, tree_depth=3, finetune=True):

    # print(mat_bench.dataset_name)
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/llmat"
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"{path}/{clustering}"
    data_name = mat_bench.dataset_name.split("/")[0]
    suffix = get_ours_suffix(data_name, f_model, seed, n_init_data=10, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.005, tree_depth=tree_depth, finetune=finetune)
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


def plot_regret_tree_depth_ablation(path, f_models, data_names):
    """
    Plot the regret for different tree depths.
    """

    trace_regrets_alldata_ours = {"finetune_2": [], "fixed_2": [], "finetune_3": [], "fixed_3": [], "finetune_4": [], "fixed_4": []}

    def update_min_max(data):
        all_values = np.concatenate([np.ravel(arr) for arr in data])
        global_min = all_values.min()
        global_max = all_values.max()
        return global_min, global_max

    def min_max_scale(data, global_min, global_max, a=0, b=1):

        all_values = np.concatenate([np.ravel(arr) for arr in data])
        # global_min = all_values.min()
        # global_max = all_values.max()

        scaled_data = []
        for arr in data:
            arr = np.array(arr)
            arr_scaled = (arr - global_min) / (global_max - global_min) * (b - a) + a
            scaled_data.append(arr_scaled)
        return scaled_data

    for data_name in data_names:
        global_min = 10000000
        global_max = 0
        trace_regrets_ours = {}  #"finetune_" + str(tree_depth):[], "fixed_" + str(tree_depth): []}
        for tree_depth in tree_depths:
            for fi in finetuning:
                if fi:
                    label_mkr = "finetune"
                else:
                    label_mkr = "fixed"

                trace_regrets_ours[label_mkr + "_" + str(tree_depth)] = []

                for f_model in f_models:
                    for clustering_type in clustering_types:
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

                        print(mat_bench.dataset_name)
                        trace_bys = []
                        trace_regrets = []
                        for seed in seeds:
                            trace_by = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, tree_depth=tree_depth, finetune=fi)
                            trace_regret = deepcopy(trace_by)
                            if trace_by is not None:
                                for j in range(0, 101):
                                    if mat_bench.maximization:
                                        trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                    else:
                                        trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                                trace_bys.append(trace_by)
                                trace_regrets.append(trace_regret)
                                trace_regrets_ours[label_mkr + "_" + str(tree_depth)].append(trace_regret)
                        if trace_regrets_ours[label_mkr + "_" + str(tree_depth)]:
                            cur_min, cur_max = update_min_max(trace_regrets_ours[label_mkr + "_" + str(tree_depth)])
                            if cur_min < global_min:
                                global_min = cur_min
                            if cur_max > global_max:
                                global_max = cur_max
        print(trace_regrets_ours.keys())
        for tree_depth in tree_depths:
            for fi in finetuning:
                if fi:
                    label_mkr = "finetune"
                else:
                    label_mkr = "fixed"
                if trace_regrets_ours[label_mkr + "_" + str(tree_depth)]:
                    trace_regrets_ours[label_mkr + "_" + str(tree_depth)] = min_max_scale(trace_regrets_ours[label_mkr + "_" + str(tree_depth)], global_min, global_max)
                    trace_regrets_alldata_ours[label_mkr + "_" + str(tree_depth)] += trace_regrets_ours[label_mkr + "_" + str(tree_depth)]

    # print(trace_regrets_alldata_ours.keys())
    for tree_depth in tree_depths:
        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"
            if not trace_regrets_alldata_ours[label_mkr + "_" + str(tree_depth)]:
                continue
            # print(trace_regrets_alldata_ours[label_mkr + "_" + str(tree_depth)])
            trace_y_regrets = np.stack(trace_regrets_alldata_ours[label_mkr + "_" + str(tree_depth)])
            # trace_y_regrets = min_max_scale(trace_y_regrets)

            mean = np.mean(trace_y_regrets, axis=0)[1:101]  # Over randseeds
            sem = st.sem(trace_y_regrets, axis=0)[1:101]  # Over randseeds

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
            if fi:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{tree_depth - 2}", linestyle='--')  # 'sd' shows standard deviation as shaded region
            else:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{tree_depth - 2}")
            ax1.fill_between(T, mean - sem, mean + sem, alpha=0.1)
    plt.yscale('log')
    plt.xlim(1, max(T))
    # plt.ylim(0.2, 0.55)
    # from matplotlib.ticker import FormatStrFormatter
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.setp(ax1.get_legend().get_texts(), fontsize='12')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('Regret', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"Average Regret w.r.t Different Tree Depth ($\downarrow$)")
    plt.savefig(path + f_model + "_regret_finetune_average_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_gap_tree_depth_ablation(path, f_models, data_names):
    """
    Plot the gap for different tree depths.
    """
    plt.figure(figsize=(6, 3))
    trace_gaps_alldata_ours = {"finetune_2": [], "fixed_2": [], "finetune_3": [], "fixed_3": [], "finetune_4": [], "fixed_4": []}
    for tree_depth in tree_depths:
        for fi in finetuning:
            if fi:
                label_mkr = "finetune"
            else:
                label_mkr = "fixed"
            for f_model in f_models:
                for data_name in data_names:
                    for clustering_type in clustering_types:
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

                        trace_bys = []
                        trace_gaps = []
                        for seed in seeds:
                            trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, tree_depth=tree_depth, finetune=fi)
                            trace_gap = deepcopy(trace_by)
                            if trace_by is not None:
                                for j in range(1, 100):
                                    y_0 = trace_by[1]
                                    y_t = trace_by[j]
                                    trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                                trace_bys.append(trace_by)
                                trace_gaps.append(trace_gap)
                                trace_gaps_alldata_ours[label_mkr + "_" + str(tree_depth)].append(trace_gap)

            if not trace_gaps_alldata_ours[label_mkr + "_" + str(tree_depth)]:
                continue
            trace_y_gaps_all_ours = np.stack(trace_gaps_alldata_ours[label_mkr + "_" + str(tree_depth)])

            mean = np.mean(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds
            sem = st.sem(trace_y_gaps_all_ours, axis=0)[1:101]  # Over randseeds

            T = np.arange(len(mean)) + 1
            df = pd.DataFrame({'T': T, 'y': mean, 'std': sem})
            if fi:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{tree_depth-2}", linestyle='--')  # 'sd' shows standard deviation as shaded region
            else:
                ax1 = sns.lineplot(data=df, x='T', y='y', label=f"LLMAT-{label_mkr}-{tree_depth-2}")
            ax1.fill_between(T, mean - sem, mean + sem, alpha=0.1)

    # plt.xscale('log')

    plt.xlim(1, max(T) - 1)
    plt.ylim(0, 1)
    plt.setp(ax1.get_legend().get_texts(), fontsize='12')
    plt.xlabel('BO Rounds: T', fontsize=16)
    plt.ylabel('GAP', fontsize=16)
    plt.tight_layout()
    # plt.errorbar(df['T'], df['y'], yerr=df['std'], fmt='o')
    plt.title(f"Average GAP for Different Tree Depths ($\\uparrow$)")
    plt.savefig(path + f_model + "_GAP_finetune_average_ablation.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Fixed-feature-only tree depth ablation (no finetuning)
# ──────────────────────────────────────────────────────────────────────


def plot_regret_tree_depth_fixed(path, f_models, data_names):
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5})

    trace_regrets_alldata = {td: [] for td in tree_depths}

    for data_name in data_names:
        for tree_depth in tree_depths:
            for f_model in f_models:
                for clustering_type in clustering_types:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=False,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, tree_depth=tree_depth, finetune=False)
                        if trace_by is not None:
                            trace_regret = deepcopy(trace_by)
                            for j in range(0, 101):
                                if mat_bench.maximization:
                                    trace_regret[j] = mat_bench.ground_truth_opt - np.mean(trace_by[1:j + 2])
                                else:
                                    trace_regret[j] = np.mean(trace_by[1:j + 2]) - mat_bench.ground_truth_opt
                            initial_regret = trace_regret[0]
                            if initial_regret > 1e-10:
                                trace_regret = trace_regret / initial_regret
                            trace_regrets_alldata[tree_depth].append(trace_regret)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("viridis", len(tree_depths))

    for i, tree_depth in enumerate(tree_depths):
        if not trace_regrets_alldata[tree_depth]:
            continue
        trace_arr = np.stack(trace_regrets_alldata[tree_depth])
        mean = np.mean(trace_arr, axis=0)[1:101]
        sem = st.sem(trace_arr, axis=0)[1:101]
        T = np.arange(len(mean)) + 1

        ax.plot(T, mean, color=colors[i], linewidth=2.0, label=f"depth={tree_depth-2}")
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.12, color=colors[i])

    ax.set_yscale('log')
    ax.set_xlim(1, 100)
    ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Regret', fontsize=14, fontweight='bold')
    ax.set_title('Average Regret vs. Tree Depth — Fixed Features ($\\downarrow$)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, framealpha=0.9, fancybox=True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "regret_fixed_tree_depth_ablation.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


def plot_gap_tree_depth_fixed(path, f_models, data_names):
    sns.set_style("whitegrid", {"axes.linewidth": 0.8, "grid.linewidth": 0.5, "grid.alpha": 0.2})
    sns.set_context("paper", font_scale=1.1, rc={"lines.linewidth": 1.5})

    trace_gaps_alldata = {td: [] for td in tree_depths}

    for tree_depth in tree_depths:
        for f_model in f_models:
            for data_name in data_names:
                for clustering_type in clustering_types:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=False,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, tree_depth=tree_depth, finetune=False)
                        if trace_by is not None:
                            trace_gap = deepcopy(trace_by)
                            for j in range(1, 100):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                trace_gap[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            trace_gaps_alldata[tree_depth].append(trace_gap)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("viridis", len(tree_depths))

    for i, tree_depth in enumerate(tree_depths):
        if not trace_gaps_alldata[tree_depth]:
            continue
        trace_arr = np.stack(trace_gaps_alldata[tree_depth])
        mean = np.mean(trace_arr, axis=0)[1:101]
        sem = st.sem(trace_arr, axis=0)[1:101]
        T = np.arange(len(mean)) + 1

        ax.plot(T, mean, color=colors[i], linewidth=2.0, label=f"depth={tree_depth -2}")
        ax.fill_between(T, mean - sem, mean + sem, alpha=0.12, color=colors[i])

    ax.set_xlim(1, 99)
    ax.set_ylim(0, 1)
    ax.set_xlabel('BO Rounds: T', fontsize=14, fontweight='bold')
    ax.set_ylabel('GAP', fontsize=14, fontweight='bold')
    ax.set_title('Average GAP vs. Tree Depth — Fixed Features ($\\uparrow$)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, framealpha=0.9, fancybox=True)
    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "gap_fixed_tree_depth_ablation.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.show()


def print_tree_depth_summary(f_models, data_names):
    """Print a summary table: tree_depth vs GAP (AUC, T=50, T=100), regret, and time."""

    rows = []
    for tree_depth in tree_depths:
        all_gap_curves = []
        all_regret_curves = []
        all_times = []

        for f_model in f_models:
            for data_name in data_names:
                for clustering_type in clustering_types:
                    mat_bench = MATBench(
                        data_name=data_name,
                        run_subset_only=run_subset_only,
                        feature_type=f_model,
                        finetuning=False,
                        iupac=iupac,
                        prompt_type=prompt_type,
                        randseed=665,
                        clustering_type=clustering_type,
                        feature_reduction=feature_reduction,
                    )

                    for seed in seeds:
                        trace_by = load_ours(mat_bench, f_model, r_type="best_y", seed=seed, tree_depth=tree_depth, finetune=False)
                        if trace_by is not None:
                            gap_curve = np.zeros_like(trace_by)
                            for j in range(1, 100):
                                y_0 = trace_by[1]
                                y_t = trace_by[j]
                                gap_curve[j] = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                            all_gap_curves.append(gap_curve)

                        trace_yh = load_ours(mat_bench, f_model, r_type="trace_y_his", seed=seed, tree_depth=tree_depth, finetune=False)
                        if trace_yh is not None:
                            regret_curve = deepcopy(trace_yh)
                            for j in range(0, 101):
                                if mat_bench.maximization:
                                    regret_curve[j] = mat_bench.ground_truth_opt - np.mean(trace_yh[1:j + 2])
                                else:
                                    regret_curve[j] = np.mean(trace_yh[1:j + 2]) - mat_bench.ground_truth_opt
                            all_regret_curves.append(regret_curve)

                        trace_t = load_ours(mat_bench, f_model, r_type="trace_time", seed=seed, tree_depth=tree_depth, finetune=False)
                        if trace_t is not None:
                            all_times.append(np.sum(trace_t))

        n = len(all_gap_curves)
        if n > 0:
            gap_arr = np.stack(all_gap_curves)
            gap_auc = np.mean(np.trapz(gap_arr[:, 1:100], axis=1))
            gap_auc_sem = st.sem(np.trapz(gap_arr[:, 1:100], axis=1))
            gap_t25 = np.mean(gap_arr[:, 25])
            gap_t25_sem = st.sem(gap_arr[:, 25])
            gap_t50 = np.mean(gap_arr[:, 50])
            gap_t50_sem = st.sem(gap_arr[:, 50])
            gap_t100 = np.mean(gap_arr[:, 99])
            gap_t100_sem = st.sem(gap_arr[:, 99])
        else:
            gap_auc = gap_auc_sem = gap_t25 = gap_t25_sem = gap_t50 = gap_t50_sem = gap_t100 = gap_t100_sem = float('nan')

        n_reg = len(all_regret_curves)
        if n_reg > 0:
            reg_arr = np.stack(all_regret_curves)
            reg_final = np.mean(reg_arr[:, 100])
            reg_final_sem = st.sem(reg_arr[:, 100])
        else:
            reg_final = reg_final_sem = float('nan')

        n_time = len(all_times)
        time_mean = np.mean(all_times) if n_time else float('nan')
        time_sem = st.sem(all_times) if n_time > 1 else 0.0

        rows.append({
            "depth": tree_depth - 2,
            "n": n,
            "GAP_AUC": gap_auc, "GAP_AUC_sem": gap_auc_sem,
            "GAP@25": gap_t25, "GAP@25_sem": gap_t25_sem,
            "GAP@50": gap_t50, "GAP@50_sem": gap_t50_sem,
            "GAP@100": gap_t100, "GAP@100_sem": gap_t100_sem,
            "Regret@100": reg_final, "Regret@100_sem": reg_final_sem,
            "Time(s)": time_mean, "Time_sem": time_sem,
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 130)
    print("Tree Depth Ablation Summary (Fixed Features)")
    print("=" * 130)
    header = (f"{'depth':>5} | {'n':>4} | {'GAP AUC (↑)':>16} | {'GAP@T=25':>16} | "
              f"{'GAP@T=50':>16} | {'GAP@T=100':>16} | {'Regret@100 (↓)':>18} | {'Time(s)':>14}")
    print(header)
    print("-" * 130)
    for _, r in df.iterrows():
        auc_str = f"{r['GAP_AUC']:.2f} ± {r['GAP_AUC_sem']:.2f}"
        g25_str = f"{r['GAP@25']:.4f} ± {r['GAP@25_sem']:.4f}"
        g50_str = f"{r['GAP@50']:.4f} ± {r['GAP@50_sem']:.4f}"
        g100_str = f"{r['GAP@100']:.4f} ± {r['GAP@100_sem']:.4f}"
        reg_str = f"{r['Regret@100']:.4f} ± {r['Regret@100_sem']:.4f}"
        time_str = f"{r['Time(s)']:.1f} ± {r['Time_sem']:.1f}"
        print(f"{int(r['depth']):>5} | {int(r['n']):>4} | {auc_str:>16} | {g25_str:>16} | "
              f"{g50_str:>16} | {g100_str:>16} | {reg_str:>18} | {time_str:>14}")

    best_auc = df['GAP_AUC'].idxmax()
    best_reg = df['Regret@100'].idxmin()
    best_time = df['Time(s)'].idxmin()
    print("-" * 130)
    print(f"Best GAP AUC:  depth={int(df.loc[best_auc, 'depth'])}  ({df.loc[best_auc, 'GAP_AUC']:.2f})")
    print(f"Best Regret:   depth={int(df.loc[best_reg, 'depth'])}  ({df.loc[best_reg, 'Regret@100']:.4f})")
    print(f"Best Time:     depth={int(df.loc[best_time, 'depth'])}  ({df.loc[best_time, 'Time(s)']:.1f}s)")
    print("=" * 130)

    return df


if __name__ == "__main__":
    f_models = ["molformer"]  #"molformer", "t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large"]
    data_names = ["redox-mer", "solvation", "kinase", "laser", "pce", "photoswitch"]

    # plot_regret_tree_depth_ablation(path="./results/figs/0907/Regrets/", f_models=f_models, data_names=data_names)
    # plot_gap_tree_depth_ablation(path="./results/figs/0907/GAPs/", f_models=f_models, data_names=data_names)

    # f_models_fixed = ["t5-base-chem", "molformer", "t5-base", "gpt2-medium", "gpt2-large", "llama-2-7b"]
    f_models_fixed = ["molformer"]
    print_tree_depth_summary(f_models=f_models_fixed, data_names=data_names)
    # plot_regret_tree_depth_fixed(path="./results/figs/tree_depth/", f_models=f_models_fixed, data_names=data_names)
    # plot_gap_tree_depth_fixed(path="./results/figs/tree_depth/", f_models=f_models_fixed, data_names=data_names)
