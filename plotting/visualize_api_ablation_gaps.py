import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.stats as st


def compute_normalized_gap(trace):
    """
    Compute normalized GAP metric from a trace.
    GAP[t] = (y_t - y_0) / (y_opt - y_0)
    """
    y_opt = trace[0]
    y_0 = trace[1]

    if np.isclose(y_opt, y_0):
        return np.ones(len(trace) - 1)

    best_y_values = trace[1:]
    gap = (best_y_values - y_0) / (y_opt - y_0)

    return np.nan_to_num(gap, nan=1.0, posinf=1.0, neginf=0.0)


def visualize_api_ablation_gaps():
    """
    Visualizes the average normalized GAP metrics as subplots for each feature model,
    comparing different LLM API types (GPT-4o, Gemini, Qwen 3) used for clustering.
    """
    data_name = "redox-mer"
    algorithm = "llmat"

    base_results_path = f"results/{data_name}/"
    feature_models = [
        "t5-base-just-smiles-average",
        "gpt2-large-just-smiles-average",
        "gpt2-medium-just-smiles-average",
        "molformer",
        "t5-base-chem-just-smiles-average",
        # "llama-2-7b-just-smiles-average",
    ]

    api_configs = {
        "GPT-4o": "llms_prompt0",
        "Gemini": "llms_gemini",
        "Qwen 3": "llms_qwen3",
    }

    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = True

    n_models = len(feature_models)
    n_total = n_models + 1  # +1 for the aggregated plot
    n_cols = 3
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    # Collect per-model mean GAPs for the aggregated plot
    # {api_label: [mean_gap_model0, mean_gap_model1, ...]}
    agg_mean_gaps = {api_label: [] for api_label in api_configs}

    for idx, f_model in enumerate(feature_models):
        ax = axes_flat[idx]
        base_path = os.path.join(base_results_path, f_model, algorithm)

        # Check if any API dirs exist for this feature model
        has_data = any(glob.glob(os.path.join(base_path, api_dir, "trace_best_y_*.npy")) for api_dir in api_configs.values())

        if not has_data:
            print(f"No API ablation directories found in {base_path}")
            ax.set_title(f"{f_model}\n(No data)", fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=16, alpha=0.5)
            ax.set_xlabel("BO Rounds: T")
            ax.set_ylabel("GAP")
            continue

        subplot_max_len = 0
        for api_label, api_dir in api_configs.items():
            result_dir = os.path.join(base_path, api_dir)
            file_pattern = os.path.join(result_dir, "trace_best_y_*.npy")
            best_y_files = glob.glob(file_pattern)

            if not best_y_files:
                continue

            all_gaps = []
            for best_y_file in best_y_files:
                try:
                    trace = np.load(best_y_file)
                    if len(trace) < 2: continue
                    gap = compute_normalized_gap(trace)
                    all_gaps.append(gap)
                except Exception as e:
                    print(f"Could not process file {best_y_file}: {e}")

            if not all_gaps:
                continue

            # Pad within this API's own runs only
            api_max_len = max(len(g) for g in all_gaps)
            if api_max_len > subplot_max_len:
                subplot_max_len = api_max_len

            padded_gaps = [np.pad(g, (0, api_max_len - len(g)), 'edge') for g in all_gaps]
            gaps_array = np.array(padded_gaps)
            mean_gap = np.mean(gaps_array, axis=0)
            sem_gap = st.sem(gaps_array, axis=0)

            iterations = np.arange(len(mean_gap)) + 1

            ax.plot(iterations, mean_gap, label=api_label)
            ax.fill_between(iterations, mean_gap - sem_gap, mean_gap + sem_gap, alpha=0.2)

            # Store per-model mean for aggregation
            agg_mean_gaps[api_label].append(mean_gap)

        ax.set_title(f"{f_model}", fontweight='bold')
        ax.set_xlabel("BO Rounds: T")
        ax.set_ylabel("GAP")
        ax.legend()
        if subplot_max_len > 0:
            ax.set_xlim(1, subplot_max_len)
        ax.set_ylim(0, 1.05)

    # --- Aggregated plot: average over all feature models ---
    ax_agg = axes_flat[n_models]
    agg_max_len = 0
    for api_label in api_configs:
        per_model_means = agg_mean_gaps[api_label]
        if not per_model_means:
            continue

        # Pad all per-model means to the same length
        agg_len = max(len(m) for m in per_model_means)
        agg_max_len = max(agg_max_len, agg_len)
        padded = [np.pad(m, (0, agg_len - len(m)), 'edge') for m in per_model_means]
        stacked = np.array(padded)
        mean_all = np.mean(stacked, axis=0)
        sem_all = st.sem(stacked, axis=0)

        iterations = np.arange(len(mean_all)) + 1
        ax_agg.plot(iterations, mean_all, label=api_label)
        ax_agg.fill_between(iterations, mean_all - sem_all, mean_all + sem_all, alpha=0.2)

    ax_agg.set_title("Average over all feature models", fontweight='bold')
    ax_agg.set_xlabel("BO Rounds: T")
    ax_agg.set_ylabel("GAP")
    ax_agg.legend()
    if agg_max_len > 0:
        ax_agg.set_xlim(1, agg_max_len)
    ax_agg.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(n_total, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Average GAP for API Ablation by Feature Model", fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_filename = "gap_visualization_api_ablation.png"
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.close()


def compute_normalized_regret(trace):
    """
    Compute normalized simple regret from a trace.
    Regret[t] = (y_opt - y_t) / (y_opt - y_0) = 1 - GAP[t]
    """
    y_opt = trace[0]
    y_0 = trace[1]

    if np.isclose(y_opt, y_0):
        return np.zeros(len(trace) - 1)

    best_y_values = trace[1:]
    regret = (y_opt - best_y_values) / (y_opt - y_0)

    return np.nan_to_num(regret, nan=0.0, posinf=1.0, neginf=0.0)


def visualize_api_ablation_regret():
    """
    Visualizes the average normalized simple regret as subplots for each feature model,
    comparing different LLM API types (GPT-4o, Gemini, Qwen 3) used for clustering.
    """
    data_name = "redox-mer"
    algorithm = "llmat"

    base_results_path = f"results/{data_name}/"
    feature_models = [
        "t5-base-just-smiles-average",
        "gpt2-large-just-smiles-average",
        "gpt2-medium-just-smiles-average",
        "molformer",
        "t5-base-chem-just-smiles-average",
        # "llama-2-7b-just-smiles-average",
    ]

    api_configs = {
        "GPT-4o": "llms_prompt0",
        "Gemini": "llms_gemini",
        "Qwen 3": "llms_qwen3",
    }

    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = True

    n_models = len(feature_models)
    n_total = n_models + 1  # +1 for the aggregated plot
    n_cols = 3
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    # Collect per-model mean regrets for the aggregated plot
    agg_mean_regrets = {api_label: [] for api_label in api_configs}

    for idx, f_model in enumerate(feature_models):
        ax = axes_flat[idx]
        base_path = os.path.join(base_results_path, f_model, algorithm)

        has_data = any(glob.glob(os.path.join(base_path, api_dir, "trace_best_y_*.npy")) for api_dir in api_configs.values())

        if not has_data:
            print(f"No API ablation directories found in {base_path}")
            ax.set_title(f"{f_model}\n(No data)", fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=16, alpha=0.5)
            ax.set_xlabel("BO Rounds: T")
            ax.set_ylabel("Regret")
            continue

        subplot_max_len = 0
        for api_label, api_dir in api_configs.items():
            result_dir = os.path.join(base_path, api_dir)
            file_pattern = os.path.join(result_dir, "trace_best_y_*.npy")
            best_y_files = glob.glob(file_pattern)

            if not best_y_files:
                continue

            all_regrets = []
            for best_y_file in best_y_files:
                try:
                    trace = np.load(best_y_file)
                    if len(trace) < 2: continue
                    regret = compute_normalized_regret(trace)
                    all_regrets.append(regret)
                except Exception as e:
                    print(f"Could not process file {best_y_file}: {e}")

            if not all_regrets:
                continue

            api_max_len = max(len(r) for r in all_regrets)
            if api_max_len > subplot_max_len:
                subplot_max_len = api_max_len

            padded_regrets = [np.pad(r, (0, api_max_len - len(r)), 'edge') for r in all_regrets]
            regrets_array = np.array(padded_regrets)
            mean_regret = np.mean(regrets_array, axis=0)
            sem_regret = st.sem(regrets_array, axis=0)

            iterations = np.arange(len(mean_regret)) + 1

            ax.plot(iterations, mean_regret, label=api_label)
            ax.fill_between(iterations, mean_regret - sem_regret, mean_regret + sem_regret, alpha=0.2)

            agg_mean_regrets[api_label].append(mean_regret)

        ax.set_title(f"{f_model}", fontweight='bold')
        ax.set_xlabel("BO Rounds: T")
        ax.set_ylabel("Regret")
        ax.legend()
        if subplot_max_len > 0:
            ax.set_xlim(1, subplot_max_len)
        ax.set_ylim(-0.05, 1.05)

    # --- Aggregated plot: average over all feature models ---
    ax_agg = axes_flat[n_models]
    agg_max_len = 0
    for api_label in api_configs:
        per_model_means = agg_mean_regrets[api_label]
        if not per_model_means:
            continue

        agg_len = max(len(m) for m in per_model_means)
        agg_max_len = max(agg_max_len, agg_len)
        padded = [np.pad(m, (0, agg_len - len(m)), 'edge') for m in per_model_means]
        stacked = np.array(padded)
        mean_all = np.mean(stacked, axis=0)
        sem_all = st.sem(stacked, axis=0)

        iterations = np.arange(len(mean_all)) + 1
        ax_agg.plot(iterations, mean_all, label=api_label)
        ax_agg.fill_between(iterations, mean_all - sem_all, mean_all + sem_all, alpha=0.2)

    ax_agg.set_title("Average over all feature models", fontweight='bold')
    ax_agg.set_xlabel("BO Rounds: T")
    ax_agg.set_ylabel("Regret")
    ax_agg.legend()
    if agg_max_len > 0:
        ax_agg.set_xlim(1, agg_max_len)
    ax_agg.set_ylim(-0.05, 1.05)

    # Hide unused subplots
    for idx in range(n_total, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Average Regret for API Ablation by Feature Model", fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_filename = "regret_visualization_api_ablation.png"
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.close()


if __name__ == "__main__":
    visualize_api_ablation_gaps()
    visualize_api_ablation_regret()
