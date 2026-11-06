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


def visualize_gaps_by_feature_model():
    """
    Visualizes the average normalized GAP metrics as subplots for each feature model.
    """
    data_name = "redox-mer"
    algorithm = "llmat"

    base_results_path = f"results/{data_name}/"
    feature_models = [
        "t5-base-just-smiles-average",
        # "fingerprints",
        "gpt2-large-just-smiles-average",
        "gpt2-medium-just-smiles-average",
        # "llama-2-7b-just-smiles-average",
        "molformer",
        "t5-base-chem-just-smiles-average"
    ]

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
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    for idx, f_model in enumerate(feature_models):
        ax = axes_flat[idx]
        base_path = os.path.join(base_results_path, f_model, algorithm)
        prompt_dirs = sorted(glob.glob(os.path.join(base_path, "llms_prompt*")))

        if not prompt_dirs:
            print(f"No 'llms_prompt' directories found in {base_path}")
            ax.set_title(f"{f_model}\n(No data)", fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=16, alpha=0.5)
            ax.set_xlabel("BO Rounds: T")
            ax.set_ylabel("Normalized GAP")
            continue

        max_len = 0
        for prompt_dir in prompt_dirs:
            prompt_name = os.path.basename(prompt_dir)
            file_pattern = os.path.join(prompt_dir, "trace_best_y_*.npy")
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

            current_max_len = max(len(gap) for gap in all_gaps)
            if current_max_len > max_len:
                max_len = current_max_len

            padded_gaps = [np.pad(g, (0, max_len - len(g)), 'edge') for g in all_gaps]
            gaps_array = np.array(padded_gaps)
            mean_gap = np.mean(gaps_array, axis=0)
            sem_gap = st.sem(gaps_array, axis=0)

            iterations = np.arange(len(mean_gap)) + 1

            ax.plot(iterations, mean_gap, label=prompt_name)
            ax.fill_between(iterations, mean_gap - sem_gap, mean_gap + sem_gap, alpha=0.2)

        ax.set_title(f"{f_model}", fontweight='bold')
        ax.set_xlabel("BO Rounds: T")
        ax.set_ylabel("Normalized GAP")
        ax.legend()
        if max_len > 0:
            ax.set_xlim(1, max_len)
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Average Normalized GAP for Different Clustering Prompts by Feature Model", fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_filename = "gap_visualization_by_feature_model.png"
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.close()


if __name__ == "__main__":
    visualize_gaps_by_feature_model()
