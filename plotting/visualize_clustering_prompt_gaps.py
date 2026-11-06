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

    # Handle the case where the initial point is already optimal
    if np.isclose(y_opt, y_0):
        return np.ones(len(trace) - 1)

    best_y_values = trace[1:]
    gap = (best_y_values - y_0) / (y_opt - y_0)

    # Handle potential NaNs if y_opt is close to y_0 but not exactly
    return np.nan_to_num(gap, nan=1.0, posinf=1.0, neginf=0.0)


def visualize_clustering_prompt_gaps():
    """
    Visualizes the average normalized GAP metrics for different clustering prompts,
    referencing the style of plot_gap_ablation_avg.py.
    """
    base_path = "results/redox-mer/t5-base-just-smiles-average/llmat/"
    prompt_dirs = sorted(glob.glob(os.path.join(base_path, "llms_prompt*")))

    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['legend.fontsize'] = 14  # Set font size for legend

    fig, ax = plt.subplots(figsize=(8, 3))

    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)
        file_pattern = os.path.join(prompt_dir, "trace_best_y_*.npy")
        best_y_files = glob.glob(file_pattern)

        if not best_y_files:
            print(f"No 'trace_best_y' files found in {prompt_dir}")
            continue

        all_gaps = []
        for best_y_file in best_y_files:
            try:
                trace = np.load(best_y_file)
                if len(trace) < 2:
                    continue  # Not enough data to compute GAP
                gap = compute_normalized_gap(trace)
                all_gaps.append(gap)
            except Exception as e:
                print(f"Could not process file {best_y_file}: {e}")

        if not all_gaps:
            print(f"No valid GAP data could be calculated for {prompt_name}")
            continue

        # Pad shorter runs to the length of the longest run
        max_len = max(len(gap) for gap in all_gaps)
        padded_gaps = []
        for gap in all_gaps:
            pad_len = max_len - len(gap)
            padded_gaps.append(np.pad(gap, (0, pad_len), 'edge'))

        gaps_array = np.array(padded_gaps)
        mean_gap = np.mean(gaps_array, axis=0)
        sem_gap = st.sem(gaps_array, axis=0)

        iterations = np.arange(len(mean_gap)) + 1

        ax.plot(iterations, mean_gap, label=prompt_name)
        ax.fill_between(iterations, mean_gap - sem_gap, mean_gap + sem_gap, alpha=0.2)

    ax.set_title("Average GAP for Different Clustering Prompts")
    ax.set_xlabel("BO Rounds: T")
    ax.set_ylabel("GAP")
    ax.legend()
    ax.set_xlim(1, max_len if 'max_len' in locals() else 100)
    ax.set_ylim(0, 1.05)

    output_filename = "gap_visualization_normalized.png"
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.close()


if __name__ == "__main__":
    visualize_clustering_prompt_gaps()
