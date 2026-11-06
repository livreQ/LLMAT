# #!/usr/bin/env python
# """
# Plot comparison of LAPEFT and LLMAT on redox-mer dataset (multi-objective optimization)
# Shows hypervolume, real objectives, and scalarized best over different random seeds
# Redox-mer objectives: Minimize Ered, minimize Gsol (both minimization)
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as st
# from pathlib import Path

# # Set style for better plots
# plt.style.use('default')
# sns.set_palette("husl")

# # Color scheme matching reference
# FEATURE2COLOR = {
#     "lapeft-t5-base-chem": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
#     "llmat_kmeans-t5-base-chem": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
#     "llmat_llms-t5-base-chem": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
# }

# def load_redox_results():
#     """
#     Load results for redox-mer dataset from both LAPEFT and LLMAT
#     """
#     base_path = Path("results/multiobj/redox-mer/t5-base-chem-average")

#     results = {'lapeft': {}, 'llmat_kmeans': {}, 'llmat_llms': {}}

#     # Load LAPEFT results
#     lapeft_path = base_path / "multiobj_lapeft"
#     if lapeft_path.exists():
#         for npy_file in lapeft_path.glob("*.npy"):
#             suffix = npy_file.stem.replace("trace_hypervolume_", "").replace("trace_scalarized_best_y_",
#                                                                              "").replace("trace_best_y_", "").replace("trace_timing_", "").replace("timing_train_",
#                                                                                                                                                    "").replace("timing_preds_", "")

#             if suffix not in results['lapeft']:
#                 results['lapeft'][suffix] = {}

#             if "trace_hypervolume" in str(npy_file):
#                 results['lapeft'][suffix]['hypervolume'] = np.load(npy_file)
#             elif "trace_scalarized_best_y" in str(npy_file):
#                 results['lapeft'][suffix]['scalarized'] = np.load(npy_file)
#             elif "trace_best_y" in str(npy_file):
#                 results['lapeft'][suffix]['best_y'] = np.load(npy_file)

#     # Load LLMAT kmeans results
#     llmat_kmeans_path = base_path / "multiobj_llmat" / "kmeans"
#     if llmat_kmeans_path.exists():
#         for npy_file in llmat_kmeans_path.glob("*.npy"):
#             suffix = npy_file.stem.replace("trace_hypervolume_", "").replace("trace_scalarized_best_y_",
#                                                                              "").replace("trace_best_y_", "").replace("trace_timing_", "").replace("timing_train_",
#                                                                                                                                                    "").replace("timing_preds_", "")

#             if suffix not in results['llmat_kmeans']:
#                 results['llmat_kmeans'][suffix] = {}

#             if "trace_hypervolume" in str(npy_file):
#                 results['llmat_kmeans'][suffix]['hypervolume'] = np.load(npy_file)
#             elif "trace_scalarized_best_y" in str(npy_file):
#                 results['llmat_kmeans'][suffix]['scalarized'] = np.load(npy_file)
#             elif "trace_best_y" in str(npy_file):
#                 results['llmat_kmeans'][suffix]['best_y'] = np.load(npy_file)

#     # Load LLMAT llms results
#     llmat_llms_path = base_path / "multiobj_llmat" / "llms"
#     if llmat_llms_path.exists():
#         for npy_file in llmat_llms_path.glob("*.npy"):
#             suffix = npy_file.stem.replace("trace_hypervolume_", "").replace("trace_scalarized_best_y_",
#                                                                              "").replace("trace_best_y_", "").replace("trace_timing_", "").replace("timing_train_",
#                                                                                                                                                    "").replace("timing_preds_", "")

#             if suffix not in results['llmat_llms']:
#                 results['llmat_llms'][suffix] = {}

#             if "trace_hypervolume" in str(npy_file):
#                 results['llmat_llms'][suffix]['hypervolume'] = np.load(npy_file)
#             elif "trace_scalarized_best_y" in str(npy_file):
#                 results['llmat_llms'][suffix]['scalarized'] = np.load(npy_file)
#             elif "trace_best_y" in str(npy_file):
#                 results['llmat_llms'][suffix]['best_y'] = np.load(npy_file)

#     return results

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from pathlib import Path

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Color scheme matching reference
FEATURE2COLOR = {
    "lapeft-t5-base-chem": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    "llmat_kmeans-t5-base-chem": (0.8392156862745098, 0.15294117647058825, 0.1568623529411765),
    "llmat_llms-t5-base-chem": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
}

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 16})  # Set default font size for all text elements
plt.rcParams['axes.labelsize'] = 16  # Set font size for axes labels
plt.rcParams['axes.titlesize'] = 16  # Set font size for axes titles
# plt.rcParams['legend.fontsize'] = 16  # Set font size for legend
plt.rcParams['figure.titlesize'] = 16  # Set font size for figure title
plt.rcParams['axes.labelweight'] = 'bold'  # Set axes label weight to bold
plt.rcParams['axes.titleweight'] = 'bold'  # Set axes title weight to bold
plt.rcParams['lines.linewidth'] = 1  # Set line width for plots
plt.rcParams['lines.markersize'] = 8  # Set marker size for plots
plt.rcParams['grid.linestyle'] = '--'  # Set grid line style
plt.rcParams['grid.linewidth'] = 1.5  # Set grid line width
plt.rcParams['axes.grid'] = True  # Enable grid lines

from matplotlib.font_manager import FontProperties

legend_font = FontProperties(weight='bold', size=12)


def load_redox_results():
    """
    Load results for redox-mer dataset from both LAPEFT and LLMAT
    """
    base_path = Path("results/multiobj/redox-mer/t5-base-chem-average")

    results = {'lapeft': {}, 'llmat_kmeans': {}, 'llmat_llms': {}}

    # Helper function to load and store results
    def load_files(method_key, path):
        if path.exists():
            for npy_file in path.glob("*.npy"):
                stem = npy_file.stem
                # Extract seed/suffix: remove known prefixes
                suffix = stem.replace("trace_hypervolume_", "").replace("trace_scalarized_best_y_",
                                                                        "").replace("trace_best_y_", "").replace("trace_timing_", "").replace("timing_train_",
                                                                                                                                              "").replace("timing_preds_", "")

                if suffix not in results[method_key]:
                    results[method_key][suffix] = {}

                if "trace_hypervolume" in stem:
                    results[method_key][suffix]['hypervolume'] = np.load(npy_file)
                elif "trace_scalarized_best_y" in stem:
                    results[method_key][suffix]['scalarized'] = np.load(npy_file)
                elif "trace_best_y" in stem:
                    results[method_key][suffix]['best_y'] = np.load(npy_file)

    # Load results for all methods
    load_files('lapeft', base_path / "multiobj_lapeft")
    load_files('llmat_kmeans', base_path / "multiobj_llmat" / "kmeans")
    load_files('llmat_llms', base_path / "multiobj_llmat" / "llms")

    return results


def plot_hv_and_scalarized_subfigures(results):
    """
    Plot Hypervolume (top) and Scalarized Best Value (bottom) in separate subfigures
    sharing the x-axis, similar to the objectives plot.
    """
    # Create two subplots stacked vertically, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.set_size_inches(8, 4)

    methods = [('LAPEFT', 'lapeft'), ('LLMAT(K-means)', 'llmat_kmeans'), ('LLMAT (LLMs)', 'llmat_llms')]

    def plot_metric(ax, data_key, ylabel, negate_for_plot=False):
        """Helper function to process and plot mean/SEM for a given metric."""

        lines = []
        labels = []

        for method_name, key in methods:
            if key in results and results[key]:
                data_list = [seed_data[data_key] for seed_data in results[key].values() if data_key in seed_data]

                if data_list:
                    # Pad shorter trajectories
                    max_len = max(len(d) for d in data_list)
                    data_array = np.array([np.pad(d, (0, max_len - len(d)), 'edge') for d in data_list])

                    mean = np.mean(data_array, axis=0)[1:]  # Skip first point, over randseeds
                    sem = st.sem(data_array, axis=0)[1:]  # Standard Error of the Mean (SEM)
                    T = np.arange(len(mean)) + 1

                    if negate_for_plot:
                        # Negate for maximization visualization (since objectives are minimization)
                        mean = -mean
                        # SEM remains the same magnitude

                    c = FEATURE2COLOR[f"{key}-t5-base-chem"]

                    # Plot and store line handles for legend
                    line, = ax.plot(T, mean, color=c, label=method_name)
                    ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)

                    lines.append(line)
                    labels.append(method_name)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        return lines, labels

    # --- 1. Plot Hypervolume (ax1 - Top Subplot) ---
    hv_lines, hv_labels = plot_metric(
        ax=ax1,
        data_key='hypervolume',
        ylabel=r"Hypervolume",
        negate_for_plot=False  # Hypervolume is Maximized
    )

    ax1.set_title(r"Multi-Redox Hypervolume ($\uparrow$)")
    ax1.legend(hv_lines, hv_labels, loc='upper right', prop=legend_font)

    # --- 2. Plot Scalarized Best Value (ax2 - Bottom Subplot) ---
    scalar_lines, scalar_labels = plot_metric(
        ax=ax2,
        data_key='scalarized',
        ylabel=r"Scalarized Value",
        negate_for_plot=True  # Negate to show improvement as upward trend
    )

    # Set X-axis label only on the bottom subplot
    ax2.set_xlabel(r"$t$", fontsize=12)
    ax2.set_title(r"Multi-Redox Scalarized Objective ($\downarrow$)")
    ax2.set_xlim(1, 100)

    # Save the combined figure
    plt.savefig('redox_hv_scalarized_subfigures_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# --- Main Function for execution ---


def plot_objectives_comparison(results):
    """
    Plot real objectives comparison - matching reference style
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.set_size_inches(8, 4)

    objectives = ['Redox Potential', 'Solvation Energy']
    obj_indices = [0, 1]  # Assuming first two objectives

    methods = [('LAPEFT', 'lapeft'), ('LLAMAT (K-means)', 'llmat_kmeans'), ('LLMAT (LLMs)', 'llmat_llms')]

    axes = [ax1, ax2]
    for obj_idx, (obj_name, ax) in enumerate(zip(objectives, axes)):
        # Plot methods
        for method_name, key in methods:
            if key in results and results[key]:
                obj_values = []
                for seed_data in results[key].values():
                    if 'best_y' in seed_data:
                        best_y = seed_data['best_y']
                        if len(best_y) > obj_idx:
                            obj_values.append(best_y[:, obj_idx])

                if obj_values:
                    obj_array = np.array(obj_values)
                    max_len = max(len(obj) for obj in obj_values)
                    obj_padded = []
                    for obj in obj_values:
                        if len(obj) < max_len:
                            obj_padded.append(np.pad(obj, (0, max_len - len(obj)), 'edge'))
                        else:
                            obj_padded.append(obj)

                    obj_array = np.array(obj_padded)
                    mean = np.mean(obj_array, axis=0)[1:]  # Skip first point, over randseeds
                    sem = st.sem(obj_array, axis=0)[1:]  # Over randseeds
                    T = np.arange(len(mean)) + 1

                    # Negate objectives for minimization problems
                    mean = -mean
                    sem = sem  # SEM stays the same since we're negating

                    c = FEATURE2COLOR[f"{key}-t5-base-chem"]
                    ax.plot(T, mean, color=c, label=method_name)
                    ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.3)

        ax.set_ylabel(f"{obj_name}", fontsize=12)
        ax.set_xlim(1, 100)
        ax.grid(True, alpha=0.3)

    ax2.set_xlabel(r"$t$", fontsize=12)
    ax1.set_title(r"Multi_redox: Ered ($\downarrow$)")
    ax2.set_title(r"Multi_redox: Gsol ($\downarrow$)")
    # Add legend to top subplot
    ax1.legend(prop=legend_font)

    plt.savefig('redox_objectives_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("Loading redox-mer dataset results...")
    results = load_redox_results()

    print("Generating combined hypervolume and scalarized comparison plot (using subfigures)...")
    plot_hv_and_scalarized_subfigures(results)

    print("Combined plot saved as:")
    print("  - redox_hv_scalarized_subfigures_comparison.png")

    print("Generating objectives comparison plots...")
    plot_objectives_comparison(results)


if __name__ == "__main__":
    main()
