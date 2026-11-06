import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
from botorch.utils.multi_objective.hypervolume import (Hypervolume,
                                                       infer_reference_point)
from botorch.utils.multi_objective.pareto import is_non_dominated

# Add src to sys.path to allow importing utils.plots
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import plotting.plots as plot_utils

parser = argparse.ArgumentParser()
parser.add_argument("--poster", default=False, action="store_true")
args = parser.parse_args()

PROBLEMS = ["redox-mer", "laser"]
IS_MAX = {
    "redox-mer": False,
    "solvation": False,
    "kinase": False,
    "laser": True,
    "pce": True,
    "photoswitch": True,
}
FEATURE_NAMES_BASE = ["fingerprints", "molformer"]
REAL_FEATURE_NAMES_LLM = [
    "t5-base",
    "t5-base-chem",
]
FEATURE_NAMES_LLM = [f"{n}-average" for n in REAL_FEATURE_NAMES_LLM]
FEATURE_NAMES = FEATURE_NAMES_BASE + FEATURE_NAMES_LLM
REAL_FEATURE_NAMES = FEATURE_NAMES_BASE + REAL_FEATURE_NAMES_LLM

# Define all methods to compare
METHODS = ["laplace", "llmat"]

PROBLEM2TITLE = {
    "redox-mer": r"Multi-Redox ($\uparrow$)",
    "laser": r"Multi-Laser ($\uparrow$)",
}
PROBLEM2LABELS = {
    "redox-mer": ["Redox Potential", "Solvation Energy"],
    "laser": ["Strength", "Electronic Gap"],
}
METHOD2LABEL = {"random": "RS", "gp": "GP", "laplace": "LA", "llmat": "LLMAT"}
FEATURE2LABEL = {
    "fingerprints": "FP",
    "molformer": "MolFormer",
    "gpt2-medium": "GPT2-M",
    "gpt2-large": "GPT2-L",
    "llama-2-7b": "LL2-7B",
    "t5-base": "T5",
    "t5-base-chem": "T5-Chem",
}

# Define colors for each method-feature combination
# Using a set of distinct colors for better visualization
FEATURE_METHOD_TO_COLOR = {
    "laplace-fingerprints": (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    "laplace-molformer": (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    "laplace-t5-base-average": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    "laplace-t5-base-chem-average": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    "llmat-fingerprints": (0.6, 0.6, 0.6),  # Grey
    "llmat-molformer": (0.9921568627450981, 0.7490196078431373, 0.43529411764705883),  # Light Orange
    "llmat-t5-base-average": (0.39215686274509803, 0.7098039215686275, 0.803921568627451),  # Light Blue
    "llmat-t5-base-chem-average": (0.6196078431372549, 0.00392156862745098, 0.25882352941176473), # Dark Red/Purple
}

RANDSEEDS = [1, 2, 3, 4, 5]

FIG_WIDTH = 1
FIG_HEIGHT = 0.225
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=True, poster=args.poster
)
plt.rcParams.update(rc_params)

METRICS_TO_PLOT = [
    "trace_hypervolume",
    "trace_scalarized_best_y",
    "trace_Gsol",
    "trace_Ered",
]
METRIC_TITLES = {
    "trace_hypervolume": "Hypervolume",
    "trace_scalarized_best_y": "Scalarized Best Y",
    "trace_Gsol": "Gsol Objective",
    "trace_Ered": "Ered Objective",
}


def plot_metric(metric_prefix: str):
    fig, axs = plt.subplots(len(PROBLEMS), 1, sharex=True, constrained_layout=True)
    fig.set_size_inches(fig_width, fig_height * len(PROBLEMS))

    for i, (problem, ax) in enumerate(
        zip(PROBLEMS, axs.flatten() if isinstance(axs, np.ndarray) else [axs])
    ):
        if problem is None:
            continue

        # For hypervolume, plot optimal val
        if metric_prefix == "trace_hypervolume":
            targets = torch.load(
                f"data/cache/multiobj/{problem}/fingerprints_multi_targets.bin"
            )  # (n_data, n_obj)
            Y_pareto = targets[is_non_dominated(targets)]
            ref_point = infer_reference_point(Y_pareto)
            hv = Hypervolume(ref_point)
            max_hypervolume = hv.compute(Y_pareto)
            ax.axhline(max_hypervolume, c="k", ls="dashed", zorder=1000, label="Optimal HV")

        # Plot methods
        for method in METHODS:
            for feature_name, real_feature_name in zip(FEATURE_NAMES, REAL_FEATURE_NAMES):
                # Skip random/gp for non-fingerprint features if they are not defined
                # in the current context
                if (method == "random" or method == "gp") and feature_name != "fingerprints":
                    continue

                # Adjust path for llmat if needed, assuming llmat does not have a method subfolder like laplace
                # For this script, we assume method is part of the path
                path = f"results/multiobj/{problem}/{method}/{feature_name}"
                
                # Check if the directory exists before attempting to load data
                if not os.path.exists(path):
                    print(f"Path does not exist, skipping: {path}")
                    continue

                try:
                    trace_data = np.stack(
                        [
                            np.load(f"{path}/{metric_prefix}_10_ts_{rs}.npy")
                            for rs in RANDSEEDS
                        ]
                    )
                    # Exclude the first point (initial random samples) for plotting trends
                    mean = np.mean(trace_data, axis=0)[1:]  # Over randseeds
                    sem = st.sem(trace_data, axis=0)[1:]  # Over randseeds
                    T = np.arange(len(mean)) + 1

                    label_str = f"{METHOD2LABEL[method]}-{FEATURE2LABEL[real_feature_name]}"
                    color_key = f"{method}-{feature_name}"
                    if color_key in FEATURE_METHOD_TO_COLOR:
                        c = FEATURE_METHOD_TO_COLOR[color_key]
                    else:
                        # Assign a default color if not found, or skip
                        print(f"No color defined for {color_key}, skipping plot for this combination.")
                        continue
                    
                    ax.plot(
                        T,
                        mean,
                        color=c,
                        label=label_str,
                    )
                    ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)
                except FileNotFoundError:
                    print(f"Data file not found for {path}/{metric_prefix}. Skipping.")
                    continue
                except Exception as e:
                    print(f"Error loading or plotting data for {path}/{metric_prefix}: {e}. Skipping.")
                    continue

        title = f"{PROBLEM2TITLE[problem]}"
        ax.set_title(title)
        if i == len(PROBLEMS) - 1:  # Only for the bottom row
            ax.set_xlabel(r"Number of acquisition function evaluations")
        ax.set_ylabel(METRIC_TITLES[metric_prefix])
        ax.set_xlim(1, 100) # Assuming 100 iterations based on existing script
        ax.grid(True, linestyle='--', alpha=0.6)


    handles, labels = [], []
    # Collect unique handles and labels from all subplots to avoid duplicates in legend
    for ax in fig.get_axes():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Sort labels to ensure consistent legend order
    sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels = [l for l, h in sorted_labels_handles]
    sorted_handles = [h for l, h in sorted_labels_handles]


    fig.legend(
        sorted_handles,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15), # Adjusted bbox_to_anchor to place legend above titles
        mode="expand",
        ncols=4, # Adjust as needed based on number of entries
        fancybox=True,
        shadow=True,
    )

    # Save results
    path_to_save = "poster/figs" if args.poster else "figs" # Changed to current directory for figs
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    fname = f"{metric_prefix.replace('trace_', '')}_comparison"
    fname += "_poster" if args.poster else ""
    plt.savefig(f"{path_to_save}/{fname}.pdf", bbox_inches="tight")
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    for metric in METRICS_TO_PLOT:
        print(f"Generating plot for {metric}...")
        plot_metric(metric)
    print("All plots generated.")