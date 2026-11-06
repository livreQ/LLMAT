#!/usr/bin/env python
"""
Plot comparison of LAPEFT and LLMAT on laser dataset (multi-objective optimization)
Shows hypervolume, real objectives, and scalarized best over different random seeds
"""

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
    "llmat_kmeans-t5-base-chem": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    "llmat_llms-t5-base-chem": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
}

def load_laser_results():
    """
    Load results for laser dataset from both LAPEFT and LLMAT
    """
    base_path = Path("results/multiobj/laser/t5-base-chem-average")

    results = {
        'lapeft': {},
        'llmat_kmeans': {},
        'llmat_llms': {}
    }

    # Load LAPEFT results
    lapeft_path = base_path / "multiobj_lapeft"
    if lapeft_path.exists():
        for npy_file in lapeft_path.glob("*.npy"):
            suffix = npy_file.stem.replace("trace_hypervolume_", "").replace("trace_scalarized_best_y_", "").replace("trace_best_y_", "").replace("trace_timing_", "").replace("timing_train_", "").replace("timing_preds_", "")

            if suffix not in results['lapeft']:
                results['lapeft'][suffix] = {}

            if "trace_hypervolume" in str(npy_file):
                results['lapeft'][suffix]['hypervolume'] = np.load(npy_file)
            elif "trace_scalarized_best_y" in str(npy_file):
                results['lapeft'][suffix]['scalarized'] = np.load(npy_file)
            elif "trace_best_y" in str(npy_file):
                results['lapeft'][suffix]['best_y'] = np.load(npy_file)

    # Load LLMAT kmeans results
    llmat_kmeans_path = base_path / "multiobj_llmat" / "kmeans"
    if llmat_kmeans_path.exists():
        for npy_file in llmat_kmeans_path.glob("*.npy"):
            suffix = npy_file.stem.replace("trace_hypervolume_", "").replace("trace_scalarized_best_y_", "").replace("trace_best_y_", "").replace("trace_timing_", "").replace("timing_train_", "").replace("timing_preds_", "")

            if suffix not in results['llmat_kmeans']:
                results['llmat_kmeans'][suffix] = {}

            if "trace_hypervolume" in str(npy_file):
                results['llmat_kmeans'][suffix]['hypervolume'] = np.load(npy_file)
            elif "trace_scalarized_best_y" in str(npy_file):
                results['llmat_kmeans'][suffix]['scalarized'] = np.load(npy_file)
            elif "trace_best_y" in str(npy_file):
                results['llmat_kmeans'][suffix]['best_y'] = np.load(npy_file)

    # Load LLMAT llms results
    llmat_llms_path = base_path / "multiobj_llmat" / "llms"
    if llmat_llms_path.exists():
        for npy_file in llmat_llms_path.glob("*.npy"):
            suffix = npy_file.stem.replace("trace_hypervolume_", "").replace("trace_scalarized_best_y_", "").replace("trace_best_y_", "").replace("trace_timing_", "").replace("timing_train_", "").replace("timing_preds_", "")

            if suffix not in results['llmat_llms']:
                results['llmat_llms'][suffix] = {}

            if "trace_hypervolume" in str(npy_file):
                results['llmat_llms'][suffix]['hypervolume'] = np.load(npy_file)
            elif "trace_scalarized_best_y" in str(npy_file):
                results['llmat_llms'][suffix]['scalarized'] = np.load(npy_file)
            elif "trace_best_y" in str(npy_file):
                results['llmat_llms'][suffix]['best_y'] = np.load(npy_file)

    return results

def plot_hypervolume_comparison(results):
    """
    Plot hypervolume comparison between methods - matching reference style
    """
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(8, 3)

    methods = [
        ('LAPEFT', 'lapeft'),
        ('LLMAT (K-means)', 'llmat_kmeans'),
        ('LLMAT (LLMs)', 'llmat_llms')
    ]

    # Plot methods
    for method_name, key in methods:
        if key in results and results[key]:
            hv_list = [seed_data['hypervolume'] for seed_data in results[key].values() if 'hypervolume' in seed_data]
            if hv_list:
                hv_array = np.array(hv_list)
                max_len = max(len(hv) for hv in hv_list)

                # Pad shorter trajectories
                hv_padded = []
                for hv in hv_list:
                    if len(hv) < max_len:
                        hv_padded.append(np.pad(hv, (0, max_len - len(hv)), 'edge'))
                    else:
                        hv_padded.append(hv)

                hv_array = np.array(hv_padded)
                mean = np.mean(hv_array, axis=0)[1:]  # Skip first point, over randseeds
                sem = st.sem(hv_array, axis=0)[1:]  # Over randseeds
                T = np.arange(len(mean)) + 1

                c = FEATURE2COLOR[f"{key}-t5-base-chem"]
                ax.plot(T, mean, color=c, label=method_name)
                ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title(r"Laser Dataset ($\uparrow$)")
    ax.set_xlim(1, 100)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend()

    plt.savefig('laser_hypervolume_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_objectives_comparison(results):
    """
    Plot real objectives comparison - matching reference style
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.set_size_inches(8, 6)

    objectives = ['Strength', 'Electronic Gap']
    obj_indices = [0, 1]  # Assuming first two objectives

    methods = [
        ('LAPEFT', 'lapeft'),
        ('LLMAT (K-means)', 'llmat_kmeans'),
        ('LLMAT (LLMs)', 'llmat_llms')
    ]

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

                    c = FEATURE2COLOR[f"{key}-t5-base-chem"]
                    ax.plot(T, mean, color=c, label=method_name)
                    ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)

        ax.set_ylabel(obj_name, fontsize=12)
        ax.set_xlim(1, 100)
        ax.grid(True, alpha=0.3)

    ax2.set_xlabel(r"$t$", fontsize=12)

    # Add legend to top subplot
    ax1.legend()

    plt.savefig('laser_objectives_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scalarized_comparison(results):
    """
    Plot scalarized best comparison - matching reference style
    """
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.set_size_inches(8, 3)

    methods = [
        ('LAPEFT', 'lapeft'),
        ('LLMAT (K-means)', 'llmat_kmeans'),
        ('LLMAT (LLMs)', 'llmat_llms')
    ]

    # Plot methods
    for method_name, key in methods:
        if key in results and results[key]:
            scalar_list = [seed_data['scalarized'] for seed_data in results[key].values() if 'scalarized' in seed_data]
            if scalar_list:
                scalar_array = np.array(scalar_list)
                max_len = max(len(s) for s in scalar_list)

                # Pad shorter trajectories
                scalar_padded = []
                for s in scalar_list:
                    if len(s) < max_len:
                        scalar_padded.append(np.pad(s, (0, max_len - len(s)), 'edge'))
                    else:
                        scalar_padded.append(s)

                scalar_array = np.array(scalar_padded)
                mean = np.mean(scalar_array, axis=0)[1:]  # Skip first point, over randseeds
                sem = st.sem(scalar_array, axis=0)[1:]  # Over randseeds
                T = np.arange(len(mean)) + 1

                c = FEATURE2COLOR[f"{key}-t5-base-chem"]
                ax.plot(T, mean, color=c, label=method_name)
                ax.fill_between(T, mean - sem, mean + sem, color=c, alpha=0.25)

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel("Scalarized Best Value", fontsize=12)
    ax.set_title(r"Laser Dataset ($\uparrow$)")
    ax.set_xlim(1, 100)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend()

    plt.savefig('laser_scalarized_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading laser dataset results...")
    results = load_laser_results()

    print("Generating hypervolume comparison plots...")
    plot_hypervolume_comparison(results)

    print("Generating objectives comparison plots...")
    plot_objectives_comparison(results)

    print("Generating scalarized comparison plots...")
    plot_scalarized_comparison(results)

    print("Plots saved as:")
    print("  - laser_hypervolume_comparison.png")
    print("  - laser_objectives_comparison.png")
    print("  - laser_scalarized_comparison.png")

if __name__ == "__main__":
    main()
