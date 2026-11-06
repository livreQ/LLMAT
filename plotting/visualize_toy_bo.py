#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Visualization script for toy BO experiments
Visualizes candidates, clusters, and decision boundaries for LLMAT on toy functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import yaml
import os
import sys
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import helpers
from torch.utils.data import DataLoader

import torch.nn as nn  # Need to import for classifier loading/use
# Import directly using importlib to avoid __init__.py transformers dependency
import importlib.util

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
plt.rcParams['legend.fontsize'] = 16  # Set font size for legend


def import_module_from_file(module_name, file_path):
    """Import a module from a file path, bypassing __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get the base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import llmat_toy
llmat_toy_path = os.path.join(base_dir, 'src/algos/llmat/llmat_toy.py')
llmat_toy = import_module_from_file('llmat_toy', llmat_toy_path)
LLMAT_Toy = llmat_toy.LLMAT_Toy

# Import llmat_bo_toy
llmat_bo_toy_path = os.path.join(base_dir, 'src/algos/llmat/llmat_bo_toy.py')
llmat_bo_toy = import_module_from_file('llmat_bo_toy', llmat_bo_toy_path)
ToyExpRunner = llmat_bo_toy.ToyExpRunner
FTDataset_Toy = llmat_bo_toy.FTDataset_Toy
select_from_leaf_to_root = llmat_bo_toy.select_from_leaf_to_root


class BOVisualizer:
    """Visualizer for Bayesian Optimization process"""

    def __init__(self, args, toy_bench, save_dir="visualizations"):
        self.args = args
        self.toy_bench = toy_bench
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Store history
        self.history = {'selected_points': [], 'selected_clusters': [], 'node_boundaries': [], 'acquisition_values': [], 'model_predictions': [], 'iteration': []}

    def setup_bo(self):
        """Initialize BO components"""
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        head_cfg = self.args.fix_args["head"]
        feature_dim = len(self.toy_bench.dataset['features'].iloc[0])

        # Initialize toy runner
        self.toy_runner = ToyExpRunner(self.toy_bench, seed, self.args.normalize_y)
        self.dataset_train = self.toy_runner.generate_initialization(self.args.n_init_data)

        # Get dataset info
        self.target_col = self.toy_bench.target_col
        self.target_col_transformed = self.toy_bench.target_col_transformed
        self.cluster_col = self.toy_bench._get_cluster_col()

        # Initialize Alpha model
        self.alpha_model = LLMAT_Toy(
            self.dataset_train,
            head_cfg,
            'cpu',  # Use CPU for visualization
            gamma=self.args.gamma,
            acf_type=self.args.acqf,
            lmbda=self.args.lmbda,
            eta=self.args.eta,
            tree_depth=self.args.tree_depth,
            threshold=self.args.threshold,
            alpha=self.args.alpha,
            feature_dim=feature_dim,
            target_col=self.target_col_transformed,
        )

        # Train model (builds tree structure)
        self.alpha_model.train_model()

    def get_decision_boundaries(self, x_range, node, path):
        """Get decision boundaries for a given node"""
        # Create a grid of points
        x_grid = np.linspace(x_range[0], x_range[1], 200)

        # For 1D, features are just the x values
        features_grid = torch.tensor([[x] for x in x_grid], dtype=torch.float32)

        # Get model predictions
        self.alpha_model.clf_head.eval()
        self.alpha_model.clf_head.load_state_dict(node.clf)

        with torch.no_grad():
            clf_logits = self.alpha_model.clf_head(features_grid).squeeze()

        # Calculate acquisition function
        tau = node.tau if hasattr(node, 'tau') else np.quantile(self.alpha_model.training_set[self.target_col], q=1 - self.alpha_model.gamma)

        # Get classification probability
        clf_prob = torch.sigmoid(clf_logits)
        # print("==========pred sigmoids============")
        # print(cx)
        acq_vals = (clf_prob / (1 - clf_prob))

        return x_grid, clf_logits.numpy(), acq_vals.numpy(), tau

    def visualize_iteration(self, iteration, selected_point, selected_cluster, path, node_acqfs_all, dataset):
        """Visualize a single BO iteration"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # 2x2 layout: True Func, Progress, Decision Boundary, Acquisition
        fig.suptitle(f'Iteration {iteration} - Selected Point: {selected_point:.4f}, Cluster: {selected_cluster}', fontsize=16, fontweight='bold')

        # Get x range from dataset
        all_x = [row['features'][0] for _, row in self.toy_bench.dataset.iterrows()]
        all_x.extend([row['features'][0] for _, row in self.dataset_train.iterrows()])
        x_min, x_max = min(all_x), max(all_x)

        # Create function object to get bounds
        if 'levy' in self.toy_bench.data_name:
            from benchmarks.TOY.toy_benchmarks import Levy
            func = Levy(dims=self.toy_bench.dims)
        elif 'ackley' in self.toy_bench.data_name:
            from benchmarks.TOY.toy_benchmarks import Ackley
            func = Ackley(dims=self.toy_bench.dims)
        else:
            func = None

        # Clip x_range to function bounds if available
        if func is not None:
            x_min = max(x_min - 0.1 * (x_max - x_min), func.lb[0])
            x_max = min(x_max + 0.1 * (x_max - x_min), func.ub[0])
        else:
            x_min = x_min - 0.1 * (x_max - x_min)
            x_max = x_max + 0.1 * (x_max - x_min)

        x_range = [x_min, x_max]

        # Get decision boundaries for the root node
        root_node = path[0][0]
        x_grid_root, clf_logits_root, acq_vals_root, tau0 = self.get_decision_boundaries(x_range, root_node, path)

        # Get decision boundaries for the leaf node
        leaf_node = path[-1][0]
        x_grid, clf_logits, acq_vals, tau = self.get_decision_boundaries(x_range, leaf_node, path)

        # Plot 1: True function with observed points and clusters
        ax1 = axes[0, 0]

        # Plot true function
        if func is not None:
            true_y = [func(np.array([x])) for x in x_grid]
        else:
            # Fallback: interpolate from dataset
            dataset_x = [row['features'][0] for _, row in self.toy_bench.dataset.iterrows()]
            dataset_y = self.toy_bench.dataset[self.target_col].values
            true_y = np.interp(x_grid, dataset_x, dataset_y)

        ax1.plot(x_grid, true_y, 'k-', linewidth=2, alpha=0.3, label='True Function')

        # Find and mark the true optimum with a vertical line
        true_y_array = np.array(true_y)
        if self.toy_bench.maximization:
            opt_idx = np.argmax(true_y_array)
            opt_label = 'True Maximum'
        else:
            opt_idx = np.argmin(true_y_array)
            opt_label = 'True Minimum'
        opt_x = x_grid[opt_idx]
        opt_y = true_y_array[opt_idx]
        ax1.axvline(x=opt_x, color='green', linestyle='--', linewidth=2.5, label=opt_label, zorder=9, alpha=0.7)

        # Plot ALL observed points including current training set
        obs_x = [row['features'][0] for _, row in self.alpha_model.training_set.iterrows()]
        # Use original target column for visualization (not transformed)
        obs_y = self.alpha_model.training_set[self.toy_bench.target_col].values
        obs_clusters = self.alpha_model.training_set[self.cluster_col].values

        scatter = ax1.scatter(obs_x, obs_y, c=obs_clusters, cmap='tab10', s=100, edgecolors='black', linewidths=2, label='Observed Points', zorder=5, alpha=0.7)

        # Highlight recent selections (last 10)
        recent_n = min(10, len(self.history['selected_points']))
        if recent_n > 0:
            recent_points = self.history['selected_points'][-recent_n:]
            recent_x = [pt['x'] for pt in recent_points]
            recent_y = [pt['y'] for pt in recent_points]
            ax1.scatter(recent_x, recent_y, c='orange', s=200, marker='o', edgecolors='red', linewidths=3, label=f'Last {recent_n} Selected', zorder=8, alpha=0.9)

        # Highlight the newly selected point
        new_x = selected_point
        # Use original target column for visualization (not transformed)
        new_y = self.alpha_model.training_set.iloc[-1][self.toy_bench.target_col]
        ax1.scatter([new_x], [new_y], c='red', s=400, marker='*', edgecolors='darkred', linewidths=3, label='Current Selection', zorder=10)

        # Show root node's best candidate
        root_node = path[0][0]
        if root_node.id in node_acqfs_all and len(node_acqfs_all[root_node.id][2]) > 0:
            # Get candidates and acquisition values for root node
            candidates_idx = node_acqfs_all[root_node.id][2]
            acq_vals_node = torch.cat(node_acqfs_all[root_node.id][3], dim=0).cpu().squeeze()

            # Find best candidate for root node
            best_idx = torch.argmax(acq_vals_node).item()
            best_candidate_idx = candidates_idx[best_idx]

            # Get x value from dataset
            candidate_row = dataset[dataset['Entry Number'] == best_candidate_idx].iloc[0]
            root_candidate_x = candidate_row['features'][0]

            # Draw vertical line for root node's candidate
            ax1.axvline(x=root_candidate_x, color='blue', linestyle=':', linewidth=2.5, label='Root Node Candidate', zorder=8, alpha=0.7)

        # Note: tau is on transformed scale, convert back if needed
        # For visualization, we show original (non-transformed) values
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('f(x) (original scale)', fontsize=12)

        # Add problem type to title
        problem_type = "Minimization" if not self.toy_bench.maximization else "Maximization"
        ax1.set_title(f'True Function & Observed Points (Total: {len(obs_x)}) - {problem_type}', fontsize=14)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Cluster')

        # Plot 2: Combined Decision Boundaries (Root + Leaf on same plot)
        ax2 = axes[1, 0]

        # Calculate probabilities for both nodes
        self.alpha_model.clf_head.load_state_dict(root_node.clf)
        prob_good_root = torch.sigmoid(torch.tensor(clf_logits_root)).numpy()

        leaf_node = path[-1][0]
        self.alpha_model.clf_head.load_state_dict(leaf_node.clf)
        prob_good_leaf = torch.sigmoid(torch.tensor(clf_logits)).numpy()

        # Plot probabilities on primary axis
        ax2.plot(x_grid, prob_good_root, 'b-', linewidth=2, label=f'P(good) - Root (Node {root_node.id})', alpha=0.8)
        ax2.plot(x_grid, prob_good_leaf, 'r-', linewidth=2, label=f'P(good) - Leaf (Node {leaf_node.id})', alpha=0.8)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold')

        # Mark good region based on leaf node prediction
        good_mask = prob_good_leaf > 0.5
        if np.any(good_mask):
            ax2.fill_between(x_grid, 0, prob_good_leaf, where=good_mask, alpha=0.2, color='green', label='Good Region (Leaf)')

        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_ylim([-0.05, 1.05])

        # Plot true function and data points on secondary axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_grid, true_y, 'k--', linewidth=1, alpha=0.3, label='True Function')

        # Mark true optimum with vertical line
        ax2.axvline(x=opt_x, color='green', linestyle='--', linewidth=2, label=opt_label, zorder=9, alpha=0.7)

        # Mark observed points on f(x) axis
        ax2_twin.scatter(obs_x, obs_y, c=obs_clusters, cmap='tab10', s=100, edgecolors='black', linewidths=2, zorder=5, alpha=0.7, label='Observed Points')

        # Mark each node's best candidate
        for node, _ in path:
            if node.id in node_acqfs_all and len(node_acqfs_all[node.id][2]) > 0:
                # Get candidates and acquisition values for this node
                candidates_idx = node_acqfs_all[node.id][2]
                acq_vals_node = torch.cat(node_acqfs_all[node.id][3], dim=0).cpu().squeeze()

                # Find best candidate for this node
                best_idx = torch.argmax(acq_vals_node).item()
                best_candidate_idx = candidates_idx[best_idx]

                # Get x value from dataset
                candidate_row = dataset[dataset['Entry Number'] == best_candidate_idx].iloc[0]
                best_candidate_x = candidate_row['features'][0]

                # Get true y value on the curve by interpolation
                best_candidate_y_true = np.interp(best_candidate_x, x_grid, true_y)

                # Use different markers for different nodes - plot on f(x) curve
                if node.id == root_node.id:
                    ax2_twin.scatter([best_candidate_x], [best_candidate_y_true], c='blue', s=120, marker='D', edgecolors='darkblue', linewidths=1.5, zorder=8,
                                     label=f'Root Candidate', alpha=0.8)
                elif node.id == leaf_node.id:
                    ax2_twin.scatter([best_candidate_x], [best_candidate_y_true], c='red', s=120, marker='D', edgecolors='darkred', linewidths=1.5, zorder=8,
                                     label=f'Leaf Candidate', alpha=0.8)

        # Mark final selected point on f(x) axis
        ax2_twin.scatter([new_x], [new_y], c='red', s=250, marker='*', edgecolors='darkred', linewidths=3, zorder=10, label='Final Selection')

        ax2_twin.set_ylabel('f(x) (original scale)', fontsize=12)

        ax2.set_xlabel('x', fontsize=12)
        ax2.set_title(f'Decision Boundaries (Root & Leaf)\nP(f_transformed(x) ≥ τ_transformed)', fontsize=14)

        # Combine legends from both axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')

        ax2.grid(True, alpha=0.3)

        # Plot 3: Combined Acquisition Functions (Root + Leaf on same plot)
        ax3 = axes[1, 1]

        # Plot acquisition functions on primary axis
        ax3.plot(x_grid, acq_vals_root, 'b-', linewidth=2, label=f'Acq - Root (Node {root_node.id})', alpha=0.8)
        ax3.plot(x_grid, acq_vals, 'r-', linewidth=2, label=f'Acq - Leaf (Node {leaf_node.id})', alpha=0.8)

        ax3.set_ylabel('Acquisition Value', fontsize=12)

        # Plot true function and data points on secondary axis
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x_grid, true_y, 'k--', linewidth=1, alpha=0.3, label='True Function')

        # Mark true optimum with vertical line
        ax3.axvline(x=opt_x, color='green', linestyle='--', linewidth=2, label=opt_label, zorder=9, alpha=0.7)

        # Mark observed points on f(x) axis
        ax3_twin.scatter(obs_x, obs_y, c=obs_clusters, cmap='tab10', s=100, edgecolors='black', linewidths=2, zorder=5, alpha=0.7, label='Observed Points')

        # Mark each node's best candidate
        for node, _ in path:
            if node.id in node_acqfs_all and len(node_acqfs_all[node.id][2]) > 0:
                # Get candidates and acquisition values for this node
                candidates_idx = node_acqfs_all[node.id][2]
                acq_vals_node = torch.cat(node_acqfs_all[node.id][3], dim=0).cpu().squeeze()

                # Find best candidate for this node
                best_idx = torch.argmax(acq_vals_node).item()
                best_candidate_idx = candidates_idx[best_idx]

                # Get x value from dataset
                candidate_row = dataset[dataset['Entry Number'] == best_candidate_idx].iloc[0]
                best_candidate_x = candidate_row['features'][0]

                # Get true y value on the curve by interpolation
                best_candidate_y_true = np.interp(best_candidate_x, x_grid, true_y)

                # Use different markers for different nodes - plot on f(x) curve
                if node.id == root_node.id:
                    ax3_twin.scatter([best_candidate_x], [best_candidate_y_true], c='blue', s=120, marker='D', edgecolors='darkblue', linewidths=1.5, zorder=8,
                                     label=f'Root Candidate', alpha=0.8)
                elif node.id == leaf_node.id:
                    ax3_twin.scatter([best_candidate_x], [best_candidate_y_true], c='red', s=120, marker='D', edgecolors='darkred', linewidths=1.5, zorder=8,
                                     label=f'Leaf Candidate', alpha=0.8)

        # Mark final selected point on f(x) axis
        ax3_twin.scatter([new_x], [new_y], c='red', s=250, marker='*', edgecolors='darkred', linewidths=3, zorder=10, label='Final Selection')

        ax3_twin.set_ylabel('f(x) (original scale)', fontsize=12)

        ax3.set_xlabel('x', fontsize=12)
        ax3.set_title(f'Acquisition Functions (Root & Leaf)\n({self.args.acqf.upper()})', fontsize=14)

        # Combine legends from both axes
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')

        ax3.grid(True, alpha=0.3)

        # Plot 6: Optimization Progress (Best value over iterations)
        ax6 = axes[0, 1]
        if len(self.history['selected_points']) > 0:
            iterations_so_far = list(range(1, len(self.history['selected_points']) + 1))
            selected_values = [pt['y'] for pt in self.history['selected_points']]

            # Calculate cumulative best (minimum for minimization, maximum for maximization)
            if self.toy_bench.maximization:
                cumulative_best = np.maximum.accumulate(selected_values)
                optimization_dir = "Maximization"
            else:
                cumulative_best = np.minimum.accumulate(selected_values)
                optimization_dir = "Minimization"

            ax6.plot(iterations_so_far, selected_values, 'o-', color='lightblue', linewidth=1, markersize=6, label='Selected Values', alpha=0.6)
            ax6.plot(iterations_so_far, cumulative_best, 'r-', linewidth=2.5, label='Best So Far', zorder=10)

            # Mark ground truth optimum
            ax6.axhline(y=self.toy_bench.ground_truth_opt, color='green', linestyle='--', linewidth=2, label=f'True Optimum ({self.toy_bench.ground_truth_opt:.3f})')

            # Calculate gap to optimum
            current_best = cumulative_best[-1]
            if self.toy_bench.maximization:
                gap = self.toy_bench.ground_truth_opt - current_best
            else:
                gap = current_best - self.toy_bench.ground_truth_opt

            ax6.set_xlabel('Iteration', fontsize=12)
            ax6.set_ylabel('f(x) (original scale)', fontsize=12)
            ax6.set_title(f'Optimization Progress - {optimization_dir} (Gap: {gap:.4f})', fontsize=14)
            ax6.legend(fontsize=10)
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        # Plot 6: Optimization Progress (Cumulative Best and Selected Values)

        # ax6 = axes[0, 1]
        # if len(self.history['selected_points']) > 0:
        #     iterations_so_far = list(range(1, len(self.history['selected_points']) + 1))
        #     selected_values = [pt['y'] for pt in self.history['selected_points']]

        #     if self.toy_bench.maximization:
        #         cumulative_best = np.maximum.accumulate(selected_values)
        #         optimization_dir = "Maximization"
        #     else:
        #         cumulative_best = np.minimum.accumulate(selected_values)
        #         optimization_dir = "Minimization"

        #     # 1. Plot CUMULATIVE BEST on the primary axis
        #     ax6.plot(iterations_so_far, cumulative_best, 'r-', linewidth=2.5, label='Best So Far', zorder=10)
        #     ax6.axhline(y=self.toy_bench.ground_truth_opt, color='green', linestyle='--', linewidth=2, label=f'True Optimum ({self.toy_bench.ground_truth_opt:.3f})')
        #     ax6.set_ylabel('Best So Far f(x)', fontsize=12, color='r')
        #     ax6.tick_params(axis='y', labelcolor='r')

        #     # 2. Create the twin axis for the SELECTED VALUES
        #     ax6_twin = ax6.twinx()
        #     ax6_twin.plot(iterations_so_far, selected_values, 'o-', color='lightblue', linewidth=1, markersize=6, label='Selected Values', alpha=0.8)
        #     ax6_twin.set_ylabel('Selected Value f(x)', fontsize=12, color='b')
        #     ax6_twin.tick_params(axis='y', labelcolor='b')

        #     # Combine legends
        #     lines, labels = ax6.get_legend_handles_labels()
        #     lines2, labels2 = ax6_twin.get_legend_handles_labels()
        #     ax6.legend(lines + lines2, labels + labels2, fontsize=9, loc='best')

        #     current_best = cumulative_best[-1]
        #     if self.toy_bench.maximization:
        #         gap = self.toy_bench.ground_truth_opt - current_best
        #     else:
        #         gap = current_best - self.toy_bench.ground_truth_opt

        #     ax6.set_xlabel('Iteration', fontsize=12)
        #     ax6.set_title(f'Optimization Progress - {optimization_dir} (Gap: {gap:.4f})', fontsize=14)
        #     ax6.grid(True, alpha=0.3)
        # else:
        #     ax6.set_title('Optimization Progress (No data yet)', fontsize=14)
        #     ax6.set_xlabel('Iteration', fontsize=12)
        #     ax6.set_ylabel('f(x) (original scale)', fontsize=12)

        # plt.tight_layout()

        # Save figure
        save_path = os.path.join(self.save_dir, f'iteration_{iteration:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {save_path}")

    def visualize_cluster_selection_summary(self):
        """Create a summary plot showing cluster selection over iterations"""
        if len(self.history['iteration']) == 0:
            print("No history to visualize")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        iterations = self.history['iteration']

        # Plot 1: Selected clusters over time
        ax1 = axes[0]
        selected_clusters = self.history['selected_clusters']

        for i, (iter_num, clusters) in enumerate(zip(iterations, selected_clusters)):
            for cluster in clusters:
                ax1.scatter(iter_num, cluster, s=100, alpha=0.6)

        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Cluster ID', fontsize=12)
        ax1.set_title('Clusters Selected for Evaluation at Each Iteration', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Selected point positions over time
        ax2 = axes[1]
        selected_points = [pt['x'] for pt in self.history['selected_points']]
        selected_values = [pt['y'] for pt in self.history['selected_points']]

        scatter = ax2.scatter(iterations, selected_points, c=selected_values, cmap='viridis', s=150, edgecolors='black', linewidths=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Selected x Position', fontsize=12)
        ax2.set_title('Selected Point Positions Over Iterations (colored by f(x))', fontsize=14)
        plt.colorbar(scatter, ax=ax2, label='f(x)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'cluster_selection_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved summary to {save_path}")

    def visualize_node_details(self, iteration, path, x_range, dataset, node_acqfs_all):
        """
        Visualize decision boundary and acquisition function for each node in the path
        Shows Node 0 and Node 1 (and more if available) in separate subplots
        """
        import torch.nn as nn

        # Limit to first 2 nodes for clarity (Node 0 and Node 1)
        nodes_to_show = min(2, len(path))

        fig, axes = plt.subplots(nodes_to_show, 2, figsize=(16, 4 * nodes_to_show))
        if nodes_to_show == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Iteration {iteration} - Node-Level Analysis', fontsize=16, fontweight='bold')

        # Create x grid
        x_min, x_max = x_range
        print(f"DEBUG: x_range for node details = [{x_min}, {x_max}]")
        x_grid = np.linspace(x_min, x_max, 200)
        features_grid = torch.tensor([[x] for x in x_grid], dtype=torch.float32)

        # Get true function values
        func = self.toy_bench.func if hasattr(self.toy_bench, 'func') else None
        if func is not None:
            true_y = np.array([func(np.array([x])) for x in x_grid])
        else:
            dataset_x = sorted(dataset['features'].apply(lambda f: f[0]).values)
            dataset_y = [dataset[dataset['features'].apply(lambda f: f[0]) == x][self.toy_bench.target_col].values[0] for x in dataset_x]
            true_y = np.interp(x_grid, dataset_x, dataset_y)

        for idx in range(nodes_to_show):
            node, choice = path[idx]

            # Left plot: Decision Boundary
            ax_clf = axes[idx, 0]

            # Get this node's classifier
            try:
                clf_temp = nn.Sequential(nn.Linear(self.alpha_model.feature_dim, self.alpha_model.last_hidden_size), nn.ReLU(), nn.Linear(self.alpha_model.last_hidden_size, 1))
                if hasattr(node, 'clf') and node.clf is not None:
                    clf_temp.load_state_dict(node.clf)
                    clf_temp.eval()

                    with torch.no_grad():
                        node_clf_logits = clf_temp(features_grid).squeeze()
                        node_prob = torch.sigmoid(node_clf_logits).numpy()

                    # Plot decision boundary on primary axis
                    ax_clf.plot(x_grid, node_prob, 'b-', linewidth=2, label='P(good)')
                    ax_clf.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                    ax_clf.fill_between(x_grid, 0, node_prob, where=(node_prob > 0.5), alpha=0.3, color='blue', label='Good Region')
                    ax_clf.set_ylabel('P(good)', fontsize=12)
                    ax_clf.set_ylim([-0.05, 1.05])

                    # Plot true function and data points on secondary axis
                    ax_clf_twin = ax_clf.twinx()
                    ax_clf_twin.plot(x_grid, true_y, 'k--', linewidth=1, alpha=0.3, label='True Function')

                    # Mark true optimum with vertical line
                    true_y_array = np.array(true_y)
                    if self.toy_bench.maximization:
                        opt_idx = np.argmax(true_y_array)
                        opt_label = 'True Max'
                    else:
                        opt_idx = np.argmin(true_y_array)
                        opt_label = 'True Min'
                    opt_x = x_grid[opt_idx]
                    opt_y = true_y_array[opt_idx]
                    ax_clf.axvline(x=opt_x, color='green', linestyle='--', linewidth=2, label=opt_label, zorder=9, alpha=0.7)

                    # Get observed points
                    obs_x = [row['features'][0] for _, row in self.alpha_model.training_set.iterrows()]
                    obs_y = self.alpha_model.training_set[self.toy_bench.target_col].values
                    obs_clusters = self.alpha_model.training_set[self.cluster_col].values if self.cluster_col in self.alpha_model.training_set.columns else None

                    # Plot observed points
                    if obs_clusters is not None:
                        ax_clf_twin.scatter(obs_x, obs_y, c=obs_clusters, cmap='tab10', s=80, edgecolors='black', linewidths=1.5, zorder=5, alpha=0.6, label='Observed')
                    else:
                        ax_clf_twin.scatter(obs_x, obs_y, c='gray', s=80, edgecolors='black', linewidths=1.5, zorder=5, alpha=0.6, label='Observed')

                    # Plot node's best candidate or final selection based on node type
                    if node.id in node_acqfs_all and len(node_acqfs_all[node.id][2]) > 0:
                        # Get candidates and acquisition values for this node
                        candidates_idx = node_acqfs_all[node.id][2]
                        acq_vals_node = torch.cat(node_acqfs_all[node.id][3], dim=0).cpu().squeeze()

                        # Find best candidate for this node
                        best_idx = torch.argmax(acq_vals_node).item()
                        best_candidate_idx = candidates_idx[best_idx]

                        # Get x value from dataset
                        candidate_row = dataset[dataset['Entry Number'] == best_candidate_idx].iloc[0]
                        best_candidate_x = candidate_row['features'][0]

                        # Get true y value on the curve by interpolation
                        best_candidate_y_true = np.interp(best_candidate_x, x_grid, true_y)

                        # For leaf node, show final selected point; for others, show their best candidate
                        is_leaf = idx == len(path) - 1
                        if is_leaf and len(self.history['selected_points']) > 0:
                            # Show final selected point on leaf node
                            recent_pt = self.history['selected_points'][-1]
                            ax_clf_twin.scatter([recent_pt['x']], [recent_pt['y']], c='red', s=200, marker='*', edgecolors='darkred', linewidths=2, zorder=10,
                                                label='Final Selection')
                        else:
                            # Show this node's best candidate on f(x) curve
                            ax_clf_twin.scatter([best_candidate_x], [best_candidate_y_true], c='blue', s=100, marker='D', edgecolors='darkblue', linewidths=1.5, zorder=9,
                                                label='Node Candidate', alpha=0.8)

                    ax_clf_twin.set_ylabel('f(x) (original scale)', fontsize=12)

                    ax_clf.set_xlabel('x', fontsize=12)

                    # Set axis limits
                    y_margin = (true_y.max() - true_y.min()) * 0.1
                    # ax_clf.set_ylim([true_y.min() - y_margin, true_y.max() + y_margin])
                    ax_clf.set_xlim([x_min, x_max])  # Explicitly set x-axis limits

                    node_label = f"Node {node.id}"
                    if idx == 0:
                        node_label += " (Root)"
                    elif idx == len(path) - 1:
                        node_label += " (Leaf)"
                    else:
                        node_label += f" ({'Left' if choice == 0 else 'Right'} child)"

                    ax_clf.set_title(f'{node_label} - Decision Boundary', fontsize=14, fontweight='bold')

                    # Combine legends from both axes
                    lines1, labels1 = ax_clf.get_legend_handles_labels()
                    lines2, labels2 = ax_clf_twin.get_legend_handles_labels()
                    ax_clf.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

                    ax_clf.grid(True, alpha=0.3)

            except Exception as e:
                print(f"Warning: Could not plot decision boundary for node {node.id}: {e}")
                ax_clf.text(0.5, 0.5, f'Node {node.id}\nNo classifier available', ha='center', va='center', transform=ax_clf.transAxes)

            # Right plot: Acquisition Function
            ax_acq = axes[idx, 1]

            try:
                # Calculate acquisition function for this node
                with torch.no_grad():
                    # Get regression predictions
                    reg_preds = self.alpha_model.reg_head(features_grid).squeeze()

                    # Get tau for this node
                    tau = node.tau if hasattr(node, 'tau') else np.quantile(self.alpha_model.training_set[self.target_col], q=1 - self.alpha_model.gamma)

                    # Get classification probability (already inverted above)
                    clf_prob = torch.tensor(node_prob, dtype=torch.float32)

                    acq_vals = (clf_prob / (1 - clf_prob))

                    acq_vals = acq_vals.numpy()

                # Plot acquisition function on primary axis
                ax_acq.plot(x_grid, acq_vals, 'r-', linewidth=2, label='Acquisition')
                ax_acq.fill_between(x_grid, 0, acq_vals, where=(acq_vals > acq_vals.mean()), alpha=0.3, color='red', label='High Acquisition')
                ax_acq.set_ylabel('Acquisition Value', fontsize=12)

                # Plot true function and data points on secondary axis
                ax_acq_twin = ax_acq.twinx()
                ax_acq_twin.plot(x_grid, true_y, 'k--', linewidth=1, alpha=0.3, label='True Function')

                # Mark true optimum with vertical line (reuse from decision boundary plot)
                ax_acq.axvline(x=opt_x, color='green', linestyle='--', linewidth=2, label=opt_label, zorder=9, alpha=0.7)

                # Plot observed points
                if obs_clusters is not None:
                    ax_acq_twin.scatter(obs_x, obs_y, c=obs_clusters, cmap='tab10', s=80, edgecolors='black', linewidths=1.5, zorder=5, alpha=0.6, label='Observed')
                else:
                    ax_acq_twin.scatter(obs_x, obs_y, c='gray', s=80, edgecolors='black', linewidths=1.5, zorder=5, alpha=0.6, label='Observed')

                # Plot node's best candidate or final selection based on node type (reuse from decision boundary)
                if node.id in node_acqfs_all and len(node_acqfs_all[node.id][2]) > 0:
                    # For leaf node, show final selected point; for others, show their best candidate
                    is_leaf = idx == len(path) - 1
                    if is_leaf and len(self.history['selected_points']) > 0:
                        # Show final selected point on leaf node
                        recent_pt = self.history['selected_points'][-1]
                        ax_acq_twin.scatter([recent_pt['x']], [recent_pt['y']], c='red', s=200, marker='*', edgecolors='darkred', linewidths=2, zorder=10, label='Final Selection')
                    else:
                        # Show this node's best candidate on f(x) curve (reuse values from decision boundary section)
                        ax_acq_twin.scatter([best_candidate_x], [best_candidate_y_true], c='blue', s=100, marker='D', edgecolors='darkblue', linewidths=1.5, zorder=9,
                                            label='Node Candidate', alpha=0.8)

                ax_acq_twin.set_ylabel('f(x) (original scale)', fontsize=12)

                ax_acq.set_xlabel('x', fontsize=12)

                # Set axis limits
                ax_acq.set_xlim([x_min, x_max])  # Explicitly set x-axis limits

                # Set acquisition axis limits with some margin
                acq_margin = (acq_vals.max() - acq_vals.min()) * 0.1
                ax_acq.set_ylim([max(0, acq_vals.min() - acq_margin), acq_vals.max() + acq_margin])

                # Set f(x) axis limits on secondary axis
                ax_acq_twin.set_ylim([true_y.min() - y_margin, true_y.max() + y_margin])

                ax_acq.set_title(f'{node_label} - Acquisition Function ({self.args.acqf.upper()})', fontsize=14, fontweight='bold')

                # Combine legends from both axes
                lines1, labels1 = ax_acq.get_legend_handles_labels()
                lines2, labels2 = ax_acq_twin.get_legend_handles_labels()
                ax_acq.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

                ax_acq.grid(True, alpha=0.3)

            except Exception as e:
                print(f"Warning: Could not plot acquisition for node {node.id}: {e}")
                ax_acq.text(0.5, 0.5, f'Node {node.id}\nAcquisition calculation failed', ha='center', va='center', transform=ax_acq.transAxes)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'node_details_iter_{iteration}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved node details to {save_path}")

    def visualize_root_and_leaf_acqf(self, iteration, path, x_range, dataset, node_acqfs_all):
        """
        Visualize the Acquisition Function (AF) for the Root Node (Node 0) and the
        Leaf Node (final node in path) side-by-side.
        """
        if len(path) == 0:
            print("Error: Path is empty, cannot plot root/leaf.")
            return

        # Determine indices for the root and leaf nodes
        root_idx = 0
        leaf_idx = len(path) - 1

        # Nodes to plot: (index in path, node object, plot title modifier)
        nodes_to_plot = []
        # Always include the root node
        nodes_to_plot.append((root_idx, path[root_idx][0], "Root Node"))

        # Include the leaf node if it is different from the root node
        if leaf_idx != root_idx:
            nodes_to_plot.append((leaf_idx, path[leaf_idx][0], "Leaf Node"))

        # --- Setup Plot and Grid ---
        num_plots = len(nodes_to_plot)
        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 3))

        if num_plots == 1:
            axes = [axes]  # Ensure axes is iterable even for a single subplot

        # fig.suptitle(f'Iteration {iteration} - Root vs. Leaf Acquisition Function', fontsize=16, fontweight='bold')

        # Create x grid
        x_min, x_max = x_range
        x_grid = np.linspace(x_min, x_max, 200)
        features_grid = torch.tensor([[x] for x in x_grid], dtype=torch.float32)

        # Get true function values
        func = self.toy_bench.func if hasattr(self.toy_bench, 'func') else None
        if func is not None:
            true_y = np.array([func(np.array([x])) for x in x_grid])
        else:
            dataset_x = sorted(dataset['features'].apply(lambda f: f[0]).values)
            dataset_y = [dataset[dataset['features'].apply(lambda f: f[0]) == x][self.toy_bench.target_col].values[0] for x in dataset_x]
            true_y = np.interp(x_grid, dataset_x, dataset_y)

        # Calculate global optimum for annotation
        true_y_array = np.array(true_y)
        if self.toy_bench.maximization:
            opt_idx = np.argmax(true_y_array)
            opt_label = 'True Max'
        else:
            opt_idx = np.argmin(true_y_array)
            opt_label = 'True Min'
        opt_x = x_grid[opt_idx]

        # Get observed points
        obs_x = [row['features'][0] for _, row in self.alpha_model.training_set.iterrows()]
        obs_y = self.alpha_model.training_set[self.toy_bench.target_col].values
        # Assuming self.cluster_col is defined for coloring if needed
        obs_clusters = self.alpha_model.training_set.get(self.cluster_col)

        # Calculate f(x) y-axis margin for secondary axis
        y_margin = (true_y.max() - true_y.min()) * 0.1

        # --- Plotting Loop ---
        for plot_idx, (path_idx, node, title_modifier) in enumerate(nodes_to_plot):
            ax_acq = axes[plot_idx]
            node_id = node.id

            try:
                # 1. Get the classification probability P(good) for this node
                clf_temp = nn.Sequential(nn.Linear(self.alpha_model.feature_dim, self.alpha_model.last_hidden_size), nn.ReLU(), nn.Linear(self.alpha_model.last_hidden_size, 1))
                if hasattr(node, 'clf') and node.clf is not None:
                    clf_temp.load_state_dict(node.clf)
                    clf_temp.eval()
                    with torch.no_grad():
                        node_clf_logits = clf_temp(features_grid).squeeze()
                        node_prob = torch.sigmoid(node_clf_logits).numpy()
                else:
                    # Fallback if classifier is missing (should not happen for a decision node)
                    node_prob = np.full_like(x_grid, 0.5)

                # 2. Calculate Acquisition Function: AF = P(good) / (1 - P(good))
                with torch.no_grad():
                    clf_prob = torch.tensor(node_prob, dtype=torch.float32)
                    acq_vals = (clf_prob / (1 - clf_prob))
                    acq_vals = acq_vals.numpy()

                # Plot acquisition function on primary axis
                ax_acq.plot(x_grid, acq_vals, 'r-', linewidth=2, label='Acquisition')
                ax_acq.fill_between(x_grid, 0, acq_vals, where=(acq_vals > acq_vals.mean()), alpha=0.3, color='red', label='High Acquisition')
                ax_acq.set_ylabel('Acquisition Value', fontsize=12)

                # Plot true function and data points on secondary axis
                ax_acq_twin = ax_acq.twinx()
                ax_acq_twin.plot(x_grid, true_y, 'k--', linewidth=1, alpha=0.3, label='True Function')

                # Mark true optimum
                ax_acq.axvline(x=opt_x, color='green', linestyle='--', linewidth=2, label=opt_label, zorder=9, alpha=0.7)

                # Plot observed points
                if obs_clusters is not None:
                    ax_acq_twin.scatter(obs_x, obs_y, c=obs_clusters, cmap='tab10', s=80, edgecolors='black', linewidths=1.5, zorder=5, alpha=0.6, label='Observed')
                else:
                    ax_acq_twin.scatter(obs_x, obs_y, c='gray', s=80, edgecolors='black', linewidths=1.5, zorder=5, alpha=0.6, label='Observed')

                # Plot node's best candidate or final selection
                is_leaf = path_idx == leaf_idx
                if node_id in node_acqfs_all and len(node_acqfs_all[node_id][2]) > 0:
                    # Get candidates and acquisition values for this node
                    candidates_idx = node_acqfs_all[node_id][2]
                    acq_vals_node = torch.cat(node_acqfs_all[node_id][3], dim=0).cpu().squeeze()

                    # Find best candidate for this node
                    best_idx = torch.argmax(acq_vals_node).item()
                    best_candidate_idx = candidates_idx[best_idx]

                    # Get x value from dataset
                    candidate_row = dataset[dataset['Entry Number'] == best_candidate_idx].iloc[0]
                    best_candidate_x = candidate_row['features'][0]
                    best_candidate_y_true = np.interp(best_candidate_x, x_grid, true_y)

                    if is_leaf and len(self.history['selected_points']) > 0:
                        # Show final selected point on leaf node
                        recent_pt = self.history['selected_points'][-1]
                        ax_acq_twin.scatter([recent_pt['x']], [recent_pt['y']], c='red', s=200, marker='*', edgecolors='darkred', linewidths=2, zorder=10, label='Final Selection')
                    else:
                        # Show this node's best candidate
                        ax_acq_twin.scatter([best_candidate_x], [best_candidate_y_true], c='blue', s=100, marker='D', edgecolors='darkblue', linewidths=1.5, zorder=9,
                                            label='Node Candidate', alpha=0.8)

                ax_acq_twin.set_ylabel('f(x) (original scale)', fontsize=12)
                ax_acq.set_xlabel('x', fontsize=12)

                # Set axis limits
                ax_acq.set_xlim([x_min, x_max])
                acq_margin = (acq_vals.max() - acq_vals.min()) * 0.1
                ax_acq.set_ylim([max(0, acq_vals.min() - acq_margin), acq_vals.max() + acq_margin])
                ax_acq_twin.set_ylim([true_y.min() - y_margin, true_y.max() + y_margin])

                ax_acq.set_title(f'Iteration {iteration} - {title_modifier} (Node {node_id}) - AF ({self.args.acqf.upper()})', fontsize=14, fontweight='bold')

                # Combine legends
                lines1, labels1 = ax_acq.get_legend_handles_labels()
                lines2, labels2 = ax_acq_twin.get_legend_handles_labels()
                ax_acq.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
                ax_acq.grid(True, alpha=0.3)

            except Exception as e:
                print(f"Warning: Could not plot acquisition for node {node_id}: {e}")
                ax_acq.text(0.5, 0.5, f'{title_modifier} (Iteration {iteration}) - Node {node_id})\nAcquisition calculation failed', ha='center', va='center',
                            transform=ax_acq.transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        save_path = os.path.join(self.save_dir, f'root_leaf_acqf_iter_{iteration}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved root and leaf AF plots to {save_path}")

    def run_visualization(self, n_iterations=30, visualize_every=10):
        """Run BO and visualize every N iterations"""
        print(f"Running BO with visualization every {visualize_every} iterations...")

        best_y = self.dataset_train[self.target_col_transformed].max()

        for t in range(n_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {t+1}/{n_iterations}")
            print(f"{'='*60}")

            # Get clusters to evaluate
            clusters = self.alpha_model.get_clusters(pval=self.args.p_val, n_clusters=self.toy_bench.n_clusters)
            print(f"Selected clusters: {clusters}")

            # Filter dataset by clusters
            dataset = self.toy_bench.dataset
            dataset = dataset[dataset[self.cluster_col].isin(clusters)]

            # Select path through tree
            leaf, path = self.alpha_model.select()
            node_ids = [node.id for node, _ in path]

            # Initialize node_acqfs_all
            node_acqfs_all = {}
            for node_id in node_ids:
                node_acqfs_all[node_id] = [0, [], [], [], [], [], [], []]

            # Get acquisition values
            ft_dataset = FTDataset_Toy(dataset, self.cluster_col)
            dataloader = DataLoader(ft_dataset, batch_size=256, shuffle=False)

            for data in dataloader:
                node_acqfs = self.alpha_model.cal_acqfs(data, path)
                for node_id in node_ids:
                    if node_id not in node_acqfs:
                        continue
                    node_acqfs_all[node_id][0] = node_acqfs[node_id][0]
                    node_acqfs_all[node_id][1].append(node_acqfs[node_id][1])
                    node_acqfs_all[node_id][2] += node_acqfs[node_id][2]
                    node_acqfs_all[node_id][3] += [t.cpu() for t in node_acqfs[node_id][3]]
                    node_acqfs_all[node_id][4] += [t.cpu() for t in node_acqfs[node_id][4]]
                    node_acqfs_all[node_id][5] += [t.cpu() for t in node_acqfs[node_id][5]]
                    labels = torch.greater_equal(node_acqfs[node_id][4][0], node_acqfs[node_id][0])
                    node_acqfs_all[node_id][6] += [labels.cpu()]

            # Select candidate
            result = select_from_leaf_to_root(node_acqfs_all, node_ids.copy())

            if result is None:
                print("No candidates found, selecting randomly")
                random_idx = np.random.randint(len(dataset))
                index = dataset.index[random_idx]
                new_data = helpers.pop_df(self.toy_bench.dataset, index)
            else:
                candidates_idx, y_reg, y_preds, y_class, acq_vals = result
                idx_best = torch.argmax(acq_vals).item()
                candidates = dataset[dataset["Entry Number"].isin(candidates_idx)]

                index = dataset[dataset['Entry Number'] == candidates_idx[idx_best]].index[0]
                new_data = helpers.pop_df(self.toy_bench.dataset, index)

            # Update history
            selected_x = new_data['features'][0]
            # Store original (non-transformed) y value for visualization
            selected_y = new_data[self.toy_bench.target_col]
            selected_cluster = new_data[self.cluster_col]

            self.history['iteration'].append(t + 1)
            self.history['selected_points'].append({'x': selected_x, 'y': selected_y})
            self.history['selected_clusters'].append(clusters)
            self.history['node_boundaries'].append(path)

            # Update model
            self.alpha_model.condition_on_observations(new_data)

            # Update best y
            if new_data[self.target_col_transformed] > best_y:
                best_y = new_data[self.target_col_transformed]

            print(f"Selected: x={selected_x:.4f}, f(x)={selected_y:.4f}, cluster={selected_cluster}")
            print(f"Best so far: {best_y:.4f}")

            # Visualize if needed
            if (t + 1) % visualize_every == 0 or t == 0:
                self.visualize_iteration(t + 1, selected_x, selected_cluster, path, node_acqfs_all, dataset)

                # Also create node-specific visualization
                # Use function bounds instead of data range for better visualization
                if hasattr(self.toy_bench, 'bounds'):
                    x_range = [self.toy_bench.bounds[0][0], self.toy_bench.bounds[0][1]]
                else:
                    x_range = [dataset['features'].apply(lambda f: f[0]).min(), dataset['features'].apply(lambda f: f[0]).max()]
                self.visualize_node_details(t + 1, path, x_range, dataset, node_acqfs_all)
                self.visualize_root_and_leaf_acqf(t + 1, path, x_range, dataset, node_acqfs_all)

        # Create summary plots
        self.visualize_cluster_selection_summary()
        print(f"\nVisualization complete! All plots saved to {self.save_dir}/")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize toy BO experiments')
    parser.add_argument('--config', type=str, default='configs/llmat/llmat.yaml', help='Path to algorithm config file')
    parser.add_argument('--data_config', type=str, default='configs/data_config/data.yaml', help='Path to data config file')
    parser.add_argument('--n_iterations', type=int, default=30, help='Number of BO iterations to run')
    parser.add_argument('--visualize_every', type=int, default=10, help='Visualize every N iterations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='visualizations/toy_levy', help='Directory to save visualizations')
    parser.add_argument('--data_name', type=str, default=None, help='Toy function name (levy or ackley). If not specified, uses first from data config')
    parser.add_argument('--dims', type=int, default=1, help='Number of dimensions (default: 1 for visualization)')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples. If not specified, uses value from data config')

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
        args_dict = vars(args)
        for key, value in vars(config).items():
            if key not in args_dict or args_dict[key] is None:
                args_dict[key] = value
        args = argparse.Namespace(**args_dict)

    # Set defaults for toy benchmark
    args.benchmark = 'toy'
    args.algorithm = 'llmat'
    args.n_init_data = 10
    args.normalize_y = False
    args.threshold = 0.3
    args.alpha = 1.0
    args.p_val = 0.1

    # Load data config to get toy benchmark parameters
    with open(args.data_config) as f:
        data_config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Get toy benchmark parameters from config
    toy_config = data_config.benchmarks['toy']
    toy_dims = toy_config.get('toy_dims', 10)
    toy_n_samples = toy_config.get('toy_n_samples', 1000)
    toy_datasets = toy_config.get('datasets', ['levy'])
    toy_features = toy_config.get('feature_models', ['random_embedding'])

    # Use command line arguments if provided, otherwise use config defaults
    data_name = args.data_name if args.data_name else (toy_datasets[0] if toy_datasets else 'levy')
    feature_type = toy_features[0] if toy_features else 'random_embedding'
    dims = args.dims  # Use command line arg (default is 1 for visualization)
    n_samples = args.n_samples if args.n_samples else min(toy_n_samples, 200)  # Limit for faster visualization

    # Create toy benchmark
    from benchmarks.TOY.toy_benchmarks import ToyBench

    print(f"Loading {data_name.upper()} function...")
    print(f"  Dimensions: {dims}")
    print(f"  Samples: {n_samples}")
    print(f"  Feature type: {feature_type}")

    toy_bench = ToyBench(data_name=data_name, dims=dims, n_samples=n_samples, run_subset_only=False, finetuning=False, clustering_type='kmeans', randseed=args.seed,
                         feature_type=feature_type)

    print(f"Benchmark loaded: {toy_bench.dataset_name}")
    print(f"Dataset size: {len(toy_bench.dataset)}")
    print(f"Number of clusters: {toy_bench.n_clusters}")

    # Create visualizer
    visualizer = BOVisualizer(args, toy_bench, save_dir=args.save_dir)
    visualizer.setup_bo()

    # Run visualization
    visualizer.run_visualization(n_iterations=args.n_iterations, visualize_every=args.visualize_every)


if __name__ == "__main__":
    main()
