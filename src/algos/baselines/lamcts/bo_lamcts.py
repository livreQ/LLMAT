#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Thu Nov 27 16:39:22 2025
# File Name: bo_lamcts.py
# Description:
"""

import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from benchmarks.MAT.material_benchmarks import MATBench


class MCTSNode:

    def __init__(self, candidate_indices, parent=None, depth=0, feature_matrix=None):
        self.candidate_indices = np.array(candidate_indices)
        self.parent = parent
        self.depth = depth
        self.feature_matrix = feature_matrix  # Shared reference to full features

        # MCTS Stats
        self.visits = 0
        self.values = []
        self.mean_value = 0.0
        self.max_value = -float('inf')

        # Tree Structure
        self.children = []
        self.is_leaf = True
        self.split_classifier = None  # If using SVM

        # Tracking visited candidates in this specific node
        self.visited_indices = set()

    def get_ucb(self, total_visits, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')
        # UCB = Mean + Cp * sqrt(2 * ln(N) / n)
        return self.mean_value + exploration_weight * math.sqrt(2 * math.log(total_visits) / self.visits)

    def attempt_split(self, min_samples_split=10):
        """
        Dynamically splits the node into children if it has enough data.
        Strategy: K-Means (k=2) on the features of candidates in this node.
        """
        # 1. Check if we have enough candidates to split
        if len(self.candidate_indices) < min_samples_split:
            return False

        # 2. Get features for current pool
        current_features = self.feature_matrix[self.candidate_indices]

        # 3. Perform 2-Means clustering to define the split boundary
        # In discrete space, we can just assign indices directly to children
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(current_features)

        # 4. Create Children
        idx_left = self.candidate_indices[labels == 0]
        idx_right = self.candidate_indices[labels == 1]

        if len(idx_left) == 0 or len(idx_right) == 0:
            return False  # Failed to split efficiently

        left_child = MCTSNode(idx_left, parent=self, depth=self.depth + 1, feature_matrix=self.feature_matrix)
        right_child = MCTSNode(idx_right, parent=self, depth=self.depth + 1, feature_matrix=self.feature_matrix)

        self.children = [left_child, right_child]
        self.is_leaf = False
        return True

    def update(self, reward):
        self.visits += 1
        self.values.append(reward)
        self.max_value = max(self.max_value, reward)
        self.mean_value = np.mean(self.values)

    def sample_candidate(self):
        """
        Selects a random unvisited candidate from this node's pool.
        """
        # Efficient difference: available = pool - visited
        # We assume candidate_indices are unique
        available = [i for i in self.candidate_indices if i not in self.visited_indices]

        if not available:
            return None

        selected = np.random.choice(available)
        self.visited_indices.add(selected)
        return selected


class LaMCTS:

    def __init__(self, env, exploration_weight=1.0, min_samples_to_split=20):
        self.env = env
        self.cp = exploration_weight
        self.min_samples_to_split = min_samples_to_split
        self.history = []

        # Initialize Root
        # We need the raw features for splitting.
        # MATBench stores features in a flattened or tensor format depending on loading.
        # We normalize this to a numpy matrix.
        self.features = np.stack(self.env.dataset_ft["features"].values)
        all_indices = self.env.dataset_ft.index.values

        self.root = MCTSNode(candidate_indices=all_indices, feature_matrix=self.features)
        print(f"Initialized LaMCTS Root with {len(all_indices)} candidates.")

    def select_leaf(self):
        """
        Traverses the tree from root to a suitable leaf using UCB.
        """
        node = self.root

        while not node.is_leaf:
            # Select child with highest UCB
            best_score = -float('inf')
            best_child = None

            # Avoid log(0)
            total_visits = sum(c.visits for c in node.children)
            if total_visits == 0: total_visits = 1

            for child in node.children:
                score = child.get_ucb(total_visits, self.cp)
                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                # Should not happen unless children are empty
                return node

            node = best_child

        return node

    def search(self, budget=100):
        print(f"Starting LaMCTS Search (Budget: {budget})...")

        for i in range(budget):
            # 1. Selection
            leaf = self.select_leaf()

            # 2. Simulation / Sampling
            # If leaf is fully explored, we might need to handle it (backtrack or fail)
            # Simple version: just try to sample
            candidate_idx = leaf.sample_candidate()

            # If the leaf is exhausted, we could force a split or pick another node.
            # For this benchmark, we'll skip the iteration if empty (or could penalize node).
            if candidate_idx is None:
                print("Leaf exhausted, skipping step.")
                continue

            # 3. Evaluation
            # Use the query column (SMILES/IUPAC) to get the reward
            query_col = self.env.datasets_info[self.env.data_name][-1]
            candidate_val = self.env.dataset.loc[candidate_idx, query_col]

            reward, _ = self.env.complete_call(candidate_val)

            # Adjust reward for maximization context
            actual_reward = reward if self.env.maximization else -reward

            # 4. Backpropagation
            curr = leaf
            while curr is not None:
                curr.update(actual_reward)
                # Also mark this index as visited in parent nodes to avoid re-sampling
                # (Optional optimization, but strictly node-based visited sets are easier)
                curr = curr.parent

            # 5. Dynamic Split Check
            # If this leaf has been visited enough times, split it!
            if leaf.visits >= self.min_samples_to_split:
                success = leaf.attempt_split(min_samples_split=self.min_samples_to_split)
                if success:
                    print(f"Iter {i}: Split node at depth {leaf.depth} into 2 children.")

            # Logging
            self.history.append({
                'iter': i,
                'reward': actual_reward,
                'candidate': candidate_val,
                'depth': leaf.depth,
                'best_so_far': max([x['reward'] for x in self.history] + [actual_reward])
            })

            if i % 10 == 0:
                print(f"Iter {i}: Reward {actual_reward:.4f}, Depth {leaf.depth}, Best {self.history[-1]['best_so_far']:.4f}")

        return pd.DataFrame(self.history)


if __name__ == "__main__":
    # Configuration
    DATA_NAME = "solvation"
    FEATURE_TYPE = "molformer"
    BUDGET = 200

    # Initialize Environment
    # Note: We rely on MATBench loading the features into dataset_ft
    mat_bench = MATBench(
        data_name=DATA_NAME,
        run_subset_only=False,
        feature_type=FEATURE_TYPE,
        finetuning=False,
        iupac=False,
        prompt_type="just-smiles",
        randseed=42,
        clustering_type="kmeans",  # This is just for init, we ignore it in LaMCTS logic
        feature_reduction="average")

    # Initialize Agent
    # min_samples_to_split controls when a region is considered "interesting enough" to partition further
    agent = LaMCTS(mat_bench, exploration_weight=1.0, min_samples_to_split=20)

    # Run
    results = agent.search(budget=BUDGET)

    print("\nTop 5 Results:")
    print(results.sort_values(by='reward', ascending=False).head(5))
