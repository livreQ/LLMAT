#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.edu
# Created Time : 2025
# File Name: multibo_lapeft.py
# Description: Multi-objective Bayesian Optimization with Laplace-PEFT.
"""
import numpy as np
import pandas as pd
import torch
from src.utils import helpers
from src.algos.baselines.lapeft.multiobj_fix_feature import run_multiobj_fix_ft
# Assuming there will be a multi-objective finetuning version
# from src.algos.baselines.lapeft.multiobj_finetuning import run_multiobj_finetuning

device = helpers.check_device()
print(f"Using device: {device}")


class MultiObjMATExpRunner:
    """
    Experiment runner for multi-objective material discovery benchmarks.
    """

    def __init__(self, mat_bench, seed, list_init_points=None, uniform_cluster=True, normalize_y=False):
        self.mat_bench = mat_bench
        self.seed = seed
        self.list_init_points = list_init_points
        self.uniform_cluster = uniform_cluster
        self.normalize_y = normalize_y

    def generate_initialization(self, n_samples):
        """
        Generate initialization points for BO search, optionally uniformly distributed across clusters.
        
        Args:
            n_samples (int): Number of initialization points to generate.
            
        Returns:
            pd.DataFrame: A DataFrame containing the initialized points.
        """
        if self.list_init_points is not None:
            init_points = self.list_init_points[:n_samples]
        else:
            init_points = []
            dataset = self.mat_bench.dataset

            # Ensure n_clusters is up-to-date
            if self.mat_bench.n_clusters == 0:
                raise ValueError("Number of clusters in mat_bench is 0.")

            init_cluster_cnt = {cluster: 0 for cluster in range(self.mat_bench.n_clusters)}

            # Guard against infinite loop if n_samples is too large
            max_attempts = len(dataset) * 2
            attempts = 0

            while len(init_points) < n_samples and attempts < max_attempts:
                attempts += 1
                idx = np.random.randint(len(dataset))

                if self.uniform_cluster:
                    cluster_col = self.mat_bench._get_cluster_col()
                    if cluster_col not in dataset.columns:
                        raise ValueError(f"Cluster column '{cluster_col}' not found in dataset.")

                    cluster = dataset.loc[idx][cluster_col]

                    # Handle cases where cluster might not be in init_cluster_cnt
                    if cluster not in init_cluster_cnt:
                        continue

                    # Ensure fair sampling across clusters
                    if init_cluster_cnt[cluster] >= n_samples / self.mat_bench.n_clusters:
                        continue
                    else:
                        init_cluster_cnt[cluster] += 1

                init_points.append(helpers.pop_df(dataset, idx))

            if len(init_points) < n_samples:
                print(f"Warning: Could only generate {len(init_points)} of {n_samples} requested samples.")

            self.mat_bench.dataset = dataset

        return pd.DataFrame(init_points)

    def evaluate_point(self, candidate):
        """
        Evaluate a single point on the benchmark.
        """
        label, _ = self.mat_bench.complete_call(candidate)
        return candidate, label


def multibo_lapeft(args, mat_bench, wandb=None):
    """
    Entry point for multi-objective Bayesian optimization with LapeFT.
    """
    if args.benchmark == 'multiobj_mat':
        # Initialize the MATExpRunner
        mat_runner = MultiObjMATExpRunner(mat_bench, args.seed, None, True)
        trace_hypervolume, trace_scalarized_best_y = run_multiobj_fix_ft(args, mat_runner, wandb)
        return trace_hypervolume, trace_scalarized_best_y
    else:
        raise ValueError(f"Benchmark {args.benchmark} not supported for multi-objective Bayesian optimization.")
