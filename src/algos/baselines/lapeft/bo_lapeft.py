#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Sat Jul 27 14:22:59 2024
# File Name: lapeft_bo.py
# Description:
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src.utils import helpers
from src.algos.baselines.lapeft.finetuning import run_finetuning
from src.algos.baselines.lapeft.fix_feature import run_fix_ft

device = helpers.check_device()
print(f"Using device: {device}")


class MATExpRunner:

    def __init__(self, mat_bench, seed, normalize_y, list_init_points=None, uniform_cluster=True):
        self.mat_bench = mat_bench
        self.seed = seed
        self.list_init_points = list_init_points
        self.ground_truth_max_transformed = None
        self.normalize_y = normalize_y
        self.uniform_cluster = uniform_cluster

    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search uniformly distributed across clusters.
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''
        # assert isinstance(self.list_init_points, list)
        if self.list_init_points is not None:
            init_points = self.list_init_points[:n_samples]
        else:
            # if self.mat_bench.finetuning:
            init_points = []
            dataset = self.mat_bench.dataset
            target_col_transformed = self.mat_bench.target_col_transformed
            if self.normalize_y:
                y_preprocessor = StandardScaler()
                dataset[target_col_transformed] = y_preprocessor.fit_transform(dataset[target_col_transformed].to_numpy().reshape(-1, 1)).flatten()
            ground_truth_opt_id = dataset[target_col_transformed].idxmax()
            self.ground_truth_max_transformed = dataset.loc[ground_truth_opt_id][target_col_transformed]
            print("ground_truth_max", self.mat_bench.ground_truth_max)
            print("ground_truth_opt", self.mat_bench.ground_truth_opt)
            print("ground_truth_max_transformed", self.ground_truth_max_transformed)

            init_cluster_cnt = {cluster: 0 for cluster in range(self.mat_bench.n_clusters)}
            while len(init_points) < n_samples:
                idx = np.random.randint(len(dataset))
                # Make sure that the optimum is not included
                if dataset.loc[idx][self.mat_bench.target_col_transformed] >= self.ground_truth_max_transformed:
                    continue
                if self.uniform_cluster:
                    cluster = dataset.loc[idx][self.mat_bench._get_cluster_col()]
                    if init_cluster_cnt[cluster] >= n_samples / self.mat_bench.n_clusters:
                        continue
                    else:
                        init_cluster_cnt[cluster] += 1
                init_points.append(helpers.pop_df(dataset, idx))
            self.mat_bench.dataset = dataset

        return pd.DataFrame(init_points)

    def evaluate_point(self, candidate):
        '''
        Evaluate a single point on bbox
        '''

        label = self.mat_bench.complete_call(candidate)
        return candidate, label


def bo_lapeft(args, mat_bench, wandb=None):

    if args.benchmark == 'mat':
        # Initialize the MATExpRunner
        mat_runner = MATExpRunner(mat_bench, args.seed, args.normalize_y, None, True)
        if not mat_bench.finetuning:
            trace_best_y, all_metrics = run_fix_ft(args, mat_runner, wandb)
        else:
            trace_best_y, all_metrics = run_finetuning(args, mat_runner, wandb)

    return trace_best_y, all_metrics
