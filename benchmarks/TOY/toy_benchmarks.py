#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-25
# File Name: toy_benchmarks.py
# Description: Toy benchmarks using Levy and Ackley functions with same interface as material_benchmarks.py
"""
import os
import inspect
import pandas as pd
import torch
import numpy as np
import time
from sklearn.utils import shuffle as skshuffle
import sys
from sklearn.cluster import KMeans

current_file = inspect.getfile(inspect.currentframe())
CURRENT_DIR = os.path.dirname(os.path.abspath(current_file))

DATA_DIR_NAME = CURRENT_DIR


class tracker:
    """Simple tracker for function evaluations"""

    def __init__(self, name):
        self.name = name
        self.history = []

    def track(self, value):
        self.history.append(value)


class Levy:

    def __init__(self, dims=10):
        self.dims = dims
        self.lb = -10 * np.ones(dims)
        self.ub = 10 * np.ones(dims)
        self.tracker = tracker('Levy' + str(dims))

        #tunable hyper-parameters in LA-MCTS
        self.Cp = 10
        self.leaf_size = 8
        self.kernel_type = "poly"
        self.ninits = 40
        self.gamma_type = "auto"
        print("initialize levy at dims:", self.dims)

    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        w = []
        for idx in range(0, len(x)):
            w.append(1 + (x[idx] - 1) / 4)
        w = np.array(w)

        term1 = (np.sin(np.pi * w[0]))**2

        term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)

        term2 = 0
        for idx in range(1, len(w)):
            wi = w[idx]
            new = (wi - 1)**2 * (1 + 10 * (np.sin(np.pi * wi + 1))**2)
            term2 = term2 + new

        result = term1 + term2 + term3
        self.tracker.track(result)

        return result


class Ackley:

    def __init__(self, dims=10):
        self.dims = dims
        self.lb = -5 * np.ones(dims)
        self.ub = 10 * np.ones(dims)
        self.counter = 0
        self.tracker = tracker('Ackley' + str(dims))

        #tunable hyper-parameters in LA-MCTS
        self.Cp = 1
        self.leaf_size = 10
        self.ninits = 40
        self.kernel_type = "rbf"
        self.gamma_type = "auto"

    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
        self.tracker.track(result)

        return result


class ToyBench(object):
    """
    Toy benchmark using Levy and Ackley functions with same interface as MATBench
    For a given dataset, give x/z, query y
    """

    def __init__(self, data_name, run_subset_only=False, finetuning=False,\
                 iupac=False, prompt_type=None, randseed=666, feature_type=None,\
                 clustering_type="kmeans", feature_reduction=None, dims=10, n_samples=1000):
        self.data_name = data_name
        self.run_subset_only = run_subset_only
        self.finetuning = finetuning
        self.prompt_type = prompt_type
        self.randseed = randseed
        self.feature_type = feature_type
        self.feature_reduction = feature_reduction
        self.feature_name = None
        self.foundation_model = feature_type
        self.clustering_type = clustering_type
        self.n_clusters = 5
        self.iupac = iupac
        self.dims = dims
        self.n_samples = n_samples

        # Toy function info: [function_name, target_col, maximization, input_col]
        self.datasets_info = {
            "levy": ["levy", "score", False, "x"],  # minimization problem
            "ackley": ["ackley", "score", False, "x"],  # minimization problem
        }

        self.feature_name = self._get_feature_name()
        self.target_col = self._get_target_col()
        self.maximization = self._get_maximization()
        self.dataset_name = self.data_name

        if self.run_subset_only:
            self.dataset_name += "-subset"
            self.n_samples = min(200, self.n_samples)

        # Add dims to dataset name to avoid cache collision between different dimensions
        self.dataset_name += f"-{self.dims}d"

        self.dataset_name += "/" + self.feature_name if self.feature_name else ""
        if self.finetuning:
            self.dataset_name += "_finetuning"
        if self.clustering_type == "kmeans":
            self.dataset_name = self.dataset_name + "/kmeans/"
        else:
            self.dataset_name = self.dataset_name + "/llms/"

        print(f"Dataset name: {self.dataset_name}")
        self._load_data()

    def _get_target_col(self):
        if self.finetuning:
            print("get target", self.data_name)
            target_col = self.datasets_info[self.data_name][1]
            print(target_col)
        else:
            target_col = "targets"
        return target_col

    def _get_maximization(self):
        return self.datasets_info[self.data_name][2]

    def _get_feature_name(self):
        if self.feature_type is not None:
            feature_name = self.feature_type
        else:
            feature_name = "random_embedding"
        return feature_name

    def _get_cluster_col(self):
        if self.clustering_type == "kmeans":
            cluster_col = "cluster"
        elif self.clustering_type == "llms":
            cluster_col = "llm_cluster"
        else:
            cluster_col = "cluster"
        return cluster_col

    def _pre_clustering(self):
        """Cluster the data points based on their features"""
        if self.clustering_type == "kmeans":
            print(type(self.dataset_ft["features"].values))
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(np.stack(self.dataset_ft["features"].values))
            self.dataset_ft["cluster"] = kmeans.labels_
            self.dataset["cluster"] = kmeans.labels_
            print(self.dataset_ft)
        elif self.clustering_type == "llms":
            # For toy benchmarks, we can use random clustering or simple heuristics
            # Since we don't have LLM clustering for numerical functions
            np.random.seed(self.randseed)
            self.dataset_ft["llm_cluster"] = np.random.randint(0, self.n_clusters, len(self.dataset_ft))
            self.dataset["llm_cluster"] = self.dataset_ft["llm_cluster"]
        else:
            pass

    def _generate_toy_data(self):
        """Generate synthetic data by sampling the function space"""
        np.random.seed(self.randseed)

        # Initialize the function
        if self.data_name == "levy":
            func = Levy(dims=self.dims)
        elif self.data_name == "ackley":
            func = Ackley(dims=self.dims)
        else:
            raise ValueError(f"Unknown toy function: {self.data_name}")

        # Sample points in the function domain
        samples = []
        targets = []
        for i in range(self.n_samples):
            # Sample uniformly in the valid range
            x = np.random.uniform(func.lb, func.ub)
            y = func(x)
            samples.append(x)
            targets.append(y)

        # Use the actual x values as features (instead of random embeddings)
        # This is fair since in real molecular optimization, features represent the actual molecules
        features = samples

        # Create dataset
        dataset = pd.DataFrame({"Entry Number": np.arange(self.n_samples), "x": samples, "features": features, "targets": targets})

        return dataset, features, targets

    def _load_data(self):
        """Load or generate toy data"""
        dataset_path = DATA_DIR_NAME + "/data/dataset/" + self.dataset_name + "dataset.pkl"
        if os.path.exists(dataset_path):
            self.dataset = pd.read_pickle(dataset_path)
        else:
            self._load_data_from_scratch()
            self.save_dataset(dataset_path, file_format='pickle')

        self.dataset[self._get_cluster_col()] = self.dataset[self._get_cluster_col()].astype(int)
        print("==========data loaded=========")
        print("target col >>>>>>>>>", self.target_col)

        self.prepare_targets_for_maximization()
        cluster_col = self._get_cluster_col()
        clusters = self.dataset[cluster_col].unique()
        if len(clusters) < self.n_clusters:
            self.n_clusters = len(clusters)
            self.dataset[self._get_cluster_col()] = self.dataset[self._get_cluster_col()] - 1
        print("clusters:", self.dataset[cluster_col].unique())
        if set(self.dataset[cluster_col].unique()) != set(np.arange(self.n_clusters)):
            self.n_clusters = len(clusters)
        print("actual number of clusters:", self.n_clusters)

    def _load_data_from_scratch(self):
        """Generate toy data from scratch"""
        dataset, features, targets = self._generate_toy_data()

        self.dataset = dataset
        self.dataset_ft = pd.DataFrame({"Entry Number": dataset["Entry Number"], "x": dataset["x"], "features": features, "targets": targets})

        # Pre-clustering
        self._pre_clustering()

        if self.finetuning:
            self.target_col = self.datasets_info[self.data_name][1]
            self.dataset = dataset
        else:
            self.target_col = "targets"
            cluster_col = self._get_cluster_col()
            self.dataset_ft[cluster_col] = dataset[cluster_col]
            self.dataset = self.dataset_ft

        print("==== show data loaded =====")
        print(self.dataset)
        print(self.target_col)

        # Note: target_col_transformed will be created in prepare_targets_for_maximization()

    def prepare_targets_for_maximization(self):
        """Prepare targets for maximization (negate if minimization problem)"""
        # Turn into a maximization problem if necessary
        self.target_col_transformed = self.target_col + "_transformed"
        if not self.maximization:  # Minimization problem -> convert to maximization

            self.dataset[self.target_col_transformed] = -self.dataset[self.target_col]
        else:
            self.dataset[self.target_col_transformed] = self.dataset[self.target_col]

        self.ground_truth_opt_id = self.dataset[self.target_col_transformed].idxmax()
        self.ground_truth_max = self.dataset.iloc[self.ground_truth_opt_id][self.target_col_transformed]
        self.ground_truth_opt = self.dataset.iloc[self.ground_truth_opt_id][self.target_col]
        cluster_col = self._get_cluster_col()
        self.ground_truth_opt_cluster = self.dataset.iloc[self.ground_truth_opt_id][cluster_col]
        self.ground_truth_opt_clusters = self.dataset[self.dataset[self.target_col_transformed] == self.ground_truth_max][cluster_col].values

        print("ground_truth_max", self.ground_truth_max)
        print("ground_truth_opt", self.ground_truth_opt)
        print("ground_truth_opt_cluster", self.ground_truth_opt_cluster)
        print("ground_truth_opt_clusters", self.ground_truth_opt_clusters)

    def save_dataset(self, file_path, file_format='csv'):
        """Save dataset to file"""
        prefix_path = "/".join(file_path.split("/")[:-1])
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        if file_format == 'csv':
            self.dataset.to_csv(file_path, index=False)
        elif file_format == 'pickle':
            self.dataset.to_pickle(file_path)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'pickle'.")

    def get_data_processor(self, tokenizer):
        """Get data processor (placeholder for toy benchmarks)"""
        # Toy benchmarks don't need data processors
        return None

    def _search_dataframe(self, candidate):
        """Search for candidate in dataframe and return label"""
        # For toy benchmarks, candidate is the feature vector
        # Search by feature vector in the main dataset
        for idx, feat in enumerate(self.dataset["features"]):
            if np.allclose(feat, candidate):
                # Use the actual target column
                label = self.dataset.iloc[idx][self.target_col]
                return label

        raise ValueError(f"Candidate not found in dataset")

    def complete_call(self, candidate):
        """Complete function call with timing info"""
        test_info = {}
        time_init = time.time()
        label = self._search_dataframe(candidate)
        time_final = time.time()
        test_info['label'] = label
        test_info['time_init'] = time_init
        test_info['time_final'] = time_final
        return label, test_info

    def __call__(self, candidate):
        """Evaluate candidate"""
        label, test_info = self.complete_call(candidate)
        return label


if __name__ == "__main__":
    print("Test toy_benchmarks")
    data_names = ["levy", "ackley"]

    for data_name in data_names:
        print(f"\n{'='*50}")
        print(f"Testing {data_name}")
        print('=' * 50)

        toy_bench = ToyBench(data_name=data_name, run_subset_only=False, \
            feature_type="x_original", finetuning=False, iupac=False, prompt_type=None, randseed=666, clustering_type="kmeans", feature_reduction=None, dims=10, n_samples=200)

        print(f"\nDataset shape: {toy_bench.dataset.shape}")
        print(f"Ground truth optimum: {toy_bench.ground_truth_opt}")
        print(f"Ground truth max (transformed): {toy_bench.ground_truth_max}")
        print(f"Optimal cluster: {toy_bench.ground_truth_opt_cluster}")

        # Test evaluation
        test_candidate = toy_bench.dataset.iloc[0]["features"]
        result = toy_bench(test_candidate)
        print(f"\nTest evaluation result: {result}")
