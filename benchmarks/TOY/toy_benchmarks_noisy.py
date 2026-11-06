#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-27
# File Name: toy_benchmarks_noisy.py
# Description: Extension of toy_benchmarks.py with noise injection for ablation studies
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

# Import from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from toy_benchmarks import ToyBench, Levy, Ackley, tracker

current_file = inspect.getfile(inspect.currentframe())
CURRENT_DIR = os.path.dirname(os.path.abspath(current_file))

DATA_DIR_NAME = CURRENT_DIR


class NoisyToyBench(ToyBench):
    """
    Extension of ToyBench with noise injection capability.

    Supports different noise types:
    - gaussian: Additive Gaussian noise N(0, noise_scale * std(targets))
    - uniform: Uniform noise U(-noise_scale * std, +noise_scale * std)
    - multiplicative: Multiplicative noise (1 + N(0, noise_scale)) * target
    """

    def __init__(self, data_name, run_subset_only=False, finetuning=False,
                 iupac=False, prompt_type=None, randseed=666, feature_type=None,
                 clustering_type="kmeans", feature_reduction=None, dims=10, n_samples=1000,
                 noise_type="gaussian", noise_scale=0.0, noise_seed=None):
        """
        Args:
            noise_type: Type of noise ('gaussian', 'uniform', 'multiplicative')
            noise_scale: Scale of noise relative to target std
            noise_seed: Random seed for noise generation (if None, uses randseed)
        """
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noise_seed = noise_seed if noise_seed is not None else randseed

        # Initialize parent class
        super().__init__(data_name=data_name, run_subset_only=run_subset_only,
                        finetuning=finetuning, iupac=iupac, prompt_type=prompt_type,
                        randseed=randseed, feature_type=feature_type,
                        clustering_type=clustering_type, feature_reduction=feature_reduction,
                        dims=dims, n_samples=n_samples)

    def _inject_noise(self, targets):
        """
        Inject noise into targets based on noise_type and noise_scale.

        Args:
            targets: Original target values (numpy array)

        Returns:
            noisy_targets: Target values with noise added
        """
        if self.noise_scale == 0.0:
            return targets

        # Set random seed for reproducibility
        np.random.seed(self.noise_seed)

        targets = np.array(targets)
        target_std = np.std(targets)

        if self.noise_type == "gaussian":
            # Additive Gaussian noise: y_noisy = y + N(0, noise_scale * std(y))
            noise = np.random.normal(0, self.noise_scale * target_std, len(targets))
            noisy_targets = targets + noise

        elif self.noise_type == "uniform":
            # Uniform noise: y_noisy = y + U(-noise_scale * std, +noise_scale * std)
            noise = np.random.uniform(-self.noise_scale * target_std,
                                     self.noise_scale * target_std,
                                     len(targets))
            noisy_targets = targets + noise

        elif self.noise_type == "multiplicative":
            # Multiplicative noise: y_noisy = y * (1 + N(0, noise_scale))
            noise = np.random.normal(0, self.noise_scale, len(targets))
            noisy_targets = targets * (1 + noise)

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}. "
                           f"Choose from 'gaussian', 'uniform', 'multiplicative'")

        print(f"Injected {self.noise_type} noise with scale {self.noise_scale}")
        print(f"Target std: {target_std:.4f}, Noise std: {np.std(noisy_targets - targets):.4f}")
        print(f"SNR (signal-to-noise ratio): {target_std / np.std(noisy_targets - targets):.4f}")

        return noisy_targets

    def _generate_toy_data(self):
        """Generate synthetic data by sampling the function space and adding noise"""
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
        clean_targets = []
        for i in range(self.n_samples):
            # Sample uniformly in the valid range
            x = np.random.uniform(func.lb, func.ub)
            y = func(x)
            samples.append(x)
            clean_targets.append(y)

        # Store clean targets for true gap calculation
        clean_targets_array = np.array(clean_targets)

        # Inject noise into targets
        noisy_targets = self._inject_noise(clean_targets)

        # Use the actual x values as features
        features = samples

        # Create dataset with both clean and noisy targets
        dataset = pd.DataFrame({
            "Entry Number": np.arange(self.n_samples),
            "x": samples,
            "features": features,
            "targets": noisy_targets,  # noisy targets (used for BO)
            "targets_clean": clean_targets_array  # clean targets (for evaluation)
        })

        return dataset, features, noisy_targets

    def get_clean_optimum(self):
        """
        Get the true optimum based on clean targets (without noise).
        This is used for evaluation in ablation studies.

        Returns:
            clean_opt: The clean target value at the optimum
        """
        if 'targets_clean' not in self.dataset.columns:
            # If no clean targets, fall back to noisy optimum
            return self.ground_truth_opt

        # Find optimum based on transformed clean targets
        if not self.maximization:
            # Minimization: negate clean targets
            clean_transformed = -self.dataset['targets_clean']
        else:
            clean_transformed = self.dataset['targets_clean']

        clean_opt_id = clean_transformed.idxmax()
        clean_opt = self.dataset.iloc[clean_opt_id]['targets_clean']

        return clean_opt

    def _get_feature_name(self):
        """Override to include noise info in feature name"""
        if self.feature_type is not None:
            feature_name = self.feature_type
        else:
            feature_name = "random_embedding"

        # Add noise info to feature name to prevent cache collision
        if self.noise_scale > 0:
            feature_name += f"_noise-{self.noise_type}-{self.noise_scale}"

        return feature_name


if __name__ == "__main__":
    print("Test toy_benchmarks_noisy")
    data_names = ["levy", "ackley"]
    noise_scales = [0.0, 0.1, 0.2, 0.5]

    for data_name in data_names:
        for noise_scale in noise_scales:
            print(f"\n{'='*60}")
            print(f"Testing {data_name} with noise_scale={noise_scale}")
            print('=' * 60)

            toy_bench = NoisyToyBench(
                data_name=data_name,
                run_subset_only=True,
                feature_type="x_original",
                finetuning=False,
                iupac=False,
                prompt_type=None,
                randseed=666,
                clustering_type="kmeans",
                feature_reduction=None,
                dims=10,
                n_samples=200,
                noise_type="gaussian",
                noise_scale=noise_scale,
                noise_seed=42
            )

            print(f"\nDataset shape: {toy_bench.dataset.shape}")
            print(f"Target statistics:")
            print(f"  Mean: {toy_bench.dataset['targets'].mean():.4f}")
            print(f"  Std: {toy_bench.dataset['targets'].std():.4f}")
            print(f"  Min: {toy_bench.dataset['targets'].min():.4f}")
            print(f"  Max: {toy_bench.dataset['targets'].max():.4f}")
