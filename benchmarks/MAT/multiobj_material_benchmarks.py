#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.edu
# Created Time : 2025
# File Name: multiobj_material_benchmarks.py
# Description: Multi-objective benchmark for material discovery tasks
"""
import pandas as pd
import torch
import os
import sys
import time
import numpy as np
from sklearn.cluster import KMeans

from benchmarks.MAT.prompting import PromptBuilder
from benchmarks.MAT.multiobj_data_processor import (
    MultiRedoxDataProcessor,
    MultiLaserDataProcessor,
)
from benchmarks.MAT.prompt_cluster_multiobj import PromptClusterMultiObj

DATA_DIR_NAME = "benchmarks/MAT"


class MultiObjMATBench:
    """
    Multi-objective benchmark class for material discovery tasks.
    Handles datasets with multiple optimization objectives.
    """

    datasets_info = {
        # Format: (filename, obj_cols, maximizations, smiles_col)
        "redox-mer": (
            "redox_mer.csv",
            ["Ered", "Gsol"],
            [False, False],  # Both minimize
            "SMILES"),
        "laser": (
            "laser_multi10k.csv.gz",
            ["Fluorescence Oscillator Strength", "Electronic Gap"],
            [True, True],  # Both maximize
            "SMILES"),
    }

    def __init__(
        self,
        data_name="redox-mer",
        run_subset_only=False,
        feature_type="molformer",
        finetuning=False,
        iupac=False,
        prompt_type="just-smiles",
        randseed=42,
        clustering_type="kmeans",
        feature_reduction="average",
        n_clusters=5,
    ):
        """
        Initialize multi-objective benchmark.

        Args:
            data_name: Name of the dataset ('redox-mer', 'laser')
            run_subset_only: Whether to use subset (for testing)
            feature_type: Type of features ('molformer', 't5-base-chem', etc.)
            finetuning: Whether to use finetuned features
            iupac: Whether to use IUPAC names (not used for multi-obj)
            prompt_type: Type of prompt for LLM features
            randseed: Random seed
            clustering_type: Type of clustering ('kmeans', 'llms')
            feature_reduction: Feature reduction method ('average', 'last')
            n_clusters: Number of clusters
        """
        assert data_name in self.datasets_info, f"Dataset {data_name} not supported"

        self.data_name = data_name
        self.run_subset_only = run_subset_only
        self.feature_type = feature_type
        self.foundation_model = feature_type  # Add this line - needed for LLMAT
        self.finetuning = finetuning
        self.iupac = iupac
        self.prompt_type = prompt_type
        self.randseed = randseed
        self.clustering_type = clustering_type
        self.feature_reduction = feature_reduction
        self.n_clusters = n_clusters

        # Get dataset info
        filename, obj_cols, maximizations, smiles_col = self.datasets_info[data_name]
        self.filename = filename
        self.obj_cols = obj_cols
        self.maximizations = maximizations
        self.smiles_col = smiles_col
        self.n_objectives = len(obj_cols)

        # Feature name for loading cached features
        if feature_type not in ["fingerprints", "molformer"]:
            self.feature_name = f"{feature_type}-{feature_reduction}"
        else:
            self.feature_name = feature_type

        # Initialize dataset
        self.dataset = None
        self.dataset_ft = None
        self.targets_multi = None  # Multi-objective targets

        # Construct dataset name for pickle path
        self.dataset_name = self.data_name
        if self.run_subset_only:
            self.dataset_name += "-subset"

        self.dataset_name += "/" + self.feature_name
        if self.finetuning:
            self.dataset_name += "_finetuning"
        if self.clustering_type == "kmeans":
            self.dataset_name = self.dataset_name + "/kmeans/"
        else:
            self.dataset_name = self.dataset_name + "/llms/"

        print(f"Dataset name: {self.dataset_name}")

        # Load data
        self._load_data()

    def _load_data(self):
        """
        Main data loading method. Tries to load from pickle first,
        falls back to CSV if pickle doesn't exist.
        """
        dataset_path = DATA_DIR_NAME + "/data/dataset_multiobj/" + self.dataset_name + "dataset.pkl"
        if os.path.exists(dataset_path):
            print(f"Loading dataset from pickle: {dataset_path}")
            self.dataset = pd.read_pickle(dataset_path)

            # Restore targets_multi from individual columns
            targets_list = []
            for obj_col in self.obj_cols:
                targets_list.append(self.dataset[obj_col].values)
            self.targets_multi = torch.tensor(np.stack(targets_list, axis=1))

            # Backward compatibility: handle old cluster column naming
            cluster_col = self._get_cluster_col()
            if cluster_col not in self.dataset.columns:
                # Look for old naming convention
                old_cluster_col = f"kmeans_cluster_{self.feature_name}" if self.clustering_type == "kmeans" else None
                if old_cluster_col and old_cluster_col in self.dataset.columns:
                    print(f"Renaming cluster column from '{old_cluster_col}' to '{cluster_col}'")
                    self.dataset[cluster_col] = self.dataset[old_cluster_col]
                    self.dataset.drop(columns=[old_cluster_col], inplace=True)
                else:
                    print(f"Cluster column '{cluster_col}' not found in loaded dataset. Re-clustering may be needed.")
        else:
            print(f"Pickle not found, loading from CSV...")
            self._load_data_from_csv()
            self.save_dataset(dataset_path, file_format='pickle')

        # Ensure cluster column is integer
        cluster_col = self._get_cluster_col()
        self.dataset[cluster_col] = self.dataset[cluster_col].astype(int)

        print("==========data loaded=========")
        print("target cols >>>>>>>>>", self.obj_cols)

        # Prepare targets for maximization
        self.prepare_targets_for_maximization()

        # Validate clusters
        clusters = self.dataset[cluster_col].unique()
        if len(clusters) < self.n_clusters:
            self.n_clusters = len(clusters)
            self.dataset[cluster_col] = self.dataset[cluster_col] - 1

        print(f"Clusters: {sorted(self.dataset[cluster_col].unique())}")
        if set(self.dataset[cluster_col].unique()) != set(np.arange(self.n_clusters)):
            self.n_clusters = len(clusters)
        print(f"Actual number of clusters: {self.n_clusters}")

    def _get_cluster_col(self):
        """Get the cluster column name based on clustering type."""
        if self.clustering_type == "kmeans":
            return "cluster"  # Changed from f"kmeans_cluster_{self.feature_name}"
        elif self.clustering_type == "llms":
            return "llm_cluster"
        else:
            raise ValueError(f"Unknown clustering type: {self.clustering_type}")

    def _pre_clustering(self):
        """Perform clustering on the dataset."""
        cluster_col = self._get_cluster_col()

        if self.clustering_type == "kmeans":
            # K-means clustering on features
            features_array = np.array([f for f in self.dataset_ft["features"]])
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.randseed, n_init=10)
            clusters = kmeans.fit_predict(features_array)
            self.dataset[cluster_col] = clusters
            print(f"K-means clustering complete. Clusters: {np.unique(clusters)}")

        elif self.clustering_type == "llms":
            # LLM-based clustering using multi-objective prompt
            if cluster_col not in self.dataset.columns:
                print(f"Running LLM clustering for multi-objective {self.data_name}...")
                clusterer = PromptClusterMultiObj(data_name=self.data_name, molecule_col=self.smiles_col, obj_cols=self.obj_cols)
                self.dataset = clusterer.gpt_clustering_multiobj(self.dataset)
            else:
                print(f"LLM cluster column '{cluster_col}' already exists.")

        # Validate clusters
        clusters = self.dataset[cluster_col].unique()
        if len(clusters) < self.n_clusters:
            self.n_clusters = len(clusters)
            print(f"Warning: Actual number of clusters ({len(clusters)}) less than requested ({self.n_clusters})")

        print(f"Clusters: {sorted(clusters)}")
        print(f"Actual number of clusters: {self.n_clusters}")

    def _load_data_from_csv(self):
        """Load dataset and features from cached files."""
        # Load CSV data
        if self.run_subset_only:
            csv_path = DATA_DIR_NAME + "/data/random_subset_200/" + self.filename
        else:
            csv_path = DATA_DIR_NAME + "/data/" + self.filename

        print(f"Loading dataset from: {csv_path}")
        dataset = pd.read_csv(csv_path)

        # Handle index column
        if "Unnamed: 0" in dataset.columns:
            dataset.rename(columns={"Unnamed: 0": "Entry Number"}, inplace=True)
        elif "Entry Number" not in dataset.columns:
            dataset["Entry Number"] = dataset.index.to_numpy()

        self.dataset = dataset

        # Load cached features and targets
        feature_dir = DATA_DIR_NAME + "/data/features/multiobj/" + self.data_name + "/"
        features = torch.load(feature_dir + f"{self.feature_name}_multi_feats.bin")
        targets = torch.load(feature_dir + f"{self.feature_name}_multi_targets.bin")

        print(f"Loaded features shape: {len(features)}")
        print(f"Loaded targets shape: {targets.shape}")
        assert targets.shape[1] == self.n_objectives, \
            f"Expected {self.n_objectives} objectives, got {targets.shape[1]}"

        # Get subset if needed
        if "Entry Number" in dataset.columns:
            features_to_get = dataset["Entry Number"].to_numpy()
        else:
            features_to_get = np.arange(len(dataset))

        print(f"Getting features for entries: {len(features_to_get)}")

        # Extract subset
        mol_subset, features_subset, targets_subset = [], [], []
        for entry_num in features_to_get:
            mol_subset.append(dataset.iloc[entry_num][self.smiles_col])
            features_subset.append(features[entry_num].flatten().numpy())
            # Keep multi-objective targets as array
            targets_subset.append(targets[entry_num].numpy())

        features = features_subset
        targets_multi = np.array(targets_subset)  # Shape: (n_samples, n_objectives)
        molecules = mol_subset

        # Store multi-objective targets
        self.targets_multi = torch.tensor(targets_multi)

        # Create dataset for fixed features
        self.dataset_ft = pd.DataFrame({
            "Entry Number": features_to_get,
            self.smiles_col: molecules,
            "features": features,
            "targets_multi": list(targets_multi),  # Store as list for DataFrame
        })

        # Add individual target columns
        for i, obj_col in enumerate(self.obj_cols):
            self.dataset_ft[obj_col] = targets_multi[:, i]

        # Pre-clustering
        self._pre_clustering()

        # Set up dataset based on finetuning mode
        if self.finetuning:
            # For finetuning, use original dataset with SMILES
            self.target_cols = self.obj_cols
            # dataset already loaded
        else:
            # For fixed features, use dataset_ft
            self.target_cols = self.obj_cols
            cluster_col = self._get_cluster_col()
            self.dataset_ft[cluster_col] = self.dataset[cluster_col]
            self.dataset = self.dataset_ft

        print("==== Multi-objective data loaded =====")
        print(f"Dataset shape: {self.dataset.shape}")
        print(f"Objectives: {self.obj_cols}")
        print(f"Maximizations: {self.maximizations}")
        print(self.dataset.head())

    def prepare_targets_for_maximization(self):
        """
        Prepare multi-objective targets for maximization.
        Creates transformed columns for each objective.
        """
        self.target_cols_transformed = [obj_col + "_transformed" for obj_col in self.obj_cols]

        if not hasattr(self, 'maximization_prepared'):
            for i, (obj_col, maximization) in enumerate(zip(self.obj_cols, self.maximizations)):
                transformed_col = self.target_cols_transformed[i]

                if not maximization:  # Need to negate for minimization
                    if self.finetuning:
                        self.dataset[transformed_col] = -self.dataset[obj_col]
                    else:
                        # For fixed features, it has already been transformed when caching features
                        self.dataset[transformed_col] = self.dataset[obj_col]
                        self.dataset[obj_col] = -self.dataset[transformed_col]
                else:
                    self.dataset[transformed_col] = self.dataset[obj_col]
            self.maximization_prepared = True

        print("Transformed objectives for maximization:")
        for obj_col, transformed_col in zip(self.obj_cols, self.target_cols_transformed):
            print(f"  {obj_col} -> {transformed_col}")

    def compute_pareto_front(self):
        """
        Compute the Pareto front for multi-objective optimization.
        Returns indices of Pareto-optimal points.
        """
        # Get transformed targets (all maximization)
        if not hasattr(self, 'target_cols_transformed'):
            self.prepare_targets_for_maximization()

        targets = self.dataset[self.target_cols_transformed].values
        n_points = targets.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if j dominates i (all >= and at least one >)
                    if np.all(targets[j] >= targets[i]) and np.any(targets[j] > targets[i]):
                        is_pareto[i] = False
                        break

        pareto_indices = np.where(is_pareto)[0]
        print(f"Pareto front contains {len(pareto_indices)} points")
        return pareto_indices

    def get_ground_truth_info(self):
        """
        Get ground truth information for multi-objective optimization.
        Returns Pareto front and reference point for hypervolume.
        """
        if not hasattr(self, 'target_cols_transformed'):
            self.prepare_targets_for_maximization()

        # Compute Pareto front
        pareto_indices = self.compute_pareto_front()
        pareto_front = self.dataset.iloc[pareto_indices][self.target_cols_transformed].values

        # Compute reference point (for hypervolume)
        # Use minimum of each objective across all data
        ref_point = self.dataset[self.target_cols_transformed].min().values - 0.1

        # Get cluster information for Pareto points
        cluster_col = self._get_cluster_col()
        pareto_clusters = self.dataset.iloc[pareto_indices][cluster_col].values

        print("Ground truth information:")
        print(f"  Pareto front size: {len(pareto_indices)}")
        print(f"  Reference point: {ref_point}")
        print(f"  Pareto clusters: {np.unique(pareto_clusters)}")

        return {
            'pareto_indices': pareto_indices,
            'pareto_front': pareto_front,
            'ref_point': ref_point,
            'pareto_clusters': pareto_clusters,
        }

    def draw_pareto_front(self):
        """Visualize Pareto front (2D only)."""
        import matplotlib.pyplot as plt

        if self.n_objectives != 2:
            print("Visualization only supported for 2 objectives")
            return

        if not hasattr(self, 'target_cols_transformed'):
            self.prepare_targets_for_maximization()

        gt_info = self.get_ground_truth_info()
        cluster_col = self._get_cluster_col()

        # Plot all points with clusters
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all points colored by cluster
        clusters = self.dataset[cluster_col].unique()
        for cluster in sorted(clusters):
            mask = self.dataset[cluster_col] == cluster
            cluster_data = self.dataset[mask]
            ax.scatter(cluster_data[self.target_cols_transformed[0]], cluster_data[self.target_cols_transformed[1]], alpha=0.5, label=f'Cluster {cluster}', s=20)

        # Highlight Pareto front
        pareto_front = gt_info['pareto_front']
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', s=100, marker='*', label='Pareto Front', edgecolors='black', linewidths=1.5, zorder=10)

        ax.set_xlabel(f'{self.obj_cols[0]} (transformed)')
        ax.set_ylabel(f'{self.obj_cols[1]} (transformed)')
        ax.set_title(f'Pareto Front - {self.data_name} ({self.feature_name}, {self.clustering_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save figure
        save_path = DATA_DIR_NAME + "/cluster_figs/" + \
            f"{self.data_name}_multiobj_{self.feature_name}_{self.clustering_type}_pareto.pdf"

        prefix_path = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved Pareto front plot to: {save_path}")
        plt.show()

    def save_dataset(self, file_path, file_format='csv'):
        """Save dataset to file."""
        prefix_path = "/".join(file_path.split("/")[:-1])
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)

        if file_format == 'csv':
            # Convert targets_multi to string for CSV
            df_to_save = self.dataset.copy()
            if 'targets_multi' in df_to_save.columns:
                df_to_save['targets_multi'] = df_to_save['targets_multi'].apply(str)
            df_to_save.to_csv(file_path, index=False)
        elif file_format == 'pickle':
            self.dataset.to_pickle(file_path)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'pickle'.")

        print(f"Saved dataset to: {file_path}")

    def get_data_processor(self, tokenizer):
        """Get data processor for this dataset."""
        print(f"Getting data processor for: {self.data_name}")

        if self.feature_type == "molformer":
            self.prompt_type = "just-smiles"

        prompt_builder = PromptBuilder(kind=self.prompt_type)

        if self.data_name == "redox-mer":
            data_processor = MultiRedoxDataProcessor(prompt_builder, tokenizer, self.clustering_type)
        elif self.data_name == "laser":
            data_processor = MultiLaserDataProcessor(prompt_builder, tokenizer, self.clustering_type)
        else:
            raise ValueError(f"Unknown dataset: {self.data_name}")

        return data_processor

    def _search_dataframe(self, candidate):
        """Search for candidate in dataframe and return multi-objective labels."""
        if not self.finetuning and self.feature_type is not None:
            mask = self.dataset[self.smiles_col] == candidate
            idx = np.where(mask)[0]
            assert len(idx) == 1, f'Query resulted in {len(idx)} matches for {candidate}'
            idx = idx[0]
            # Return multi-objective targets
            labels = [self.dataset.iloc[idx][obj_col] for obj_col in self.obj_cols]
        else:
            # For features
            mask = self.dataset_ft['features'] == candidate
            idx = np.where(mask)[0]
            assert len(idx) == 1, f'Query resulted in {len(idx)} matches'
            idx = idx[0]
            labels = self.dataset_ft.iloc[idx]["targets_multi"]

        return np.array(labels)

    def complete_call(self, candidate):
        """Complete evaluation call for candidate."""
        test_info = {}
        time_init = time.time()
        labels = self._search_dataframe(candidate)
        time_final = time.time()
        test_info['labels'] = labels
        test_info['time_init'] = time_init
        test_info['time_final'] = time_final
        return labels, test_info

    def __call__(self, candidate):
        """Evaluate candidate and return multi-objective labels."""
        labels, test_info = self.complete_call(candidate)
        return labels


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Objective MAT Benchmark")
    print("=" * 60)

    # Test datasets
    data_names = ["laser"]  #"redox-mer",
    f_models = ["t5-base-chem"]

    for data_name in data_names:
        for f_model in f_models:
            print(f"\n{'='*60}")
            print(f"Dataset: {data_name}, Feature: {f_model}")
            print(f"{'='*60}")

            finetuning = False
            prompt_type = "just-smiles"
            iupac = False
            seed = 666
            feature_reduction = "average"

            mat_bench = MultiObjMATBench(
                data_name=data_name,
                run_subset_only=False,
                feature_type=f_model,
                finetuning=finetuning,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=seed,
                clustering_type="llms",  # Change to "llms" to use LLM clustering
                feature_reduction=feature_reduction,
            )

            # Prepare targets
            mat_bench.prepare_targets_for_maximization()

            # Get ground truth
            gt_info = mat_bench.get_ground_truth_info()

            # Draw Pareto front
            mat_bench.draw_pareto_front()

            print(f"\nCompleted: {data_name} with {f_model}")
