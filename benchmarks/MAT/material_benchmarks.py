#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Wed Aug 21 17:55:54 2024
# File Name: material_benchmarks.py
# Description:
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
from benchmarks.MAT.prompt_cluster import PromptCluster


from benchmarks.MAT.data_processor import RedoxDataProcessor, \
                                          SolvationDataProcessor,\
                                          KinaseDockingDataProcessor,\
                                          LaserEmitterDataProcessor,\
                                          PhotovoltaicsPCEDataProcessor,\
                                          PhotoswitchDataProcessor
from benchmarks.MAT.prompting import PromptBuilder

current_file = inspect.getfile(inspect.currentframe())
CURRENT_DIR = os.path.dirname(os.path.abspath(current_file))

DATA_DIR_NAME = CURRENT_DIR
#os.path.join(os.getcwd(), "material_benchmarks")
# SEEDS: Final = [665, 1319, 7222, 7541, 8916]


class MATBench(object):
    """
      for a given dataset, give x/z, query y
    """

    def __init__(self, data_name, run_subset_only, finetuning,\
                 iupac, prompt_type, randseed, feature_type=None,\
                clustering_type="kmeans", feature_reduction=None):
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
        if self.iupac:
            self.data_name += "-iupac"
        self.datasets_info = {
            "redox-mer": ["redox_mer_with_iupac.csv.gz", "Ered", False, "SMILES"],
            "redox-mer-iupac": ["redox_mer_with_iupac.csv.gz", "Ered", False, "IUPAC Name"],
            "solvation": ["redox_mer_with_iupac.csv.gz", "Gsol", False, "SMILES"],
            "solvation-iupac": ["redox_mer_with_iupac.csv.gz", "Gsol", False, "IUPAC Name"],
            "kinase": ["enamine10k.csv.gz", "score", False, "SMILES"],
            "laser": ["laser_multi10k.csv.gz", "Fluorescence Oscillator Strength", True, "SMILES"],
            "pce": ["photovoltaics_pce10k.csv.gz", "pce", True, "SMILES"],
            "photoswitch": ["photoswitches.csv.gz", "Pi-Pi* Transition Wavelength", True, "SMILES"],
        }
        self.feature_name = self._get_feature_name()
        self.target_col = self._get_target_col()
        self.maximization = self._get_maximization()
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
        #self.dataset_name = self.dataset_name + "/" + self.clustering_type + "/"
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
        return self.datasets_info[self.data_name][-2]

    def _get_feature_name(self):
        assert self.feature_type is not None
        if self.feature_type not in ["fingerprints", "molformer", "gin", "gcn", "schnet"]:
            feature_name = f"{self.feature_type}-{self.prompt_type}"
            feature_name += "-average" if self.feature_reduction == "average" else ""
        else:
            feature_name = self.feature_type
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
        if self.clustering_type == "kmeans":
            print(type(self.dataset_ft["features"].values))
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(np.stack(self.dataset_ft["features"].values))
            self.dataset_ft["cluster"] = kmeans.labels_
            self.dataset["cluster"] = kmeans.labels_
            print(self.dataset_ft)
        elif self.clustering_type == "":
            pass
        elif self.clustering_type == "llms":
            PC = PromptCluster(self.data_name, self.datasets_info[self.data_name][-1])
            dataset = PC.gpt_clustering(self.dataset)
            dataset_path = DATA_DIR_NAME + "/data/dataset/" + self.dataset_name + "dataset.pkl"
            self.save_dataset(dataset_path, file_format='pickle')
            print(dataset)
        else:
            pass

    def _load_data(self):
        dataset_path = DATA_DIR_NAME + "/data/dataset/" + self.dataset_name + "dataset.pkl"
        if os.path.exists(dataset_path):
            self.dataset = pd.read_pickle(dataset_path)
        else:
            self._load_data_from_csv()
            self.save_dataset(dataset_path, file_format='pickle')
        self.dataset[self._get_cluster_col()] = self.dataset[self._get_cluster_col()].astype(int)
        print("==========data loaded=========")
        print("target col >>>>>>>>>", self.target_col)
        #self.draw_clusters()
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

    def _load_data_from_csv(self):
        if self.run_subset_only:
            dataset = pd.read_csv(DATA_DIR_NAME + "/data/random_subset_200/" + self.datasets_info[self.data_name][0])
        else:
            print(self.data_name)
            print(self.datasets_info[self.data_name][0])
            dataset = pd.read_csv(DATA_DIR_NAME + "/data/" + self.datasets_info[self.data_name][0])
        if "Unnamed: 0" in dataset.columns:
            print(">>>>> renamed >>>>>")
            dataset.rename(columns={"Unnamed: 0": "Entry Number"}, inplace=True)
        elif "Entry Number" not in dataset.columns:
            dataset["Entry Number"] = dataset.index.to_numpy()
        else:
            pass
        self.dataset = dataset

        # dataloader = data_processor.get_dataloader(dataset, batch_size=16, shuffle=False, append_eos=APPEND_EOS)

        features = torch.load(DATA_DIR_NAME + "/data/features/" + f"{self.data_name}/{self.feature_name}_feats.bin")
        targets = torch.load(DATA_DIR_NAME + "/data/features/" + f"{self.data_name}/{self.feature_name}_targets.bin")
        # if self.run_subset_only:
        print("data cols>>>>>>>>", dataset.columns)
        print(dataset)
        if "Entry Number" in dataset.columns:
            features_to_get = dataset["Entry Number"].to_numpy()

        print(">>>> index of features to get >>>>>")
        print(features_to_get)
        mol_subset, features_subset, targets_subset = [], [], []
        for entry_num in features_to_get:
            # print(entry_num, features[entry_num])
            mol_subset.append(dataset.iloc[entry_num]["SMILES"])
            features_subset.append(features[entry_num].flatten().numpy())
            targets_subset.append(targets[entry_num].numpy()[0])
        features = features_subset
        targets = targets_subset
        molecules = mol_subset
        # else:
        # features, targets = skshuffle(features, targets, random_state=self.randseed)
        # print(features, targets)
        # print(targets)

        self.dataset_ft = pd.DataFrame({"Entry Number": features_to_get, "SMILES": molecules, "features": features, "targets": targets})
        # if not self.finetuning
        #### --------- pre clustering ----------
        self._pre_clustering()

        if self.finetuning:
            self.target_col = self.datasets_info[self.data_name][1]
            self.dataset = dataset
        else:
            self.target_col = "targets"
            cluster_col = self._get_cluster_col()
            self.dataset_ft[cluster_col] = dataset[cluster_col]
            self.dataset = self.dataset_ft
            # print("load fixed feature data")
            # print(self.dataset_ft)
            # self.ground_truth_max = torch.tensor(targets).flatten().max()
            # dataset = self.dataset_ft
        print("==== show data loaded =====")
        print(self.dataset)
        print(self.target_col)
        # print("load data")
        # print(self.dataset)

    def prepare_targets_for_maximization(self):

        # Turn into a maximization data_name if necessary
        self.target_col_transformed = self.target_col + "_transformed"
        if not self.maximization:  # Maximization
            if self.finetuning:
                self.dataset[self.target_col_transformed] = -self.dataset[self.target_col]
            else:
                self.dataset[self.target_col_transformed] = self.dataset[self.target_col]
                self.dataset[self.target_col] = -self.dataset[self.target_col_transformed]
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
        print(self.dataset)

    def draw_clusters(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        print(self.dataset)
        # plt.hist(self.dataset[self.target_col], bins=30, alpha=0.7, color='blue')
        self.prepare_targets_for_maximization()
        # Add horizontal alignment lines for each group
        cluster_col = self._get_cluster_col()
        clusters = self.dataset[cluster_col].unique()
        print("clustser ids >>>>>>>>>>>", clusters)
        # print(self.dataset[self.dataset[cluster_col] == 0][self.target_col])
        # for i, c in enumerate(clusters):
        #     y = self.dataset[self.dataset["cluster"] == c]
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=self.dataset, x=self.target_col, hue=cluster_col, fill=True, common_norm=True, alpha=0.1, linewidth=1.5)
        # plt.axhline(y=i + 0.5, color="black", linestyle="-", linewidth=0.5)
        # Find the maximum value of the target column and its corresponding cluster
        max_value = self.dataset.loc[self.ground_truth_opt_id][self.target_col]
        max_cluster = set(self.dataset[self.dataset[self.target_col] == max_value][cluster_col].values)

        # Add a vertical line at the position of the maximum value
        plt.axvline(x=max_value, color='red', linestyle='--', linewidth=1.5)

        # Annotate the plot with the cluster information
        if self.maximization:
            task = "max"
        else:
            task = "min"
        plt.text(max_value, plt.ylim()[1] * 0.9, f'Optimal: {max_value:.2f}\nCluster: {max_cluster}\ntask:{task}', color='red', ha='center')

        # Customize the plot
        plt.xlabel(self.target_col)
        plt.ylabel("")
        plt.yticks([])
        if self.clustering_type == "kmeans":
            plt.title(f'Distribution of {self.target_col} of {self.data_name} dataset clustered by {self.feature_name}')
        else:
            plt.title(f'Distribution of {self.target_col} of {self.data_name} dataset clustered by chatgpt-4o')

        # plt.tight_layout()
        save_path = DATA_DIR_NAME + "/cluster_figs/" + f"{self.data_name}_{self.feature_name}_{self.clustering_type}_clusters.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.show()

    def save_dataset(self, file_path, file_format='csv'):
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
        print("dataset name:" + self.data_name)
        if self.feature_type == "molformer":
            self.prompt_type = "just-smiles"
        prompt_builder = PromptBuilder(kind=self.prompt_type)
        if "redox-mer" in self.data_name:
            data_processor = RedoxDataProcessor(prompt_builder, tokenizer, self.iupac, self.clustering_type)
        elif "solvation" in self.data_name:
            data_processor = SolvationDataProcessor(prompt_builder, tokenizer, self.iupac, self.clustering_type)
        elif self.data_name == "kinase":
            data_processor = KinaseDockingDataProcessor(prompt_builder, tokenizer, self.clustering_type)
        elif self.data_name == "laser":
            data_processor = LaserEmitterDataProcessor(prompt_builder, tokenizer, self.clustering_type)
        elif self.data_name == "pce":
            data_processor = PhotovoltaicsPCEDataProcessor(prompt_builder, tokenizer, self.clustering_type)
        elif self.data_name == "photoswitch":
            data_processor = PhotoswitchDataProcessor(prompt_builder, tokenizer, self.clustering_type)
        else:
            print("Invalid test function!")
            sys.exit(1)
        return data_processor

    def _search_dataframe(self, candidate):
        ## use simes or iupac
        if not self.finetuning and self.feature_type is not None:
            mask = self.dataset[self.datasets_info[self.data_name][-1]] == candidate
            idx = np.where(mask)
            assert len(idx) == 1, 'The query has resulted into mulitple matches. This should not happen. ' \
                                f'The Query was {candidate}'
            idx = idx[0][0]
            label = self.dataset.iloc[idx][self.datasets_info[self.data_name][1]]
        else:
            mask = self.dataset_ft['features'] == candidate
            idx = np.where(mask)
            assert len(idx) == 1, 'The query has resulted into mulitple matches. This should not happen. ' \
                                f'The Query was {candidate}'
            idx = idx[0][0]
            label = self.dataset_ft.iloc[idx]["targets"]
        return label

    def complete_call(self, candiate):
        test_info = {}
        time_init = time.time()
        label = self._search_dataframe(candiate)
        time_final = time.time()
        test_info['label'] = label
        test_info['time_init'] = time_init
        test_info['time_final'] = time_final
        return label, test_info

    def __call__(self, candidate):

        label, test_info = self.complete_call(candidate)
        # # test_info.update(candidate)
        # self.add_config(test_info)
        return label


if __name__ == "__main__":
    print("test material_benchmarks")
    data_names = ["solvation", "redox-mer", "kinase", "laser", "pce", "photoswitch"]
    # "solvation",
    f_models = ["t5-base-chem", "gpt2-medium", "t5-base", "gpt2-large", "molformer"]
    for data_name in data_names:
        for f_model in f_models:
            finetuning = False
            prompt_type = "just-smiles"
            iupac = False
            seed = 666
            feature_reduction = "average"
            mat_bench = MATBench(
                data_name=data_name,
                run_subset_only=False,
                feature_type=f_model,
                finetuning=finetuning,
                iupac=iupac,
                prompt_type=prompt_type,
                randseed=seed,
                clustering_type="llms",
                feature_reduction=feature_reduction,
            )
            # PC = PromptCluster(mat_bench.data_name, mat_bench.datasets_info[mat_bench.data_name][-1])
            # dataset = PC.gpt_clustering(mat_bench.dataset)
