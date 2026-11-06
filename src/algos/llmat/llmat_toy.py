#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-26
# File Name: llmat_toy.py
# Description: LLMAT for toy benchmarks without foundation models
"""

import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn, optim
import math
import pandas as pd
import tqdm
from contextlib import nullcontext
from copy import deepcopy
try:
    from transformers import get_scheduler
except ImportError:
    get_scheduler = None  # Optional for toy benchmarks
from typing import *
from botorch.posteriors import DeterministicPosterior
import numpy as np
from src.algos.llmat.node import Node
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import random
import pingouin as pg
from scipy import stats
from src.utils.helpers import trace_times


class LLMAT_Toy(object):
    """
    Amortized Learning of Partition Hierarchies (ALPHA) for Bayesian Optimization.
    Simplified version for toy benchmarks without foundation models.
    Uses direct feature vectors instead of LLM embeddings.
    """

    def __init__(
            self,
            training_set: pd.DataFrame,
            head_cfg: dict,
            device: str = "cuda",
            dtype: str = "float32",
            gamma: float = 0.33,  # quantile
            acf_type: str = "pi",
            lmbda: float = 1,  # ucb weight
            eta: float = 0.5,  # meta learning rate
            tree_depth: int = 3,
            threshold: float = 0.3,  # threshold for partitioning
            alpha: float = 1.0,  # for ei, pi, etc...
            feature_dim: int = 10,  # feature dimension
            target_col: str = "targets",  # target column name to use
            selection_type: str = "uct",  # "uct" or "greedy"
    ):

        # ========== Misc ==========
        self.dtype = dtype
        self.ptdtype = {"float32": torch.float32,\
                        "bfloat16": torch.bfloat16,\
                        "float16": torch.float16}[dtype]
        self.ctx = (nullcontext() if device == "cpu" or device == "mps" \
                                  else torch.amp.autocast(device_type="cuda", dtype=self.ptdtype))
        self.enable_grad_scaler = dtype in ["float16", "bfloat16"]

        # ========== Feature Extraction ===========
        # For toy benchmarks, we use features directly without any foundation model
        self.feature_dim = feature_dim
        self.last_hidden_size = 50

        # Classification and regression heads
        self.clf_head = nn.Sequential(nn.Linear(self.feature_dim, self.last_hidden_size), nn.ReLU(), nn.Linear(self.last_hidden_size, 1))
        self.reg_head = nn.Sequential(nn.Linear(self.feature_dim, self.last_hidden_size), nn.ReLU(), nn.Linear(self.last_hidden_size, 1))

        self.training_set = training_set
        self.device = device
        self.target_col = target_col  # Store the target column name

        print(f"=== LLMAT_Toy Initialized ===")
        print(f"Target column: {self.target_col}")
        print(f"Training set shape: {self.training_set.shape}")
        print(f"Training set columns: {list(self.training_set.columns)}")
        print(f"==================================")

        ## ========= meta-learning ===========
        self.theta = self.clf_head.state_dict()
        self.eta = eta

        ## ========= MCTS ===========
        self.lmbda = lmbda
        self.tree_depth = tree_depth
        self.selection_type = selection_type
        self.nodes = []
        self.head_cfg = head_cfg
        root = Node(parent=None, reset_id=True, clf=self.clf_head.state_dict(), target_col=self.target_col)
        self.nodes.append(root)
        self.ROOT = root
        self.CURT = self.ROOT
        self.LEAF_SAMPLE_SIZE = head_cfg.get("leaf_sample_size", 10)  # Default to 10 if not in config
        self.threshold = threshold

        ## ========= BO ===========
        self.gamma = gamma
        self.acf_type = acf_type
        self.alpha = alpha

        # ========= init clf head =======
        self.init_weights_xavier()

    def init_weights_xavier(self):
        for param in self.clf_head.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def get_traindata_weight(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.training_set

        print(f"DEBUG: self.target_col = '{self.target_col}'")
        print(f"DEBUG: Available columns: {list(train_dataset.columns)}")
        print(f"DEBUG: First few rows of dataset:")
        print(train_dataset.head(3))

        # Verify the column exists
        if self.target_col not in train_dataset.columns:
            print(f"ERROR: Column '{self.target_col}' not found in dataset!")
            print(f"Available columns: {list(train_dataset.columns)}")
            raise KeyError(f"Column '{self.target_col}' not found")

        y_reg = train_dataset[self.target_col]  # Use the specified target column (transformed)
        print(f"DEBUG: y_reg column name: {y_reg.name}")
        print(f"DEBUG: y_reg first 5 values: {y_reg.head().tolist()}")
        y_cluster = train_dataset['cluster'] if 'cluster' in train_dataset else train_dataset['llm_cluster']
        x = train_dataset['features']

        tau = np.quantile(y_reg, q=1 - self.gamma)
        y_class = np.greater_equal(y_reg, tau)
        print("train length:", len(y_reg))
        print("tau:", tau)
        print("-----------------")
        print("reg targets:", y_reg)
        print("class labels:", y_class)
        print("------- finish get weight----------")

        return x, x, y_class, y_cluster, y_reg, tau

    def _weighted_vi_loss(self, y_class, y_reg, y_pred, tau):
        positive_mask = y_class == 1
        negative_mask = y_class == 0

        if self.acf_type == "ei":
            sample_weight = (y_reg - tau)[positive_mask].to(self.device, torch.float32)
        elif self.acf_type == "pi":
            sample_weight = torch.ones_like(positive_mask)
        else:
            sample_weight = torch.ones_like(positive_mask)

        if sample_weight.sum() == 0:
            print("sample weight is zero, using uniform weights")
            sample_weight = torch.ones_like(positive_mask)[positive_mask].to(self.device, torch.float32)
        else:
            sample_weight = sample_weight / sample_weight.sum()

        y_pred = y_pred.to(self.device)

        log_prob_positive = self.gamma * torch.mul(sample_weight, F.logsigmoid(y_pred[positive_mask])).mean()
        log_prob_negative = F.logsigmoid(-y_pred[negative_mask]).mean()

        loss = log_prob_positive + log_prob_negative
        loss = -loss  # Negate because we want to minimize
        return loss

    def _get_features(self, data):
        """
        Extract features from data. For toy benchmarks, features are already in the data.
        """
        if isinstance(data, dict):
            features = data["features"]
            if isinstance(features, list):
                features = torch.stack([torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in features])
            elif not isinstance(features, torch.Tensor):
                features = torch.tensor(np.stack(features))
        else:
            # Assume it's a batch from dataframe
            features = torch.stack([torch.tensor(row["features"]) if not isinstance(row["features"], torch.Tensor) else row["features"] for _, row in data.iterrows()])

        return features.to(self.device, dtype=self.ptdtype)

    def train_model(self, node=None):
        """
        Train the model by building the tree structure.
        This is a wrapper that calls tree_construction().
        """
        # Call tree construction to build the hierarchical tree
        self.tree_construction()

    def reset_to_root(self):
        """Reset current node to root"""
        self.CURT = self.ROOT

    def greedy_select(self):
        """Select the leaf node with the highest mean value (no exploration bonus)."""
        self.reset_to_root()
        curt_node = self.ROOT
        path = []
        while curt_node.is_leaf() == False:
            scores = []
            for i in curt_node.kids:
                scores.append(i.get_vbar())
            scores = np.nan_to_num(scores, nan=0)
            choice = np.random.choice(np.argwhere(scores == np.amax(scores)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            print("current node:", curt_node.get_name())
            print("greedy scores =>", scores)
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), end=' ')
        print("")
        if len(path) == 0:
            path.append((curt_node, None))
        return curt_node, path

    def select(self):
        """
        Select the leaf node and the path to it.
        selection_type="uct": UCT (mean + exploration bonus)
        selection_type="greedy": greedy (mean value only)
        """
        if self.selection_type == "greedy":
            return self.greedy_select()

        self.reset_to_root()
        curt_node = self.ROOT
        path = []

        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct2(self.lmbda))
                print("node mean and variance")
                print(i.v_bar, i.v_var)
                print(i.dataset[self.target_col] if self.target_col in i.dataset.columns else i.dataset["targets"])
            # print(UCT)
            UCT = np.nan_to_num(UCT, nan=0)
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            print("current node:", curt_node.get_name())
            print("UCT =>", UCT)
            curt_node = curt_node.kids[choice]
            print("next node:", curt_node.get_name())
        if len(path) == 0:
            path.append((curt_node, None))
        print("finish select")
        return curt_node, path

    def cal_acqfs(self, data, path):
        """
        Calculate acquisition function values for candidates.
        """
        node_acqfs = {}

        # Extract features
        features = self._get_features(data)
        targets = data["targets"].to(self.device, dtype=self.ptdtype)
        indices = data["Entry Number"].tolist()

        # Get predictions from current models
        self.clf_head.eval()
        self.reg_head.eval()

        # reg_preds = self.reg_head(features).squeeze()

        # Process each node in the path
        for node, choice in path:
            tau = node.tau if hasattr(node, 'tau') else np.quantile(self.training_set[self.target_col], q=1 - self.gamma)

            # # Calculate acquisition function
            # if self.acf_type == "ei":
            #     # Expected Improvement
            #     improvement = torch.clamp(reg_preds - tau, min=0)
            #     acq_vals = improvement * torch.sigmoid(clf_logits)
            # elif self.acf_type == "pi":
            #     # Probability of Improvement
            #     acq_vals = torch.sigmoid(clf_logits)
            # elif self.acf_type == "ucb":
            #     # Upper Confidence Bound
            #     acq_vals = reg_preds + self.lmbda * torch.abs(clf_logits)
            # else:
            #     acq_vals = torch.sigmoid(clf_logits)
            with torch.no_grad():
                self.clf_head.load_state_dict(node.clf)
                clf_logits = self.clf_head(features).squeeze()
                cx = torch.nn.functional.sigmoid(clf_logits)  # pred_probs
            # print("==========pred sigmoids============")
            # print(cx)
            acq_vals = (cx / (1 - cx))
            # Select candidates based on threshold
            promising_mask = acq_vals > self.threshold
            promising_indices = [indices[i] for i in range(len(indices)) if promising_mask[i]]

            ratio = len(promising_indices) / len(indices) if len(indices) > 0 else 0

            node_acqfs[node.id] = [
                tau,  # 0: tau
                ratio,  # 1: ratio of promising candidates
                promising_indices,  # 2: indices of promising candidates
                [acq_vals[promising_mask]],  # 3: acquisition values
                [targets[promising_mask]],  # 4: true targets
                # [reg_preds[promising_mask]],  # 5: predictions
                [features[promising_mask]],  # 5: features
            ]

        return node_acqfs

    def get_clusters(self, pval=0.05, n_clusters=5):
        """
        Select clusters using ANOVA test.
        """
        cluster_col = 'cluster' if 'cluster' in self.training_set else 'llm_cluster'

        if pval <= 0 or pval >= 1:
            # Return all clusters if p-value is invalid
            return list(range(n_clusters))

        # Perform ANOVA test
        try:
            aov = pg.anova(data=self.training_set, dv=self.target_col, between=cluster_col, detailed=True)
            p_value = aov['p-unc'].values[0]

            if p_value < pval:
                # Significant difference between clusters, return all
                return list(range(n_clusters))
            else:
                # No significant difference, return random subset
                n_select = max(1, n_clusters // 2)
                return np.random.choice(n_clusters, n_select, replace=False).tolist()
        except:
            # If ANOVA fails, return all clusters
            return list(range(n_clusters))

    def condition_on_observations(self, new_data):
        """
        Update the model with new observations.
        """
        # Add new data to training set
        self.training_set = pd.concat([self.training_set, pd.DataFrame([new_data])], ignore_index=True)

        # Retrain the model
        self.train_model()

    def init_nodes_with_train_data(self):
        """
        Initialize the root node with training data.
        """
        print("======== init root node ==========")
        # Clear previous nodes
        for node in self.nodes:
            if node is not None:
                node.clear_data()
        self.nodes.clear()

        print("========= raw training set =========")
        print(self.training_set)
        print(self.training_set.columns)

        # For toy benchmarks, features are already computed
        # Keep all columns from training set to maintain column names
        train_dataset = self.training_set.copy()

        print("========= processed training set with features =========")
        print(train_dataset)

        # Create new root node
        new_root = Node(parent=None, reset_id=True, tau=None, clf=None, is_splitable=True, threshold=self.threshold, target_col=self.target_col)
        self.nodes.append(new_root)
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_dataset(train_dataset)
        print("======== finish init root node ==========")

    def get_leaf_status(self):
        """
        Check which leaf nodes can be split.
        """
        status = []
        n_nodes = len(self.nodes)
        for node in self.nodes:
            if node.is_leaf() and node.n > self.LEAF_SAMPLE_SIZE and node.is_splitable:
                n_nodes += 2
                if n_nodes <= np.power(2, self.tree_depth) - 1:
                    status.append(True)
                else:
                    status.append(False)
            else:
                status.append(False)
        return np.array(status)

    def is_splitable(self):
        """
        Check if any nodes can be split.
        """
        status = self.get_leaf_status()
        print("leaf status:", status)
        return True in status

    def get_split_idx(self):
        """
        Get indices of nodes that can be split.
        """
        splitable_nodes_idx = np.argwhere(self.get_leaf_status() == True).reshape(-1)
        return splitable_nodes_idx

    def _finetune_head_clf(self, train_dataset, node):
        """
        Train classifier head and split node into positive/negative samples.
        Simplified version for toy benchmarks without LoRA.
        """
        tau = node.tau
        print(f"======== start training node {node.id} ==========")

        cfg = self.head_cfg

        # Convert to tensors
        features = torch.stack([torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in train_dataset["features"]])
        y_class = torch.tensor(train_dataset["labels"].values if hasattr(train_dataset["labels"], 'values') else train_dataset["labels"].tolist())
        y_reg = torch.tensor(train_dataset[self.target_col].values if hasattr(train_dataset[self.target_col], 'values') else train_dataset[self.target_col].tolist())

        features = features.to(self.device, dtype=self.ptdtype)
        y_class = y_class.to(self.device, dtype=torch.float32)
        y_reg = y_reg.to(self.device, dtype=self.ptdtype)

        # Load meta-model or parent's classifier
        if node.parent is not None:
            self.clf_head.load_state_dict(node.parent.clf)
        else:
            self.clf_head.load_state_dict(self.theta)

        self.clf_head = self.clf_head.to(self.device)
        self.clf_head.train()

        # Optimizer
        lr = float(cfg.get("lr", 1e-3))
        n_epochs = cfg.get("n_epochs", 100)
        optimizer = optim.Adam(self.clf_head.parameters(), lr=lr, weight_decay=5e-4)

        print(f"Training node {node.id} for {n_epochs} epochs...")

        # Training loop
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Forward pass
            clf_logits = self.clf_head(features).squeeze()

            # Classification loss
            loss = self._weighted_vi_loss(y_class, y_reg, clf_logits, tau)

            # Backward pass
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

        # Split samples based on predictions
        self.clf_head.eval()
        with torch.no_grad():
            clf_logits = self.clf_head(features).squeeze()
            pred_prob = torch.sigmoid(clf_logits)

        print("======= predicted sigmoids of train data ====")
        print("predicted prob:", pred_prob)

        # Split based on threshold
        pos_inds = torch.where(pred_prob >= node.threshold)[0].cpu().numpy()
        neg_inds = torch.where(pred_prob < node.threshold)[0].cpu().numpy()

        # Create DataFrames for positive and negative samples
        p_samples = train_dataset.iloc[pos_inds].copy()
        n_samples = train_dataset.iloc[neg_inds].copy()

        print("positive samples:", p_samples[[self.target_col, "labels"]] if self.target_col in p_samples.columns else p_samples)
        print("negative samples:", n_samples[[self.target_col, "labels"]] if self.target_col in n_samples.columns else n_samples)
        print("======== end training ==========")

        # Check if split is valid
        if len(p_samples) == 0 or len(n_samples) == 0:
            print("not splited")
            return False, p_samples, n_samples
        else:
            return True, p_samples, n_samples

    def train_and_split(self, node):
        """
        Train the classifier head and split the node.
        """
        print(f">>>>>>>> train node {node.id} >>>>>>>>>")
        assert node.n >= 2

        # Get training data with weights
        x_ori, x, y_class, y_cluster, y_reg, tau = self.get_traindata_weight(node.dataset)

        # Create training dataset - preserve original dataset structure
        train_dataset = node.dataset.copy()
        train_dataset["labels"] = y_class

        node.update_dataset(train_dataset)
        node.tau = tau

        # Train and split
        is_splitable, p_samples, n_samples = self._finetune_head_clf(train_dataset, node)
        node.clf = deepcopy(self.clf_head.state_dict())
        node.is_splitable = is_splitable

        if is_splitable:
            print(">>>>>>> node is splittable >>>>>")
        else:
            print(">>>>>>> node is not splittable >>>>>")

        assert len(p_samples) + len(n_samples) == node.n
        return p_samples, n_samples

    def tree_construction(self):
        """
        Construct the tree by splitting nodes hierarchically.
        """
        print("======== init nodes ==========")
        start_feature, end_feature = trace_times(start=None, end=None, device=self.device)
        self.init_nodes_with_train_data()
        print("time for prepare features:", trace_times(start_feature, end_feature, self.device))

        start_tree, end_tree = trace_times(start=None, end=None, device=self.device)
        print("======== train weighted classifiers for each feasible node ==========")

        assert self.ROOT.n == len(self.training_set)
        assert len(self.nodes) == 1

        self.theta = self.clf_head.state_dict()

        # Split nodes iteratively
        while self.is_splitable():
            to_split = self.get_split_idx()
            print(f"==>to split: {to_split}, total: {len(self.nodes)}")

            for nidx in to_split:
                parent = self.nodes[nidx]
                assert parent.n >= self.LEAF_SAMPLE_SIZE
                assert parent.is_splitable == True

                # Train and split the parent node
                good_kid_data, bad_kid_data = self.train_and_split(parent)

                # Update meta-model theta (only from root for toy version)
                if parent.parent is None:
                    for name in parent.clf:
                        self.theta[name] = (1 - self.eta) * self.theta[name].to(self.device) + self.eta * parent.clf[name].to(self.device)

                # Check split validity
                assert len(good_kid_data) + len(bad_kid_data) == parent.n
                if len(good_kid_data) == 0 or len(bad_kid_data) == 0:
                    print("no split")
                    continue

                # Create child nodes
                good_kid = Node(parent=parent, reset_id=False, tau=None, clf=None, is_splitable=True, threshold=self.threshold, target_col=self.target_col)
                bad_kid = Node(parent=parent, reset_id=False, tau=None, clf=None, is_splitable=True, threshold=self.threshold, target_col=self.target_col)

                good_kid.update_dataset(good_kid_data)
                bad_kid.update_dataset(bad_kid_data)

                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)

                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

                print(f"Split node {parent.id} into nodes {good_kid.id} and {bad_kid.id}")

        print("time for training node classifiers:", trace_times(start_tree, end_tree, self.device))
        print(f"Tree construction complete! Total nodes: {len(self.nodes)}")
