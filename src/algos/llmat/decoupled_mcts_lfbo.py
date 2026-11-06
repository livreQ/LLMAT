#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Decoupled Partitioning + Independent LFBO ablation.

Replaces LLMAT's coupled classifier with two separate mechanisms:
  1. Binary K-means on [standardized features ++ standardized target] to
     produce a target-aware good/bad split of the observed data, then a
     logistic-regression partition classifier fit on (features → good/bad
     label) so unseen candidates (which have no target) can be routed.
     The cluster with higher mean target → good kid.
  2. Independent LFBO acquisition classifier per node for scoring.

Each node's LFBO head is trained independently (no shared θ / meta-gradient).
Candidates are routed through the tree via the per-node logistic-regression
partition classifier, then scored by the leaf's LFBO head.

This isolates the gain from coupling: LLMAT uses ONE classifier for both
partitioning and acquisition, while this ablation uses K-means + a supervised
partition classifier for routing and a separate LFBO head for acquisition.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class DecoupledNode:
    """Tree node for DecoupledPartitionLFBO."""

    obj_counter = 0

    def __init__(self, parent=None, reset_id: bool = False):
        if reset_id:
            DecoupledNode.obj_counter = 0
        self.id = DecoupledNode.obj_counter
        DecoupledNode.obj_counter += 1

        self.parent = parent
        self.kids: list = []           # [good_kid, bad_kid]

        # --- dataset held at this node ---
        self.dataset = None
        self.n: int = 0
        self.v_bar: float = float("inf")
        self.v_var: float = 0.0

        # --- partition (target-aware K-means + supervised routing) ---
        self.feat_scaler = None        # fitted StandardScaler on raw features
        self.partition_clf = None      # LogisticRegression(features → good/bad)

        # --- acquisition classifier (independent LFBO head) ---
        self.lfbo_clf: dict = None     # state_dict
        self.lfbo_tau: float = None

    # -----------------------------------------------------------------------

    def is_leaf(self) -> bool:
        return len(self.kids) == 0

    def update_dataset(self, df) -> None:
        self.dataset = df
        self.n = len(df)
        tgts = df["targets"].values.astype(float)
        self.v_bar = float(tgts.mean())
        self.v_var = float(np.var(tgts, ddof=0))

    def get_uct(self, lmbda: float = 1.0) -> float:
        return self.v_bar + 2.0 * lmbda * np.sqrt(self.v_var + 1e-8)

    def get_vbar(self) -> float:
        return self.v_bar

    def route(self, features: np.ndarray) -> np.ndarray:
        """Route candidates: True = good kid, False = bad kid."""
        z = self.feat_scaler.transform(features)
        return self.partition_clf.predict(z).astype(bool)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DecoupledPartitionLFBO:
    """
    Binary K-means tree + independent LFBO per node.

    Partition: K-means on [standardized features ++ standardized target]
    splits observed data at each node. Good kid = cluster with higher mean
    target. A logistic-regression classifier is then fit on (features →
    good/bad label) so unseen candidates, which have no target, can be
    routed in feature space only.

    Acquisition: a separate LFBO classifier per node, trained with the
    weighted VI loss, scores candidates that arrive at that node.

    Public API (mirrors LLMAT so the BO runner can swap them):
      train(training_set)            → build tree, fit per-node heads
      condition_on_observations(obs) → append obs, retrain
      select()                       → (leaf_node, path)
      get_acq_vals(candidates, path, leaf) → acq values filtered along path
    """

    def __init__(
        self,
        feature_dim: int,
        tree_depth: int = 3,
        leaf_sample_size: int = 2,
        gamma: float = 0.5,
        acf_type: str = "ei",
        lmbda: float = 1.0,
        device: str = "cpu",
        head_cfg: dict = None,
        selection_type: str = "uct",
    ):
        self.feature_dim = feature_dim
        self.tree_depth = tree_depth
        self.leaf_sample_size = leaf_sample_size
        self.gamma = gamma
        self.acf_type = acf_type
        self.lmbda = lmbda
        self.device = device
        self.head_cfg = head_cfg or {"lr": 1e-3, "n_epochs": 100}
        self.selection_type = selection_type

        self.training_set = None
        self.ROOT: DecoupledNode = None
        self.nodes: list = []

    # -----------------------------------------------------------------------
    # Network factory
    # -----------------------------------------------------------------------

    def _make_head(self) -> nn.Module:
        last_hidden = 50
        head = nn.Sequential(
            nn.Linear(self.feature_dim, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1),
        ).to(self.device)
        for p in head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return head

    # -----------------------------------------------------------------------
    # LFBO loss (identical to LLMAT._weighted_vi_loss)
    # -----------------------------------------------------------------------

    def _lfbo_loss(
        self,
        y_class: torch.Tensor,
        y_reg: torch.Tensor,
        logits: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        pos = y_class == 1
        neg = y_class == 0
        if pos.sum() == 0 or neg.sum() == 0:
            return torch.zeros(1, requires_grad=True, device=self.device).squeeze()

        if self.acf_type == "ei":
            w = (y_reg - tau)[pos].clamp(min=0.0).to(self.device)
        else:
            w = torch.ones(int(pos.sum()), device=self.device)

        if w.sum() < 1e-10:
            w = torch.ones_like(w)
        else:
            w = w / w.sum()

        lp = self.gamma * (w * F.logsigmoid(logits[pos])).mean()
        ln = F.logsigmoid(-logits[neg]).mean()
        return -(lp + ln)

    # -----------------------------------------------------------------------
    # Train independent LFBO head for ONE node
    # -----------------------------------------------------------------------

    def _train_lfbo_head(self, df) -> tuple:
        features = np.stack(df["features"].tolist()).astype(np.float32)
        targets = df["targets"].values.astype(np.float32)
        tau = float(np.quantile(targets, 1.0 - self.gamma))

        head = self._make_head()
        head.train()

        x = torch.tensor(features, device=self.device)
        y_reg = torch.tensor(targets, device=self.device)
        y_class = (y_reg >= tau).float()

        lr = float(self.head_cfg.get("lr", 1e-3))
        n_epochs = int(self.head_cfg.get("n_epochs", 100))
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=5e-4)

        for _ in range(n_epochs):
            opt.zero_grad()
            loss = self._lfbo_loss(y_class, y_reg, head(x).squeeze(1), tau)
            loss.backward()
            opt.step()

        head.eval()
        return deepcopy(head.state_dict()), tau

    # -----------------------------------------------------------------------
    # Binary K-means split (target-aware clustering + supervised routing)
    # -----------------------------------------------------------------------

    def _binary_kmeans_split(self, df):
        features = np.stack(df["features"].tolist()).astype(np.float32)
        targets = df["targets"].values.astype(np.float32)

        feat_scaler = StandardScaler().fit(features)
        feat_z = feat_scaler.transform(features)

        tgt_scaler = StandardScaler().fit(targets.reshape(-1, 1))
        tgt_z = tgt_scaler.transform(targets.reshape(-1, 1))

        # Cluster on features ++ target so the split is target-aware, not
        # just geometry-based.
        cluster_space = np.concatenate([feat_z, tgt_z], axis=1)

        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(cluster_space)

        mask0 = labels == 0
        mask1 = labels == 1
        if not mask0.any() or not mask1.any():
            return None

        mean0 = targets[mask0].mean()
        mean1 = targets[mask1].mean()
        good_cluster = 0 if mean0 >= mean1 else 1

        y_partition = (labels == good_cluster).astype(int)
        if len(np.unique(y_partition)) < 2:
            return None

        # Supervised partition classifier: features → good/bad label, so
        # unseen candidates (no target available) can still be routed.
        partition_clf = LogisticRegression(max_iter=1000).fit(feat_z, y_partition)

        good_df = df.iloc[labels == good_cluster].reset_index(drop=True)
        bad_df  = df.iloc[labels != good_cluster].reset_index(drop=True)

        if len(good_df) == 0 or len(bad_df) == 0:
            return None

        return good_df, bad_df, feat_scaler, partition_clf

    # -----------------------------------------------------------------------
    # Tree construction
    # -----------------------------------------------------------------------

    def _build_tree(self, node: DecoupledNode, depth: int) -> None:
        node.lfbo_clf, node.lfbo_tau = self._train_lfbo_head(node.dataset)

        n_nodes_after_split = len(self.nodes) + 2
        max_nodes = 2 ** self.tree_depth - 1
        if (depth >= self.tree_depth - 1
                or node.n < self.leaf_sample_size * 2
                or n_nodes_after_split > max_nodes):
            return

        result = self._binary_kmeans_split(node.dataset)
        if result is None:
            return

        good_df, bad_df, feat_scaler, partition_clf = result

        node.feat_scaler = feat_scaler
        node.partition_clf = partition_clf

        good_kid = DecoupledNode(parent=node)
        bad_kid  = DecoupledNode(parent=node)
        good_kid.update_dataset(good_df)
        bad_kid.update_dataset(bad_df)
        node.kids = [good_kid, bad_kid]

        self.nodes.extend([good_kid, bad_kid])

        print(f"[DecoupledLFBO] Split node {node.id} (n={node.n}) → "
              f"good(n={good_kid.n}, v̄={good_kid.v_bar:.4f}), "
              f"bad(n={bad_kid.n}, v̄={bad_kid.v_bar:.4f})")

        self._build_tree(good_kid, depth + 1)
        self._build_tree(bad_kid,  depth + 1)

    def train(self, training_set) -> None:
        self.training_set = training_set
        DecoupledNode.obj_counter = 0
        self.nodes = []

        root = DecoupledNode(parent=None, reset_id=True)
        root.update_dataset(training_set)
        self.nodes.append(root)
        self.ROOT = root

        self._build_tree(root, depth=0)
        print(f"[DecoupledLFBO] Tree built: {len(self.nodes)} nodes, "
              f"depth up to {self.tree_depth}")

    def condition_on_observations(self, obs) -> None:
        self.training_set = self.training_set._append(obs, ignore_index=True)
        self.train(self.training_set)

    # -----------------------------------------------------------------------
    # Selection (UCT / greedy)
    # -----------------------------------------------------------------------

    def select(self):
        node = self.ROOT
        path = []

        while not node.is_leaf():
            if self.selection_type == "greedy":
                scores = [k.get_vbar() for k in node.kids]
            else:
                scores = [k.get_uct(self.lmbda) for k in node.kids]

            scores = np.nan_to_num(scores, nan=0.0)
            choice = int(np.argmax(scores))
            path.append((node, choice))
            node = node.kids[choice]

        return node, path

    # -----------------------------------------------------------------------
    # Acquisition scoring — route candidates, score at leaf
    # -----------------------------------------------------------------------

    def _score_with_head(
        self, candidate_features: np.ndarray, mask: np.ndarray, node,
    ) -> torch.Tensor:
        """Score candidates[mask] using node's LFBO head. Returns full-length tensor."""
        n = len(candidate_features)
        acq = torch.zeros(n)
        if mask.sum() == 0 or node.lfbo_clf is None:
            return acq
        head = self._make_head()
        head.load_state_dict(node.lfbo_clf)
        head.eval()
        x = torch.tensor(
            candidate_features[mask].astype(np.float32),
            device=self.device,
        )
        with torch.no_grad():
            logits = head(x)
            cx = torch.sigmoid(logits)
            scores = (cx / (1.0 - cx + 1e-8)).cpu().squeeze(1)
        acq[mask] = scores
        return acq

    def get_acq_vals(self, candidate_features: np.ndarray, path, leaf) -> tuple:
        """
        Route candidates down the path via K-means, score at every node
        along the path, then backtrack from leaf to root if the leaf has
        no surviving candidates.

        Returns (alive_mask, acq_vals) from the deepest node that has
        surviving candidates.
        """
        n = len(candidate_features)

        # Score and route at each node along the path (root → parent of leaf)
        # node_results: list of (node, alive_mask, acq_vals) from root→leaf
        node_results = []
        alive = np.ones(n, dtype=bool)

        for node, choice in path:
            # Score all currently alive candidates at this node
            acq = self._score_with_head(candidate_features, alive, node)
            node_results.append((node, alive.copy(), acq))

            # Then route for the next level
            if node.partition_clf is None:
                break

            routed_to_good = node.route(candidate_features)

            if choice == 0:     # selected good kid
                alive &= routed_to_good
            else:               # selected bad kid
                alive &= ~routed_to_good

        # Score at the leaf itself
        leaf_acq = self._score_with_head(candidate_features, alive, leaf)
        node_results.append((leaf, alive.copy(), leaf_acq))

        # Backtrack: pick the deepest node with surviving candidates
        for node, mask, acq in reversed(node_results):
            if mask.sum() > 0:
                print(f"[DecoupledLFBO] Scoring from node {node.id} "
                      f"(n_alive={int(mask.sum())})")
                return mask, acq

        # Should not happen, but fallback
        return np.ones(n, dtype=bool), torch.zeros(n)
