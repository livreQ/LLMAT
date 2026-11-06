#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025-11-26
# File Name: llmat_bo_toy.py
# Description: LLMAT for toy benchmarks without foundation models
"""

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import tqdm
import argparse
import sys
import os
import time
import math

from src.utils import helpers
from src.utils.helpers import trace_times
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import StandardScaler
from src.algos.llmat.llmat_toy import LLMAT_Toy
from torch.utils.data import Dataset, DataLoader


class FTDataset_Toy(Dataset):
    """Dataset for toy benchmarks without SMILES"""

    def __init__(self, dataframe, cluster_col=None):
        self.dataframe = dataframe
        self.cluster_col = cluster_col
        print("DataFrame columns:", self.dataframe.columns)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Extract features and targets from the row
        features = torch.tensor(row["features"]) if not isinstance(row["features"], torch.Tensor) else row["features"]

        if "Entry Number" in self.dataframe.columns:
            index = row["Entry Number"]
            targets = torch.tensor(row["targets_transformed"]) if "targets_transformed" in self.dataframe.columns else torch.tensor(row["targets"])
            clusters = torch.tensor(row[self.cluster_col]) if self.cluster_col in self.dataframe.columns else torch.tensor(0)
            return {
                "Entry Number": index,
                "features": features,
                "targets": targets,
                "clusters": clusters,
            }
        else:
            # targets = torch.tensor(row["targets"])
            targets = torch.tensor(row["targets_transformed"]) if "targets_transformed" in self.dataframe.columns else torch.tensor(row["targets"])
            clusters = torch.tensor(row["clusters"]) if "clusters" in self.dataframe.columns else torch.tensor(0)
            labels = torch.tensor(row["labels"]) if "labels" in self.dataframe.columns else torch.tensor(0)
            return {
                "features": features,
                "targets": targets,
                "labels": labels,
                "clusters": clusters,
            }


device = helpers.check_device()
print(f"Using device: {device}")


class ToyExpRunner:

    def __init__(self, toy_bench, seed, normalize_y, list_init_points=None):
        self.toy_bench = toy_bench  # not deep copied
        self.seed = seed
        self.list_init_points = list_init_points
        self.ground_truth_max_transformed = None
        self.normalize_y = normalize_y

    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''
        if self.list_init_points is not None:
            init_points = self.list_init_points[:n_samples]
        else:
            init_points = []
            dataset = self.toy_bench.dataset
            target_col_transformed = self.toy_bench.target_col_transformed
            if self.normalize_y:
                y_preprocessor = StandardScaler()
                dataset[target_col_transformed] = y_preprocessor.fit_transform(dataset[target_col_transformed].to_numpy().reshape(-1, 1)).flatten()
            ground_truth_opt_id = dataset[target_col_transformed].idxmax()
            self.ground_truth_max_transformed = dataset.loc[ground_truth_opt_id][target_col_transformed]
            print("ground_truth_max", self.toy_bench.ground_truth_max)
            print("ground_truth_opt", self.toy_bench.ground_truth_opt)
            print("ground_truth_max_transformed", self.ground_truth_max_transformed)

            init_cluster_cnt = {cluster: 0 for cluster in range(self.toy_bench.n_clusters)}
            while len(init_points) < n_samples:
                idx = np.random.randint(len(dataset))
                # Make sure that the optimum is not included
                if dataset.loc[idx][self.toy_bench.target_col_transformed] >= self.ground_truth_max_transformed:
                    continue
                cluster = dataset.loc[idx][self.toy_bench._get_cluster_col()]
                if init_cluster_cnt[cluster] >= n_samples / self.toy_bench.n_clusters:
                    continue
                else:
                    init_cluster_cnt[cluster] += 1
                init_points.append(helpers.pop_df(dataset, idx))
            self.toy_bench.dataset = dataset
        return pd.DataFrame(init_points)

    def evaluate_point(self, candidate):
        '''
        Evaluate a single point on bbox
        '''
        label = self.toy_bench.complete_call(candidate)
        return candidate, label


def get_ablation_suffix(args):
    p_val = args.p_val
    if p_val > 0:
        suffix = f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}_{args.tree_depth}_{p_val}_{args.seed}"
    else:
        suffix = f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}_{args.tree_depth}_{args.seed}"
    sel = getattr(args, "selection_type", "uct")
    if sel == "greedy":
        suffix += "_greedy"
    return suffix


def save_results(args, toy_bench, timing_train,\
                            timing_preds, trace_acqvals, trace_y_his,  trace_best_y, trace_timing):

    print(toy_bench.dataset_name)
    prefix = "/".join(toy_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/{args.algorithm}"
    if args.algorithm == "llmat":
        clustering = toy_bench.dataset_name.split("/")[-2]
        path = f"{path}/{clustering}"
    if not os.path.exists(path):
        os.makedirs(path)
    suffix = get_ablation_suffix(args)
    np.save(f"{path}/timing_train_{suffix}.npy", timing_train)
    np.save(f"{path}/timing_preds_{suffix}.npy", timing_preds)
    np.save(f"{path}/trace_acqvals_{suffix}.npy", trace_acqvals)
    np.save(f"{path}/trace_best_y_{suffix}.npy", trace_best_y)
    np.save(f"{path}/trace_y_his_{suffix}.npy", trace_y_his)
    np.save(f"{path}/trace_timing_{suffix}.npy", trace_timing)


#=============================================
def select_from_leaf_to_root(node_acqfs_all, node_ids):
    """
    Select the candidates from leaf to root
    Args:
        node_acqfs_all: {node id: tau, ratio, inds, acq_val, targets, preds}
        node_ids: list of node ids
    Returns:
        The selected candidates at the leaf node, if it is empty, backtrack to its parent node
        candidates_idx: list of candidates
        y_reg: list of targets
        y_preds: list of predictions
        y_class: list of labels
        acq_vals: list of acquisition values
    """
    node_ids.reverse()
    print(node_ids)
    for node_id in node_ids:
        candidates_idx = node_acqfs_all[node_id][2]
        if len(candidates_idx) == 0:
            print("no candidates found in this node")  # backtrack to the parent node
            continue
        acq_vals = node_acqfs_all[node_id][3]
        y_reg = node_acqfs_all[node_id][4]
        y_preds = node_acqfs_all[node_id][5]
        y_class = node_acqfs_all[node_id][6]
        print(y_reg)

        acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
        y_preds = torch.cat(y_preds, dim=0).cpu().squeeze()  # Added squeeze
        y_reg = torch.cat(y_reg, dim=0).squeeze()  # Added squeeze

        y_class = torch.cat(y_class, dim=0).squeeze()  # Added squeeze

        if y_reg.ndim == 0:
            print("Tensor is 0D (scalar)")
            y_reg = y_reg.unsqueeze(0)
        if acq_vals.ndim == 0:
            print("Tensor is 0D (scalar)")
            acq_vals = acq_vals.unsqueeze(0)
        if y_preds.ndim == 0:
            print("Tensor is 0D (scalar)")
            y_preds = y_preds.unsqueeze(0)
        if y_class.ndim == 0:
            print("Tensor is 0D (scalar)")
            y_class = y_class.unsqueeze(0)
        assert len(y_reg) == len(acq_vals)
        assert len(candidates_idx) > 0
        return candidates_idx, y_reg, y_preds, y_class, acq_vals

    # If no candidates found in any node, return None to signal this condition
    print("WARNING: No candidates found in any node in the path!")
    return None


def run_bo_toy(args, toy_bench, wandb=None):
    """
    MCTS partition selection and LFBO without foundation models (for toy benchmarks)
    """
    import numpy as np
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # No need for foundation models or LoRA for toy benchmarks
    head_cfg = args.fix_args["head"]

    # Feature dimension is simply the dimension of the input features
    feature_dim = len(toy_bench.dataset['features'].iloc[0])
    print("feature dim:", feature_dim)

    # No data processor needed for toy benchmarks
    data_processor = None
    print("========== no data processor needed for toy benchmarks =========")
    print("---- benchmark dataset head ----")
    print(toy_bench.dataset.head())
    toy_runner = ToyExpRunner(toy_bench, seed, args.normalize_y)
    # n_init_data samples for first training
    dataset_train = toy_runner.generate_initialization(args.n_init_data)  # pd dataframe
    target_col_transformed = toy_bench.target_col_transformed
    target_col = toy_bench.target_col
    print("==========finish benchmark runner init=========")
    print("ground truth max:", toy_runner.ground_truth_max_transformed)
    print("ground truth optimal:", toy_bench.ground_truth_opt)
    if not args.normalize_y:
        if toy_bench.maximization:
            assert toy_runner.ground_truth_max_transformed == toy_bench.ground_truth_opt
        else:
            assert toy_runner.ground_truth_max_transformed == -toy_bench.ground_truth_opt

    # Initialize Alpha model without foundation model
    print("========== init Alpha model (no LLM) =========")
    alpha_model = LLMAT_Toy(
        dataset_train,
        head_cfg,
        device,
        gamma=args.gamma,  # quantile
        acf_type=args.acqf,
        lmbda=args.lmbda,  # ucb weight
        eta=args.eta,  # meta learning rate
        tree_depth=args.tree_depth,
        threshold=args.threshold,
        alpha=args.alpha,
        feature_dim=feature_dim,
        target_col=target_col_transformed,  # Use transformed target column
        selection_type=getattr(args, "selection_type", "uct"),
    )
    print("--------finish init----------")
    # Build the tree with hierarchical partitioning
    alpha_model.tree_construction()
    print("============finish building ALPHA tree==============")

    id_max_observed = dataset_train[target_col_transformed].idxmax()
    max_y_observed = dataset_train.loc[id_max_observed][target_col_transformed]
    opt_y_observed = dataset_train.loc[id_max_observed][target_col]
    initial_opt_y = opt_y_observed
    best_y = max_y_observed
    #----------------  trace --------------------
    true_best = toy_bench.ground_truth_opt
    trace_best_y = [true_best] * (args.exp_len + 1)
    trace_y_his = [true_best] * (args.exp_len + 1)
    trace_timing = [0.0] * (args.exp_len + 1)
    trace_acqvals = [-math.inf] * (args.exp_len + 1)
    timing_train = []
    timing_preds = []
    #--------------- BO ----------------
    pbar_bo_iters = tqdm.trange(args.exp_len, position=0, colour="green", leave=True)
    pbar_bo_iters.set_description(f"[Best Observed f(x) = {opt_y_observed:.3f}]")

    for t in pbar_bo_iters:
        # Timing
        print(device)
        start, end = trace_times(None, None, device)
        print("==========")
        # BO iteration
        true_best = toy_bench.ground_truth_opt
        cluster_col = toy_bench._get_cluster_col()

        #-------------------- estimate acf ---------------------------
        # We first select the clusters to be estimated
        # by using ANOVA
        clusters = alpha_model.get_clusters(pval=args.p_val, n_clusters=toy_bench.n_clusters)
        dataset = toy_bench.dataset
        print("========== chosen cluster to estimate =======")
        print(clusters)
        dataset = dataset[dataset[cluster_col].isin(clusters)]
        print("=========unobserved dataset ==========")
        print(dataset)

        # At each BO round we select a most promising Path
        # and then estimate the acquisition function of the unobserved data
        # that predicted to fall into nodes along this path
        leaf, path = alpha_model.select()
        node_acqfs_all = {}  # for all the batches
        node_ids = []

        for node, choice in path:
            #{node id: tau, ratio, inds, acq_val, targets, preds, labels, features}
            node_acqfs_all[node.id] = [0, [], [], [], [], [], [], []]
            node_ids.append(node.id)
            if choice == 0:
                print("choose left child")
            elif choice == 1:
                print("choose right child")
            else:
                print("unimplemented choice")

        ## get data loader - simplified for toy benchmarks
        ft_dataset = FTDataset_Toy(dataset, toy_bench._get_cluster_col())
        dataloader = DataLoader(
            ft_dataset,
            batch_size=args.fix_args["batch_size"],
            shuffle=False,
        )

        print("data loader batch size", dataloader.batch_size)
        sub_pbar = tqdm.tqdm(dataloader, position=1, colour="blue", \
                             desc="[Prediction over dataset]", leave=False,)
        print("-----------start BO------------")
        start_pred, end_pred = trace_times(None, None, device)
        for data in sub_pbar:
            print("======== BO batch size ===========")
            print("BO batch size:", len(data["targets"]))
            # {node id: tau, ratio, inds, acq_val, targets, preds}
            node_acqfs = alpha_model.cal_acqfs(data, path)
            for node_id in node_ids:
                if node_id not in node_acqfs:
                    print(f"node {node_id} has no candidates in this batch")
                    continue
                node_acqfs_all[node_id][0] = node_acqfs[node_id][0]  # tau
                node_acqfs_all[node_id][1].append(node_acqfs[node_id][1])  # ratio
                node_acqfs_all[node_id][2] += node_acqfs[node_id][2]  # inds
                node_acqfs_all[node_id][3] += [t.cpu() for t in node_acqfs[node_id][3]]  # acq_val
                node_acqfs_all[node_id][4] += [t.cpu() for t in node_acqfs[node_id][4]]  # targets
                node_acqfs_all[node_id][5] += [t.cpu() for t in node_acqfs[node_id][5]]  # preds
                labels = torch.greater_equal(node_acqfs[node_id][4][0], node_acqfs[node_id][0])
                node_acqfs_all[node_id][6] += [labels.cpu()]  # labels
                # node_acqfs_all[node_id][7] += [t.cpu() for t in node_acqfs[node_id][6]]  # candidate features in each node
            if node_id in node_acqfs:
                print(">>>>>>>>>>leaf candidates ratio>>>>>>>>>>>>", node_acqfs[node_id][1])
                print("-----------***********------------")
                if node_acqfs[node_id][2] == 0:
                    print("no promising leaf candidates in this batch")

        # Record the end time for predictions
        timing_preds.append(trace_times(start_pred, end_pred, device))

        result = select_from_leaf_to_root(node_acqfs_all, node_ids)

        # If no candidates found in any node, select randomly from the dataset
        if result is None:
            print("No candidates found with threshold, selecting randomly from dataset...")
            random_idx = np.random.randint(len(dataset))
            index = dataset.index[random_idx]
            new_data = helpers.pop_df(toy_bench.dataset, index)

            # Update the current best y
            if new_data[target_col_transformed] > best_y:
                best_y = new_data[target_col_transformed]
                opt_y_observed = new_data[target_col]

            # Update model
            start_train, end_train = trace_times(start=None, end=None, device=device)
            alpha_model.condition_on_observations(new_data)
            timing_train.append(trace_times(start_train, end_train, device))

            pbar_bo_iters.set_description(f"[Best f(x) = {opt_y_observed:.3f}, " + f"True Best f(x) = {true_best :.3f}," + f"curr f(x) = {new_data[target_col]:.3f},")

            # Save results
            timing = trace_times(start, end, device)
            trace_best_y[t + 1] = opt_y_observed
            trace_y_his[t + 1] = new_data[target_col]
            trace_timing[t + 1] = timing
            wandb.log({"trace_best_y": opt_y_observed}, step=t)
            wandb.log({"trace_y_his": new_data[target_col]}, step=t)
            if toy_bench.maximization:
                regret = toy_bench.ground_truth_opt - np.mean(trace_y_his[1:t + 2])
            else:
                regret = np.mean(trace_y_his[1:t + 2]) - toy_bench.ground_truth_opt
            wandb.log({"trace_regret": regret}, step=t)
            GAP = np.nan_to_num((opt_y_observed - initial_opt_y) / (toy_bench.ground_truth_opt - initial_opt_y), nan=1.0)
            wandb.log({"trace_gap": GAP}, step=t)
            wandb.log({"trace_timing": timing}, step=t)
            wandb.log({"trace_timing_train": timing_train[-1]}, step=t)
            wandb.log({"trace_timing_pred": timing_preds[-1]}, step=t)
            wandb.log({"trace_acqvals": 0.0}, step=t)
            continue

        candidates_idx, y_reg, y_preds, y_class, acq_vals = result
        print("find optimal x for acq func")

        _, idx = acq_vals.topk(k=min(10, len(acq_vals)))

        # Get the indexed values and flatten them to ensure 1D
        y_reg_selected = y_reg[idx].flatten()
        y_class_selected = y_class[idx].flatten()
        y_preds_selected = y_preds[idx].flatten()
        acq_vals_selected = acq_vals[idx].flatten()

        # Convert to numpy for easier iteration
        import numpy as np
        y_reg_np = y_reg_selected.cpu().numpy() if hasattr(y_reg_selected, 'cpu') else np.array(y_reg_selected)
        y_class_np = y_class_selected.cpu().numpy() if hasattr(y_class_selected, 'cpu') else np.array(y_class_selected)
        y_preds_np = y_preds_selected.cpu().numpy() if hasattr(y_preds_selected, 'cpu') else np.array(y_preds_selected)
        acq_vals_np = acq_vals_selected.cpu().numpy() if hasattr(acq_vals_selected, 'cpu') else np.array(acq_vals_selected)

        # Ensure 1D arrays
        y_reg_np = y_reg_np.ravel()
        y_class_np = y_class_np.ravel()
        y_preds_np = y_preds_np.ravel()
        acq_vals_np = acq_vals_np.ravel()

        for y_r, y_c, y_p, acqf in zip(y_reg_np, y_class_np, y_preds_np, acq_vals_np):
            print(f"True: {y_r:.3f}, Class:{y_c:.3f}, Pred: {y_p:.3f}, Acqf: {acqf:.3f}")

        print("idx device:", idx.device)
        print("find optimal x for true value")
        _, idx2 = y_reg.topk(k=min(10, len(acq_vals)))
        idx2 = idx2.to("cpu")
        print("idx device:", idx2.device)

        # Flatten indexed tensors before iteration
        y_reg_top = y_reg[idx2].flatten()
        y_class_top = y_class[idx2].flatten()
        y_preds_top = y_preds[idx2].flatten()
        acq_vals_top = acq_vals[idx2].flatten()

        for y_r, y_c, y_p, acqf in zip(y_reg_top, y_class_top, y_preds_top, acq_vals_top):
            print(f"True: {y_r.item():.3f}, Class:{y_c.item():.3f}, Pred: {y_p.item():.3f}, Acqf: {acqf.item():.3f}")

        print("========path leaf candidates indexes=======")
        print("number of selected candidates: " + str(len(candidates_idx)))
        print(idx)
        print(dataset)
        print(dataset.index)

        # Pick a point (a row in the current dataset) that maximizes the acquisition
        candidates = dataset[dataset["Entry Number"].isin(candidates_idx)]
        print("========== path leaf candidates=======")
        print(candidates)
        assert len(candidates) == len(candidates_idx), "candidates and candidates_idx should have the same length"

        idx_best = torch.argmax(acq_vals).item()
        print(">>>>>>best id in candidates", idx_best)
        x_can = candidates.iloc[idx_best]
        entry_can = candidates_idx[idx_best]
        assert x_can["Entry Number"] == entry_can, "x_can should have the same Entry Number as entry_can"
        print(">>>>>>selected candidate>>>>", x_can["Entry Number"], x_can[target_col_transformed], x_can[target_col])

        index = dataset[dataset['Entry Number'] == candidates_idx[idx_best]].index[0]
        new_data = helpers.pop_df(toy_bench.dataset, index)  #x,y

        # Update the current best y
        if new_data[target_col_transformed] > best_y:
            best_y = new_data[target_col_transformed]
            opt_y_observed = new_data[target_col]

        # record the start time for training
        start_train, end_train = trace_times(start=None, end=None, device=device)
        # -------------------- Update surrogate --------------------
        # retrain the model
        print("============== retrain model ======================")
        alpha_model.condition_on_observations(new_data)  # update train dataset, retrain model
        # record the end time for training
        timing_train.append(trace_times(start_train, end_train, device))

        pbar_bo_iters.set_description(f"[Best f(x) = {opt_y_observed:.3f}, " + f"True Best f(x) = {true_best :.3f}," + f"curr f(x) = {new_data[target_col]:.3f},")

        # Save results
        # record the end time
        timing = trace_times(start, end, device)
        trace_best_y[t + 1] = opt_y_observed
        trace_y_his[t + 1] = new_data[target_col]
        trace_timing[t + 1] = timing
        wandb.log({"trace_best_y": opt_y_observed}, step=t)
        wandb.log({"trace_y_his": new_data[target_col]}, step=t)
        if toy_bench.maximization:
            regret = toy_bench.ground_truth_opt - np.mean(trace_y_his[1:t + 2])
        else:
            regret = np.mean(trace_y_his[1:t + 2]) - toy_bench.ground_truth_opt
        wandb.log({"trace_regret": regret}, step=t)
        GAP = np.nan_to_num((opt_y_observed - initial_opt_y) / (toy_bench.ground_truth_opt - initial_opt_y), nan=1.0)
        wandb.log({"trace_gap": GAP}, step=t)
        wandb.log({"trace_timing": timing}, step=t)
        wandb.log({"trace_timing_train": timing_train[-1]}, step=t)
        wandb.log({"trace_timing_pred": timing_preds[-1]}, step=t)
        wandb.log({"trace_acqvals": acq_vals[idx_best].item()}, step=t)

        prefix = "/".join(toy_bench.dataset_name.split("/")[:-2])
        path = f"results/{prefix}/{args.algorithm}"
        if args.algorithm == "llmat":
            clustering = toy_bench.dataset_name.split("/")[-2]
            path = f"{path}/{clustering}"
        if not os.path.exists(path):
            os.makedirs(path)
        suffix = get_ablation_suffix(args)

        if args.early_stopping:
            # Early stopping if we already got the max
            if best_y >= toy_runner.ground_truth_max_transformed:
                for j in range(t + 1, args.exp_len + 1):
                    wandb.log({"trace_best_y": trace_best_y[t + 1]}, step=j)
                    wandb.log({"trace_y_his": trace_y_his[t + 1]}, step=j)
                    wandb.log({"trace_timing": timing}, step=j)
                    wandb.log({"trace_timing_train": timing_train[-1]}, step=j)
                    wandb.log({"trace_timing_pred": timing_preds[-1]}, step=j)
                    wandb.log({"trace_acqvals": 0}, step=j)
                    if toy_bench.maximization:
                        regret = toy_bench.ground_truth_opt - np.mean(trace_y_his[1:j + 2])
                    else:
                        regret = np.mean(trace_y_his[1:j + 2]) - toy_bench.ground_truth_opt
                    GAP = np.nan_to_num((opt_y_observed - initial_opt_y) / (toy_bench.ground_truth_opt - initial_opt_y), nan=1.0)
                    wandb.log({"trace_gap": GAP}, step=j)
                    wandb.log({"trace_regret": regret}, step=j)
                break

    save_results(args, toy_runner.toy_bench, timing_train, timing_preds, trace_acqvals, trace_y_his, trace_best_y, trace_timing)
    print("====== trace best y======")
    print(trace_best_y)

    return trace_best_y, None


def llmat_bo_toy(args, toy_bench, wandb=None):
    """
    Run the LLMAT algorithm on toy benchmarks without foundation models.
    Args:
        args: command line arguments
        toy_bench: the toy benchmark to run on
        wandb: wandb logger (optional)
    Returns:
        trace_best_y: the best y value found during the optimization
        all_metrics: all metrics collected during the optimization
    """

    print(">>>>> Enter BO for Toy Benchmarks >>>>>")
    if args.benchmark == 'toy':
        trace_best_y, all_metrics = run_bo_toy(args, toy_bench, wandb=wandb)
    else:
        raise NotImplementedError(f"Benchmark {args.benchmark} not supported in toy BO")

    return trace_best_y, all_metrics
