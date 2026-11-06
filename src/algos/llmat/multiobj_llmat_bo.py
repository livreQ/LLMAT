#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: multiobj_llmat_bo.py
# Description: Multi-objective LLMAT implementation
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
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import StandardScaler
from src.algos.llmat.multi_llmat import MultiLLMAT
from src.algos.llmat.multi_llmat import MultiObjFTDataset
from torch.utils.data import DataLoader
from src.algos.llmat.llms.base_llms import get_model
from src.utils import helpers
from src.utils.helpers import trace_times
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import infer_reference_point, Hypervolume
from src.bayesopt.acqf import thompson_sampling_multivariate, scalarize
from transformers.tokenization_utils_base import BatchEncoding

device = helpers.check_device()
print(f"Using device: {device}")


class MultiObjMATExpRunner:
    """
    Multi-objective experiment runner for material discovery benchmarks.
    """

    def __init__(self, mat_bench, seed, normalize_y=False, list_init_points=None):
        self.mat_bench = mat_bench  # not deep copied
        self.seed = seed
        self.list_init_points = list_init_points
        self.normalize_y = normalize_y

    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for multi-objective BO search
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''
        if self.list_init_points is not None:
            init_points = self.list_init_points[:n_samples]
        else:
            init_points = []
            dataset = self.mat_bench.dataset

            # For multi-objective, we need to handle clustering uniformly
            init_cluster_cnt = {cluster: 0 for cluster in range(self.mat_bench.n_clusters)}

            while len(init_points) < n_samples:
                idx = np.random.randint(len(dataset))
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
        Evaluate a single point on the multi-objective benchmark
        '''
        label, _ = self.mat_bench.complete_call(candidate)
        return candidate, label


def save_results_multiobj_llmat(args, mat_bench, timing_train, timing_preds, trace_hypervolume, trace_scalarized_best_y, trace_best_y, trace_timing):
    """Save multi-objective LLMAT results."""
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/multiobj/{prefix}/{args.algorithm}"
    if args.algorithm == "multiobj_llmat":
        clustering = mat_bench.dataset_name.split("/")[-2]
        path = f"{path}/{clustering}"
    if not os.path.exists(path):
        os.makedirs(path)

    suffix = f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}_{args.tree_depth}_{args.seed}"

    np.save(f"{path}/timing_train_{suffix}.npy", timing_train)
    np.save(f"{path}/timing_preds_{suffix}.npy", timing_preds)
    np.save(f"{path}/trace_hypervolume_{suffix}.npy", trace_hypervolume)
    np.save(f"{path}/trace_scalarized_best_y_{suffix}.npy", trace_scalarized_best_y)
    np.save(f"{path}/trace_best_y_{suffix}.npy", trace_best_y)
    np.save(f"{path}/trace_timing_{suffix}.npy", trace_timing)


def select_from_leaf_to_root(node_acqfs_all, node_ids):
    """
    Select the candidates from leaf to root (same as original LLMAT)
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
    # print(node_acqfs_all)
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

        # Ensure all tensors have at least 1 dimension before concatenation
        acq_vals = [v.unsqueeze(0) if v.ndim == 0 else v for v in acq_vals]
        y_preds = [v.unsqueeze(0) if v.ndim == 0 else v for v in y_preds]
        y_reg = [v.unsqueeze(0) if v.ndim == 0 else v for v in y_reg]
        y_class = [v.unsqueeze(0) if v.ndim == 0 else v for v in y_class]

        acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
        y_preds = torch.cat(y_preds, dim=0).cpu()
        y_reg = torch.cat(y_reg, dim=0)
        y_class = torch.cat(y_class, dim=0)

        # print(y_class)
        # print(acq_vals)
        # print(y_preds)
        if y_reg.ndim == 0:
            print("Tensor is 0D (scalar)")
            y_reg = y_reg.unsqueeze(0)
        if acq_vals.ndim == 0:
            print("Tensor is 0D (scalar)")
            acq_vals = acq_vals.unsqueeze(0)
        if y_preds.ndim == 0:
            print("Tensor is 0D (scalar)")
            y_preds = y_preds.unsqueeze(0)
        assert len(y_reg) == len(acq_vals)
        assert len(candidates_idx) > 0
        return candidates_idx, y_reg, y_preds, y_class, acq_vals


def run_multiobj_llmat(args, mat_runner, wandb=None):
    """
    Run multi-objective LLMAT algorithm following the original LLMAT structure.
    """
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    mat_bench = mat_runner.mat_bench

    # Get multi-objective configuration
    if not hasattr(mat_bench, 'obj_cols') or mat_bench.obj_cols is None:
        raise ValueError("Multi-objective optimization requires obj_cols to be specified in mat_bench")

    OBJ_COLS = mat_bench.target_cols_transformed
    n_objectives = len(OBJ_COLS)
    print(f"Number of objectives: {n_objectives}")
    print(f"Objective columns: {OBJ_COLS}")

    # Get maximization flags for each objective
    if hasattr(mat_bench, 'maximizations'):
        MAXIMIZATIONS = mat_bench.maximizations
    else:
        MAXIMIZATIONS = [True] * n_objectives
        print(f"Maximizations: {MAXIMIZATIONS}")

    # Get ground truth information for hypervolume calculation
    all_targets_tensor = torch.tensor(mat_runner.mat_bench.dataset[OBJ_COLS].values)
    Y_pareto = all_targets_tensor[is_non_dominated(all_targets_tensor)]
    ref_point = infer_reference_point(Y_pareto)
    hv = Hypervolume(ref_point)
    max_hypervolume = hv.compute(Y_pareto)

    print(f"Reference point: {ref_point}")
    print(f"Max hypervolume: {max_hypervolume:.3f}")

    # Initialize LLMAT model
    lora_cfg = args.finetuning_args["lora"]
    if args.finetuning:
        head_cfg = args.finetuning_args["head"]
    else:
        head_cfg = args.fix_args["head"]

    print("========== get peft model =========")
    peft_model, tokenizer = get_model(mat_bench.foundation_model, True, lora_cfg)
    feature_dim = peft_model.feature_dim
    print("feature dim:", feature_dim)

    data_processor = mat_bench.get_data_processor(tokenizer)
    print("========== get data processor =========")

    # Generate initial training data
    dataset_train = mat_runner.generate_initialization(args.n_init_data)

    # Get multi-objective targets
    targets = dataset_train[OBJ_COLS].values

    # Convert targets to tensor format expected by LLMAT
    # For multi-objective, we use scalarized targets for the tree construction
    scalarized_targets = scalarize(torch.tensor(targets))
    dataset_train = dataset_train.copy()
    dataset_train['targets_transformed'] = scalarized_targets.numpy()  # Changed from 'targets' to 'targets_transformed'

    # Initialize data structures for LLMAT
    APPEND_EOS = mat_bench.foundation_model != "molformer" \
                       and ("t5" not in mat_bench.foundation_model)

    print("========== init Multi-Objective Alpha model =========")
    alpha_model = MultiLLMAT(
        peft_model,
        dataset_train,
        data_processor,
        head_cfg,
        lora_cfg,
        device=device,
        append_eos=APPEND_EOS,
        gamma=args.gamma,  # quantile
        acf_type=args.acqf,
        lmbda=args.lmbda,  # ucb weight
        eta=args.eta,  # meta learning rate
        tree_depth=args.tree_depth,
        finetuning=args.finetuning,
        reinit=getattr(args, 'reinit', False),
        threshold=args.threshold,
        alpha=getattr(args, 'alpha', 1.0),
        feature_dim=feature_dim,
    )
    print("--------finish init----------")
    alpha_model.restore_finetuning()  # restore the initial LoRA state
    alpha_model.train_model()
    print("============finish training ALPHA model==============")

    # Initialize tracking arrays
    trace_hypervolume = np.zeros(args.exp_len + 1)
    trace_scalarized_best_y = np.zeros(args.exp_len + 1)
    trace_best_y = np.zeros((args.exp_len + 1, n_objectives))
    trace_timing = np.zeros(args.exp_len + 1)
    timing_train = []
    timing_preds = []

    # Get initial Pareto front
    observed_targets = torch.tensor(targets)
    pareto_mask = is_non_dominated(observed_targets)
    Y_pareto_current = observed_targets[pareto_mask]
    current_hypervolume = hv.compute(Y_pareto_current) if len(Y_pareto_current) > 0 else 0.0
    scalarized_best = scalarize(Y_pareto_current.max(dim=0)[0] if len(Y_pareto_current) > 0 else torch.zeros(n_objectives))

    trace_hypervolume[0] = current_hypervolume
    trace_scalarized_best_y[0] = scalarized_best.item()
    trace_best_y[0] = Y_pareto_current.max(dim=0)[0].numpy() if len(Y_pareto_current) > 0 else np.zeros(n_objectives)

    print(f"Initial hypervolume: {current_hypervolume:.3f}")
    print(f"Initial scalarized best: {scalarized_best.item():.3f}")

    # Main BO loop - following the exact structure of original LLMAT
    pbar_bo_iters = tqdm.tqdm(range(1, args.exp_len + 1), desc="Multi-objective LLMAT")
    for t in pbar_bo_iters:
        start = time.time()

        # Get cluster column
        cluster_col = mat_bench._get_cluster_col()

        # Select clusters to estimate using ANOVA (same as original)
        clusters = alpha_model.get_clusters(pval=args.p_val, n_clusters=mat_bench.n_clusters)
        dataset = mat_bench.dataset
        print("========== chosen cluster to estimate =======")
        print(clusters)
        dataset = dataset[dataset[cluster_col].isin(clusters)]

        # Select most promising path from root to leaf
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

        # Get data loader
        if args.finetuning:
            print("finetuning batch size:", args.finetuning_args["batch_size"])
            dataloader = data_processor.get_dataloader(
                dataset,
                batch_size=args.finetuning_args["batch_size"],
                shuffle=False,
                append_eos=APPEND_EOS,
            )
        else:
            # Add targets_transformed column for FTDataset compatibility
            if 'targets_transformed' not in dataset.columns:
                # Create scalarized targets for candidate evaluation
                candidate_targets = dataset[OBJ_COLS].values
                candidate_scalarized = scalarize(torch.tensor(candidate_targets))
                dataset = dataset.copy()
                dataset['targets_transformed'] = candidate_scalarized.numpy()

            ft_dataset = MultiObjFTDataset(dataset, mat_bench._get_cluster_col())
            dataloader = DataLoader(
                ft_dataset,
                batch_size=args.fix_args["batch_size"],
                shuffle=False,
            )

        print("data loader batch size", dataloader.batch_size)
        sub_pbar = tqdm.tqdm(
            dataloader,
            position=1,
            colour="blue",
            desc="[Prediction over dataset]",
            leave=False,
        )

        start_pred, end_pred = trace_times(start=None, end=None, device=device)
        for data in sub_pbar:
            print("======== BO batch size ===========")
            print("BO batch size:", len(data["targets"]))

            # Evaluate acquisition functions along the path
            node_acqfs = alpha_model.cal_acqfs(data, path)

            for node_id in node_ids:
                if node_id not in node_acqfs:
                    print(f"node {node_id} has no candidates in this batch")
                    continue
                node_acqfs_all[node_id][0] = node_acqfs[node_id][0]  # tau
                node_acqfs_all[node_id][1].append(node_acqfs[node_id][1])  # ratio
                node_acqfs_all[node_id][2] += node_acqfs[node_id][2]  # inds
                node_acqfs_all[node_id][3] += node_acqfs[node_id][3].cpu()  # acq_val
                node_acqfs_all[node_id][4] += node_acqfs[node_id][4].cpu()  # targets
                node_acqfs_all[node_id][5] += node_acqfs[node_id][5].cpu()  # preds
                # Handle case where node.tau might be None
                tau_value = node_acqfs[node_id][0] if node_acqfs[node_id][0] is not None else 0.5
                labels = torch.greater_equal(node_acqfs[node_id][4], tau_value)
                node_acqfs_all[node_id][6] += labels.cpu()  # labels
                node_acqfs_all[node_id][7] += node_acqfs[node_id][6].cpu()  # candidate features in each node

            if node_id in node_acqfs:
                print(">>>>>>>>>>leaf candidates ratio>>>>>>>>>>>>", node_acqfs[node_id][1])
                print("-----------***********------------")
                if node_acqfs[node_id][2] == 0:
                    print("no promising leaf candidates in this batch")

        # Record prediction timing
        timing_preds.append(trace_times(start_pred, end_pred, device))

        # Select candidate from leaf to root
        candidates_idx, y_reg, y_preds, y_class, acq_vals = select_from_leaf_to_root(node_acqfs_all, node_ids)
        print("find optimal x for acq func")

        _, idx = acq_vals.topk(k=min(10, len(acq_vals)))
        for y_r, y_c, y_p, acqf in zip(y_reg[idx], y_class[idx], y_preds[idx], acq_vals[idx]):
            print(f"True: {y_r.item():.3f}, Class:{y_c.item():3f}, Pred: {y_p.item():.3f}, Acqf: {acqf.item():.3f}")

        print("idx device:", idx.device)
        print("find optimal x for true value")
        _, idx2 = y_reg.topk(k=min(10, len(acq_vals)))
        idx2 = idx2.to("cpu")
        print("idx device:", idx2.device)
        for y_r, y_c, y_p, acqf in zip(y_reg[idx2], y_class[idx2], y_preds[idx2], acq_vals[idx2]):
            print(f"True: {y_r.item():.3f}, Class:{y_c.item():3f}, Pred: {y_p.item():.3f}, Acqf: {acqf.item():.3f}")

        print("========path leaf candidates indexes=======")
        print(candidates_idx[:10])
        print("========path leaf candidates targets=======")
        print(y_reg[:10])

        # Select the best candidate based on acquisition function
        idx_best = acq_vals.argmax().item()
        index = candidates_idx[idx_best]

        # Get the new data point
        new_data = helpers.pop_df(mat_bench.dataset, mat_bench.dataset[mat_bench.dataset['Entry Number'] == index].index[0])

        # Evaluate the new point - ensure numeric conversion
        obj_values = new_data[OBJ_COLS].values
        # Convert to numeric, handling any string/object types
        if obj_values.dtype == 'object':
            # Try to convert each column to numeric
            numeric_values = []
            for i, col in enumerate(OBJ_COLS):
                try:
                    val = pd.to_numeric(obj_values[i], errors='coerce')
                    numeric_values.append(float(val))
                except:
                    numeric_values.append(0.0)  # fallback
            new_targets = torch.tensor(numeric_values, dtype=torch.float32)
        else:
            new_targets = torch.tensor(obj_values.astype(float))

        # Update observed data
        observed_targets = torch.cat([observed_targets, new_targets.unsqueeze(0)], dim=0)

        # Update Pareto front
        pareto_mask = is_non_dominated(observed_targets)
        Y_pareto_current = observed_targets[pareto_mask]
        current_hypervolume = hv.compute(Y_pareto_current)

        #  scalarization
        scalarized_values = torch.tensor([scalarize(y) for y in Y_pareto_current])
        scalarized_best = scalarized_values.max()
        best_index = scalarized_values.argmax()
        current_best = Y_pareto_current[best_index]  # exist

        # Record results
        trace_hypervolume[t] = current_hypervolume
        trace_scalarized_best_y[t] = scalarized_best.item()
        trace_best_y[t] = current_best.numpy()

        # Update model with new data
        new_data_scalarized = scalarize(new_targets).item()
        new_data_dict = new_data.copy()
        new_data_dict['targets_transformed'] = new_data_scalarized  # Changed from 'targets' to 'targets_transformed'

        start_train, end_train = trace_times(start=None, end=None, device=device)
        alpha_model.condition_on_observations(new_data_dict)
        timing_train.append(trace_times(start_train, end_train, device))

        end = time.time()
        trace_timing[t] = end - start

        # Update progress bar
        pbar_bo_iters.set_description(f"[Hypervolume = {current_hypervolume:.3f}, "
                                      f"Scalarized Best = {scalarized_best.item():.3f}, "
                                      f"Max Hypervolume = {max_hypervolume:.3f}]")

        # Log to wandb if provided
        if wandb:
            wandb.log({"trace_hypervolume": current_hypervolume}, step=t)
            wandb.log({"trace_scalarized_best_y": scalarized_best.item()}, step=t)
            for i, obj_col in enumerate(OBJ_COLS):
                wandb.log({f"trace_best_{obj_col}": current_best[i].item()}, step=t)
            wandb.log({"trace_timing": trace_timing[t]}, step=t)
            wandb.log({"trace_timing_train": timing_train[-1]}, step=t)
            wandb.log({"trace_timing_pred": timing_preds[-1]}, step=t)
            wandb.log({"trace_acqvals": acq_vals[idx_best].item()}, step=t)

    # Save results
    save_results_multiobj_llmat(args, mat_bench, timing_train, timing_preds, trace_hypervolume, trace_scalarized_best_y, trace_best_y, trace_timing)

    print("====== Multi-objective LLMAT completed ======")
    print(f"Final hypervolume: {trace_hypervolume[-1]:.3f}")
    print(f"Final scalarized best: {trace_scalarized_best_y[-1]:.3f}")

    return trace_hypervolume, trace_scalarized_best_y


def multiobj_llmat_bo(args, mat_bench, wandb=None):
    """
    Entry point for multi-objective LLMAT algorithm.
    Args:
        args: command line arguments
        mat_bench: the benchmark to run on
        wandb: wandb logger (optional)
    Returns:
        trace_hypervolume: hypervolume progression
        trace_scalarized_best_y: scalarized best y progression
    """
    print(">>>>> Enter Multi-Objective LLMAT >>>>>")

    if args.benchmark == 'multiobj_mat':
        mat_runner = MultiObjMATExpRunner(mat_bench, args.seed, args.normalize_y)
        trace_hypervolume, trace_scalarized_best_y = run_multiobj_llmat(args, mat_runner, wandb=wandb)
        return trace_hypervolume, trace_scalarized_best_y

    return None, None
