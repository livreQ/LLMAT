#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Sat Oct  5 02:05:07 2024
# File Name: llmat_bo.py
# Description:
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

from src.utils import helpers
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import StandardScaler
from src.algos.llmat.llmat import LLMAT
from src.algos.llmat.llmat import FTDataset
from torch.utils.data import DataLoader
from src.algos.llmat.llms.base_llms import get_model
from src.utils import helpers
from src.utils.helpers import trace_times
import math

device = helpers.check_device()
print(f"Using device: {device}")


class MATExpRunner:

    def __init__(self, mat_bench, seed, normalize_y, list_init_points=None):
        self.mat_bench = mat_bench  # not deep copied
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


def get_ablation_suffix(args):
    lora_cfg = args.finetuning_args["lora"]
    p_val = args.p_val
    if p_val > 0:
        if args.finetuning:
            lora_lr = lora_cfg["lr"]
            lora_reg = lora_cfg["reg"]
            suffix = f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}_{args.tree_depth}_{lora_lr}_{lora_reg}_{p_val}_{args.seed}"
        else:
            suffix = f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}_{args.tree_depth}_{p_val}_{args.seed}"
    else:
        if args.finetuning:
            lora_lr = lora_cfg["lr"]
            lora_reg = lora_cfg["reg"]
            suffix = f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}_{args.tree_depth}_{lora_lr}_{lora_reg}_{args.seed}"
        else:
            suffix = f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}_{args.tree_depth}_{args.seed}"
    sel = getattr(args, "selection_type", "uct")
    if sel == "greedy":
        suffix += "_greedy"
    return suffix

def save_results(args, mat_bench, timing_train,\
                            timing_preds, trace_acqvals, trace_y_his,  trace_best_y, trace_timing):

    print(mat_bench.dataset_name)
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/{args.algorithm}"
    if args.algorithm == "llmat":
        clustering = mat_bench.dataset_name.split("/")[-2]
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


def run_bo(args, mat_bench, wandb=None):
    """
        mcts partition selection and LFBO with finetuning llms
    """
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    lora_cfg = args.finetuning_args["lora"]
    if args.finetuning:
        head_cfg = args.finetuning_args["head"]
    else:
        head_cfg = args.fix_args["head"]

    # get peft model
    print("========== get peft model =========")
    peft_model, tokenizer = get_model(mat_bench.foundation_model, True, lora_cfg)  # get token or not
    if not args.finetuning and mat_bench.foundation_model in ("fingerprints", "gin", "schnet"):
        feature_dim = mat_bench.dataset['features'][0].shape[-1]
    else:
        feature_dim = peft_model.feature_dim
    print("feature dim:", feature_dim)
    # print("feature dim:", peft_model.feature_dim)
    data_processor = mat_bench.get_data_processor(tokenizer)
    print("========== get data processor =========")
    mat_runner = MATExpRunner(mat_bench, seed, args.normalize_y)
    # n_init_data samples for first training
    dataset_train = mat_runner.generate_initialization(args.n_init_data)  # pd dataframe
    target_col_transformed = mat_bench.target_col_transformed
    target_col = mat_bench.target_col
    print("==========finish benchmark runner init=========")
    print("ground truth max:", mat_runner.ground_truth_max_transformed)
    print("ground truth optimal:", mat_bench.ground_truth_opt)
    if not args.normalize_y:
        if mat_bench.maximization:
            assert mat_runner.ground_truth_max_transformed == mat_bench.ground_truth_opt
        else:
            assert mat_runner.ground_truth_max_transformed == -mat_bench.ground_truth_opt

    # Train
    APPEND_EOS = mat_bench.foundation_model != "molformer" \
                       and ("t5" not in mat_bench.foundation_model)
    # print(APPEND_EOS)
    # print(dataset_train)
    print("========== init Alpha model =========")
    alpha_model = LLMAT(
        peft_model,
        dataset_train,
        data_processor,
        head_cfg,
        lora_cfg,
        device,
        append_eos=APPEND_EOS,
        gamma=args.gamma,  # quantile
        acf_type=args.acqf,
        lmbda=args.lmbda,  # ucb weight
        eta=args.eta,  # meta learning rate
        tree_depth=args.tree_depth,
        finetuning=args.finetuning,
        reinit=args.reinit,
        threshold=args.threshold,
        alpha=args.alpha,
        feature_dim=feature_dim,
        selection_type=getattr(args, "selection_type", "uct"),
    )
    print("--------finish init----------")
    alpha_model.restore_finetuning()  # restore the initial LoRA state
    alpha_model.train_model()
    print("============finish training ALPHA model==============")

    id_max_observed = dataset_train[target_col_transformed].idxmax()
    max_y_observed = dataset_train.loc[id_max_observed][target_col_transformed]
    opt_y_observed = dataset_train.loc[id_max_observed][target_col]
    initial_opt_y = opt_y_observed
    best_y = max_y_observed
    #----------------  trace --------------------
    true_best = mat_bench.ground_truth_opt
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
        true_best = mat_bench.ground_truth_opt
        cluster_col = mat_bench._get_cluster_col()
        # y_preds, y_uncerts, y_reg, y_class = [], [], [], []
        # acq_vals = []
        # candidates_idx = []

        #-------------------- estimate acf ---------------------------
        # We first select the clusters to be estimated
        # by using AVONA
        clusters = alpha_model.get_clusters(pval=args.p_val, n_clusters=mat_bench.n_clusters)
        dataset = mat_bench.dataset
        print("========== chosen cluster to eatimate =======")
        print(clusters)
        # dataset = dataset[dataset[cluster_col] == cluster]
        dataset = dataset[dataset[cluster_col].isin(clusters)]
        print("=========unobserved dataset ==========")
        print(dataset)
        # At each BO round we select a most promising Path
        # and then estimate the acquisition function of the unobserved data
        # that predicted to fall into nodes along this path
        leaf, path = alpha_model.select()
        node_acqfs_all = {}  # for all the batches
        # node_cands_all = {}  # for all the batches
        node_ids = []
        # print(path)
        for node, choice in path:
            #{node id: tau, ratio, inds, acq_val, targets, preds, labels, features}
            node_acqfs_all[node.id] = [0, [], [], [], [], [], [], []]
            # node_cands_all[node.id] = []
            node_ids.append(node.id)
            if choice == 0:
                print("choose left child")
            elif choice == 1:
                print("choose right child")
            else:
                print("unimplemented choice")
        ## get data loader
        if args.finetuning:
            print("finetuning batch size:", args.finetuning_args["batch_size"])
            dataloader = data_processor.get_dataloader(
                dataset,
                batch_size=args.finetuning_args["batch_size"],
                shuffle=False,
                append_eos=APPEND_EOS,
            )  # redundant colums removed
        else:
            ft_dataset = FTDataset(dataset, mat_bench._get_cluster_col())
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
            # print(data)
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
                node_acqfs_all[node_id][3] += node_acqfs[node_id][3].cpu()  # acq_val
                node_acqfs_all[node_id][4] += node_acqfs[node_id][4].cpu()  # targets
                node_acqfs_all[node_id][5] += node_acqfs[node_id][5].cpu()  # preds
                labels = torch.greater_equal(node_acqfs[node_id][4], node_acqfs[node_id][0])
                node_acqfs_all[node_id][6] += labels.cpu()  # labels
                node_acqfs_all[node_id][7] += node_acqfs[node_id][6].cpu()  # candidate features in each node
            if node_id in node_acqfs:
                print(">>>>>>>>>>leaf candidates ratio>>>>>>>>>>>>", node_acqfs[node_id][1])
                print("-----------***********------------")
                if node_acqfs[node_id][2] == 0:
                    print("no promising leaf candidates in this batch")

        # Record the end time for predictions
        timing_preds.append(trace_times(start_pred, end_pred, device))

        candidates_idx, y_reg, y_preds, y_class, acq_vals = select_from_leaf_to_root(node_acqfs_all, node_ids)
        print("find optimal x for acq func")

        _, idx = acq_vals.topk(k=min(10, len(acq_vals)))
        for y_r, y_c, y_p, acqf in zip(y_reg[idx], y_class[idx], y_preds[idx], acq_vals[idx]):
            print(f"True: {y_r.item():.3f}, Class:{y_c.item():3f}, Pred: {y_p.item():.3f}, Acqf: {acqf.item():.3f}")
        # input()
        print("idx device:", idx.device)
        print("find optimal x for true value")
        _, idx2 = y_reg.topk(k=min(10, len(acq_vals)))
        idx2 = idx2.to("cpu")
        print("idx device:", idx2.device)
        for y_r, y_c, y_p, acqf in zip(y_reg[idx2], y_class[idx2], y_preds[idx2], acq_vals[idx2]):
            print(f"True: {y_r.item():.3f}, Class:{y_c.item():3f}, Pred: {y_p.item():.3f}, Acqf: {acqf.item():.3f}")

        print("========path leaf candidates indexes=======")
        print("number of selected candidates: " + str(len(candidates_idx)))
        print(idx)
        print(dataset)
        print(dataset.index)

        # Pick a molecule (a row in the current dataset) that maximizes the acquisition
        # if args.finetuning:
        #     candidates = dataset[dataset.index.isin(candidates_idx)]  #dataset[dataset.index == candidates_idx]
        # else:
        candidates = dataset[dataset["Entry Number"].isin(candidates_idx)]
        print("========== path leaf candidates=======")
        print(candidates)
        assert len(candidates) == len(candidates_idx), "candidates and candidates_idx should have the same length"

        idx_best = torch.argmax(acq_vals).item()
        print(">>>>>>best id in candidates", idx_best)
        x_can = candidates.iloc[idx_best]
        entry_can = candidates_idx[idx_best]
        # x_can_prime = candidates[candidates["Entry Number"] == entry_can]
        # print(candidates[candidates["Entry Number"] == entry_can])
        assert x_can["Entry Number"] == entry_can, "x_can should have the same Entry Number as entry_can"
        print(">>>>>>selected candidate>>>>", x_can["SMILES"], x_can[target_col_transformed], x_can[target_col])
        # print(candidates)
        # index = dataset[dataset["SMILES"] == x_can["SMILES"]].index
        # if args.finetuning:
        #     index = candidates_idx[idx_best]
        # else:
        index = dataset[dataset['Entry Number'] == candidates_idx[idx_best]].index[0]
        new_data = helpers.pop_df(mat_bench.dataset, index)  #x,y

        # delete new observed data in candidate dataset
        # mat_bench.dataset = dataset
        # print("current best:", best_y)
        # print("new data", new_data)
        # Update the current best y
        if new_data[target_col_transformed] > best_y:
            best_y = new_data[target_col_transformed]
            opt_y_observed = new_data[target_col]
            #print(best_y_ori)

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
        if mat_bench.maximization:
            regret = mat_bench.ground_truth_opt - np.mean(trace_y_his[1:t + 2])
        else:
            regret = np.mean(trace_y_his[1:t + 2]) - mat_bench.ground_truth_opt
        wandb.log({"trace_regret": regret}, step=t)
        GAP = np.nan_to_num((opt_y_observed - initial_opt_y) / (mat_bench.ground_truth_opt - initial_opt_y), nan=1.0)
        wandb.log({"trace_gap": GAP}, step=t)
        wandb.log({"trace_timing": timing}, step=t)
        wandb.log({"trace_timing_train": timing_train[-1]}, step=t)
        wandb.log({"trace_timing_pred": timing_preds[-1]}, step=t)
        wandb.log({"trace_acqvals": acq_vals[idx_best].item()}, step=t)

        prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
        path = f"results/{prefix}/{args.algorithm}"
        if args.algorithm == "llmat":
            clustering = mat_bench.dataset_name.split("/")[-2]
            path = f"{path}/{clustering}"
        if not os.path.exists(path):
            os.makedirs(path)
        suffix = get_ablation_suffix(args)

        # if t == 0:
        #     np.save(f"{path}/node_acqfs_all_run_{t}_{suffix}.npy", node_acqfs_all)
        if args.early_stopping:
            # Early stopping if we already got the max
            if best_y >= mat_runner.ground_truth_max_transformed:
                for j in range(t + 1, args.exp_len + 1):
                    wandb.log({"trace_best_y": trace_best_y[t + 1]}, step=j)
                    wandb.log({"trace_y_his": trace_y_his[t + 1]}, step=j)
                    wandb.log({"trace_timing": timing}, step=j)
                    wandb.log({"trace_timing_train": timing_train[-1]}, step=j)
                    wandb.log({"trace_timing_pred": timing_preds[-1]}, step=j)
                    wandb.log({"trace_acqvals": 0}, step=j)
                    if mat_bench.maximization:
                        regret = mat_bench.ground_truth_opt - np.mean(trace_y_his[1:j + 2])
                    else:
                        regret = np.mean(trace_y_his[1:j + 2]) - mat_bench.ground_truth_opt
                    # np.nan_to_num((y_t - y_0) / (y_best - y_0), nan=1.0)
                    GAP = np.nan_to_num((opt_y_observed - initial_opt_y) / (mat_bench.ground_truth_opt - initial_opt_y), nan=1.0)
                    wandb.log({"trace_gap": GAP}, step=j)
                    wandb.log({"trace_regret": regret}, step=j)
                    # np.save(f"{path}/node_acqfs_all_run_{t}_{suffix}.npy", node_acqfs_all)
                break

    save_results(args, mat_runner.mat_bench, timing_train, timing_preds, trace_acqvals, trace_y_his, trace_best_y, trace_timing)

    if alpha_model.node_train_logs:
        prefix = "/".join(mat_runner.mat_bench.dataset_name.split("/")[:-2])
        log_path = f"results/{prefix}/{args.algorithm}"
        if args.algorithm == "llmat":
            clustering = mat_runner.mat_bench.dataset_name.split("/")[-2]
            log_path = f"{log_path}/{clustering}"
        suffix = get_ablation_suffix(args)
        np.save(f"{log_path}/node_train_logs_{suffix}.npy", alpha_model.node_train_logs)

    print("====== trace best y======")
    print(trace_best_y)

    return trace_best_y, None


def llmat_bo(args, mat_bench, wandb=None):
    """
    Run the LLMAT algorithm on the given benchmark.
    Args:
        args: command line arguments
        mat_bench: the benchmark to run on
        wandb: wandb logger (optional)
    Returns:
        trace_best_y: the best y value found during the optimization
        all_metrics: all metrics collected during the optimization
    """

    print(">>>>> Enter BO >>>>>")
    if args.benchmark == 'mat':
        trace_best_y, all_metrics = run_bo(args, mat_bench, wandb=wandb)

    return trace_best_y, all_metrics
