#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
BO runner for the Decoupled Partitioning + Independent LFBO ablation.

Replaces LLMAT's coupled classifier with:
  1. Partition classifier (BCE on top-γ) for target-aware splitting
  2. Separate LFBO acquisition classifier per node for scoring
  3. EMA meta-learned initialization preserved across tree rebuilds

Results are saved under results/{prefix}/decoupled_mcts_lfbo/{clustering}/
with suffix  {n_init}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}_decoupled
"""

import os
import math
import time

import numpy as np
import pandas as pd
import torch
import tqdm

from src.utils import helpers
from src.utils.helpers import trace_times
from src.algos.llmat.decoupled_mcts_lfbo import DecoupledPartitionLFBO
from src.algos.llmat.llmat_bo import MATExpRunner


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _get_suffix(args) -> str:
    p_val = getattr(args, "p_val", 0)
    if p_val and p_val > 0:
        suffix = (f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}"
                  f"_{args.tree_depth}_{p_val}_{args.seed}")
    else:
        suffix = (f"{args.n_init_data}_{args.acqf}_{args.gamma}_{args.eta}"
                  f"_{args.tree_depth}_{args.seed}")
    if getattr(args, "selection_type", "uct") == "greedy":
        suffix += "_greedy"
    suffix += "_decoupled"
    return suffix


def _save(args, mat_bench, timing_train, timing_preds,
          trace_acqvals, trace_y_his, trace_best_y, trace_timing):
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"results/{prefix}/decoupled_mcts_lfbo/{clustering}"
    os.makedirs(path, exist_ok=True)
    sfx = _get_suffix(args)
    np.save(f"{path}/timing_train_{sfx}.npy", timing_train)
    np.save(f"{path}/timing_preds_{sfx}.npy", timing_preds)
    np.save(f"{path}/trace_acqvals_{sfx}.npy", trace_acqvals)
    np.save(f"{path}/trace_best_y_{sfx}.npy", trace_best_y)
    np.save(f"{path}/trace_y_his_{sfx}.npy", trace_y_his)
    np.save(f"{path}/trace_timing_{sfx}.npy", trace_timing)


# --------------------------------------------------------------------------
# main BO loop
# --------------------------------------------------------------------------

def run_bo_decoupled(args, mat_bench, wandb=None):
    """
    Decoupled Partition LFBO — BO main loop.

    At each round:
      1. Build classifier partition tree; fit separate LFBO acquisition heads.
      2. Select path (leaf, path) by UCT or greedy.
      3. Route candidates via partition classifiers; score survivors at leaf.
         If no candidates survive routing, fall back to root-level scoring.
      4. Query argmax candidate, update model, repeat.
    """
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = helpers.check_device()

    sample_feat = mat_bench.dataset["features"].iloc[0]
    feature_dim = (sample_feat.shape[-1]
                   if hasattr(sample_feat, "shape")
                   else len(sample_feat))

    if args.finetuning:
        head_cfg = args.finetuning_args["head"]
    else:
        head_cfg = args.fix_args["head"]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    mat_runner = MATExpRunner(mat_bench, seed, args.normalize_y)
    dataset_train = mat_runner.generate_initialization(args.n_init_data)
    target_col_t = mat_bench.target_col_transformed
    target_col = mat_bench.target_col

    def _make_train_df(df: pd.DataFrame) -> pd.DataFrame:
        # NOTE: target_col is literally "targets" in the non-finetuning
        # (fixed-feature) setting, same as the hardcoded key below. Do NOT
        # add a `target_col: ...` entry here — in a dict literal that would
        # silently overwrite the "targets": df[target_col_t] entry with the
        # untransformed (wrong-direction) target and invert the objective.
        return pd.DataFrame({
            "features": df["features"].tolist(),
            "SMILES": df["SMILES"].tolist(),
            "targets": df[target_col_t].tolist(),
            "Entry Number": df["Entry Number"].tolist(),
        })

    train_df = _make_train_df(dataset_train)

    print("========== init DecoupledPartitionLFBO ==========")
    model = DecoupledPartitionLFBO(
        feature_dim=feature_dim,
        tree_depth=args.tree_depth,
        leaf_sample_size=head_cfg.get("leaf_sample_size", 2),
        gamma=args.gamma,
        acf_type=args.acqf,
        lmbda=args.lmbda,
        device=device,
        head_cfg=head_cfg,
        selection_type=getattr(args, "selection_type", "uct"),
    )
    model.train(train_df)

    # ------------------------------------------------------------------
    # Book-keeping
    # ------------------------------------------------------------------
    id_max = dataset_train[target_col_t].idxmax()
    best_y = dataset_train.loc[id_max, target_col_t]
    opt_y_observed = dataset_train.loc[id_max, target_col]
    initial_opt_y = opt_y_observed

    true_best = mat_bench.ground_truth_opt
    trace_best_y  = [true_best] * (args.exp_len + 1)
    trace_y_his   = [true_best] * (args.exp_len + 1)
    trace_timing  = [0.0] * (args.exp_len + 1)
    trace_acqvals = [-math.inf] * (args.exp_len + 1)
    timing_train  = []
    timing_preds  = []

    pbar = tqdm.trange(args.exp_len, position=0, colour="green", leave=True)
    pbar.set_description(f"[Best f(x)={opt_y_observed:.3f}]")

    for t in pbar:
        start, end = trace_times(None, None, device)

        # ------------------------------------------------------------------
        # 1. Select path via UCT / greedy
        # ------------------------------------------------------------------
        leaf, path = model.select()
        print(f"[t={t}] selected leaf={leaf.id}, path depth={len(path)}")

        # ------------------------------------------------------------------
        # 2. Score all unobserved candidates; route them down the path
        # ------------------------------------------------------------------
        start_pred, end_pred = trace_times(None, None, device)

        dataset = mat_bench.dataset
        if len(dataset) == 0:
            print("Dataset exhausted — stopping early.")
            break

        cand_features = np.stack(dataset["features"].tolist()).astype(np.float32)

        alive, acq_vals = model.get_acq_vals(cand_features, path, leaf)

        timing_preds.append(trace_times(start_pred, end_pred, device))

        # ------------------------------------------------------------------
        # 3. Select best surviving candidate
        # ------------------------------------------------------------------
        alive_indices = np.where(alive)[0]
        best_local = int(torch.argmax(acq_vals[alive_indices]).item())
        best_global = alive_indices[best_local]

        x_can = dataset.iloc[best_global]
        index = dataset[dataset["Entry Number"] == x_can["Entry Number"]].index[0]
        new_data = helpers.pop_df(mat_bench.dataset, index)

        if new_data[target_col_t] > best_y:
            best_y = new_data[target_col_t]
            opt_y_observed = new_data[target_col]

        # ------------------------------------------------------------------
        # 4. Update model
        # ------------------------------------------------------------------
        start_train, end_train = trace_times(None, None, device)

        new_row = pd.Series({
            "features": new_data["features"],
            "SMILES": new_data["SMILES"],
            "targets": new_data[target_col_t],
            "Entry Number": new_data["Entry Number"],
        })
        model.condition_on_observations(new_row)

        timing_train.append(trace_times(start_train, end_train, device))

        # ------------------------------------------------------------------
        # 5. Log
        # ------------------------------------------------------------------
        timing = trace_times(start, end, device)
        trace_best_y[t + 1]  = opt_y_observed
        trace_y_his[t + 1]   = new_data[target_col]
        trace_timing[t + 1]  = timing
        trace_acqvals[t + 1] = float(acq_vals[best_global].item())

        if wandb is not None:
            wandb.log({"trace_best_y": opt_y_observed}, step=t)
            wandb.log({"trace_y_his": new_data[target_col]}, step=t)
            wandb.log({"trace_timing": timing}, step=t)
            wandb.log({"trace_timing_train": timing_train[-1]}, step=t)
            wandb.log({"trace_timing_pred": timing_preds[-1]}, step=t)
            wandb.log({"trace_acqvals": trace_acqvals[t + 1]}, step=t)
            if mat_bench.maximization:
                regret = mat_bench.ground_truth_opt - np.mean(trace_y_his[1:t + 2])
            else:
                regret = np.mean(trace_y_his[1:t + 2]) - mat_bench.ground_truth_opt
            wandb.log({"trace_regret": regret}, step=t)
            gap = np.nan_to_num(
                (opt_y_observed - initial_opt_y)
                / (mat_bench.ground_truth_opt - initial_opt_y),
                nan=1.0,
            )
            wandb.log({"trace_gap": gap}, step=t)

        pbar.set_description(
            f"[Best f(x)={opt_y_observed:.3f}, True best={true_best:.3f}, "
            f"curr f(x)={new_data[target_col]:.3f}]"
        )

        if args.early_stopping and best_y >= mat_runner.ground_truth_max_transformed:
            for j in range(t + 1, args.exp_len + 1):
                trace_best_y[j]  = trace_best_y[t + 1]
                trace_y_his[j]   = trace_y_his[t + 1]
                trace_timing[j]  = timing
                trace_acqvals[j] = 0.0

                if wandb is not None:
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
                    wandb.log({"trace_regret": regret}, step=j)
                    gap = np.nan_to_num(
                        (opt_y_observed - initial_opt_y)
                        / (mat_bench.ground_truth_opt - initial_opt_y),
                        nan=1.0,
                    )
                    wandb.log({"trace_gap": gap}, step=j)
            break

    _save(args, mat_bench, timing_train, timing_preds,
          trace_acqvals, trace_y_his, trace_best_y, trace_timing)

    return trace_best_y, None


# --------------------------------------------------------------------------
# entry point (mirrors llmat_bo.llmat_bo)
# --------------------------------------------------------------------------

def decoupled_mcts_lfbo_bo(args, mat_bench, wandb=None):
    """Entry point called by run_benchmarks.run_mat when algorithm='decoupled_mcts_lfbo'."""
    return run_bo_decoupled(args, mat_bench, wandb)
