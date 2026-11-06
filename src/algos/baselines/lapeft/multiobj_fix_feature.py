#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: multiobj_fix_feature.py
# Description: Multi-objective Bayesian Optimization on cached features
"""
import numpy as np
import torch
import torch.utils.data as data_utils
import tqdm
import os
from src.bayesopt.laplace_botorch import LaplaceBoTorch
from src.bayesopt.acqf import ucb, thompson_sampling_multivariate, scalarize
from src.utils import helpers
from src.utils.helpers import trace_times
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import infer_reference_point, Hypervolume

device = helpers.check_device()
print(f"Using device: {device}")




def save_results_multiobj(args, mat_bench, timing_train, timing_preds,
                          trace_hypervolume, trace_scalarized_best_y,
                          trace_best_y, trace_timing):
    """Save multi-objective optimization results."""
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/multiobj/{prefix}/{args.algorithm}"
    if not os.path.exists(path):
        os.makedirs(path)

    suffix = f"{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.fix_args['method']}_{args.seed}"

    np.save(f"{path}/timing_train_{suffix}.npy", timing_train)
    np.save(f"{path}/timing_preds_{suffix}.npy", timing_preds)
    np.save(f"{path}/trace_hypervolume_{suffix}.npy", trace_hypervolume)
    np.save(f"{path}/trace_scalarized_best_y_{suffix}.npy", trace_scalarized_best_y)
    np.save(f"{path}/trace_best_y_{suffix}.npy", trace_best_y)
    np.save(f"{path}/trace_timing_{suffix}.npy", trace_timing)


def run_multiobj_fix_ft(args, mat_runner, wandb=None):
    """
    Run multi-objective Bayesian Optimization on cached features.
    Supports multiple objectives with scalarization and hypervolume tracking.
    """
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    mat_bench = mat_runner.mat_bench
    feature_dim = mat_bench.dataset['features'][0].shape[-1]
    print("Feature dim:", feature_dim)

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
        # Default: assume all objectives should be maximized
        MAXIMIZATIONS = [True] * n_objectives
    print(f"Maximizations: {MAXIMIZATIONS}")

    if mat_bench.feature_type in ["fingerprints", "molformer"]:
        print(f"Test Function: {mat_bench.data_name}; Feature Type: {mat_bench.feature_type}; Randseed: {args.seed}")
    else:
        print(f"Test Function: {mat_bench.data_name}; Foundation LLM: {mat_bench.feature_type}; "
              f"Prompt Type: {args.prompt_type}; Reduction: {args.feature_reduction}; Randseed: {args.seed}")
    print("--" * 10)

    # Initialize dataset
    dataset_train = mat_runner.generate_initialization(args.n_init_data)
    features = dataset_train["features"].tolist()

    # Get multi-objective targets
    # The objectives in OBJ_COLS are already prepared for maximization
    targets = dataset_train[OBJ_COLS].values.tolist()

    # Initialize training data
    train_x, train_y = [], []
    while len(train_x) < args.n_init_data:
        idx = np.random.randint(len(features))
        train_x.append(torch.tensor(features.pop(idx)))
        train_y.append(torch.tensor(targets.pop(idx)))

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)  # Shape: (n_init_data, n_objectives)

    assert train_x.shape == (args.n_init_data, feature_dim)
    assert train_y.shape == (args.n_init_data, n_objectives)

    # Get ground truth information
    # Use objectives from OBJ_COLS, which are prepared for maximization
    all_targets_tensor = torch.tensor(mat_runner.mat_bench.dataset[OBJ_COLS].values)
    
    # Calculate ground_truth_maxs from the correctly transformed objectives
    ground_truth_maxs = all_targets_tensor.max(dim=0)[0]

    scalarized_ground_truth_max = scalarize(ground_truth_maxs).item()

    # Compute reference point and max hypervolume for tracking
    Y_pareto = all_targets_tensor[is_non_dominated(all_targets_tensor)]
    ref_point = infer_reference_point(Y_pareto)
    hv = Hypervolume(ref_point)
    max_hypervolume = hv.compute(Y_pareto)

    print(f"Ground-truth best per objective: {ground_truth_maxs}")
    print(f"Scalarized ground-truth max: {scalarized_ground_truth_max:.3f}")
    print(f"Reference point: {ref_point}")
    print(f"Max hypervolume: {max_hypervolume:.3f}")

    # Initialize model
    def get_net():
        activation = (torch.nn.Tanh
                     if mat_bench.feature_name == "fingerprints"
                     and mat_bench.data_name != "photoswitch"
                     else torch.nn.ReLU)
        return torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 50),
            activation(),
            torch.nn.Linear(50, 50),
            activation(),
            torch.nn.Linear(50, n_objectives),  # Output n_objectives values
        )

    if args.fix_args['method'] == "laplace":
        noise_var = 0.01 if mat_bench.data_name == "laser" else 0.001
        model = LaplaceBoTorch(
            get_net,
            train_x,
            train_y,
            noise_var=noise_var,
            hess_factorization="kron" if mat_bench.feature_type != "llama-2-7b" else "diag",
        )
        model = model.to(device) if model is not None else model
    else:  # Random search
        model = None

    # Initialize best values
    scalarized_best_y = scalarize(train_y).max().item()
    best_y_idx = scalarize(train_y).argmax()
    best_y = train_y[best_y_idx]

    # Initialize tracking arrays
    trace_best_y = [[gt.item() for gt in ground_truth_maxs] for _ in range(args.exp_len + 1)]
    trace_scalarized_best_y = [scalarized_ground_truth_max] * (args.exp_len + 1)
    trace_hypervolume = [max_hypervolume] * (args.exp_len + 1)
    trace_timing = [0.0] * (args.exp_len + 1)

    timing_train = []
    timing_preds = []

    # Get remaining features and targets
    features = mat_runner.mat_bench.dataset['features'].to_list()
    targets = mat_runner.mat_bench.dataset[OBJ_COLS].values.tolist()

    # pbar = tqdm.trange(args.exp_len)
    # pbar.set_description(f"[Best scal'd f(x) = {scalarized_best_y:.3f}]")

    # for i in pbar:
    #     # Record start time
    #     start, end = trace_times(None, None, device)
    #     features = [torch.tensor(f, dtype=torch.float32) for f in features]
    #     targets = [torch.tensor(t, dtype=torch.float32) for t in targets]

    #     if args.fix_args['method'] == "random":
    #         start_pred, end_pred = trace_times(None, None, device)
    #         idx = np.random.randint(len(features))
    #         new_x = features.pop(idx)
    #         new_y = targets.pop(idx)
    #         timing_preds.append(trace_times(start_pred, end_pred, device))

    #         # Update train_y for hypervolume computation
    #         train_y = torch.cat([train_y, new_y.unsqueeze(0)], dim=0)

    #         # Evaluate hypervolume
    #         volume = hv.compute(train_y[is_non_dominated(train_y)])

    #     else:  # Laplace or other surrogate-based methods
    #         ### Estimate acquisition function with batch data
    #         dataloader = data_utils.DataLoader(
    #             data_utils.TensorDataset(
    #                 torch.stack(features),
    #                 torch.stack(targets)
    #             ),
    #             batch_size=256,
    #             shuffle=False,
    #         )

    #         preds, uncerts, labels = [], [], []
    #         acq_vals = []

    #         start_pred, end_pred = trace_times(None, None, device)

    #         for x, y in dataloader:
    #             # Get posterior mean and covariance
    #             f_mean, f_cov = model.posterior(x, return_mean_cov_only=True)
    #             f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1)

    #             if args.acqf == "ucb":
    #                 # Scalarize UCB values
    #                 acq_vals.append(scalarize(ucb(f_mean, f_var)))
    #             else:  # Thompson sampling
    #                 samples = thompson_sampling_multivariate(f_mean, f_cov)
    #                 acq_vals.append(scalarize(samples))

    #             preds.append(f_mean)
    #             uncerts.append(f_var.sqrt())
    #             labels.append(y)

    #         timing_preds.append(trace_times(start_pred, end_pred, device))

    #         acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
    #         preds, uncerts, labels = (
    #             torch.cat(preds, dim=0).cpu(),
    #             torch.cat(uncerts, dim=0).cpu(),
    #             torch.cat(labels, dim=0)
    #         )
    #         test_loss = torch.nn.MSELoss()(preds, labels).item()

    #         # Pick the best acquisition value
    #         idx_best = torch.argmax(acq_vals).item()
    #         new_x, new_y = features.pop(idx_best), targets.pop(idx_best)

    #     # Update best scalarized value
    #     if scalarize(new_y).item() > scalarized_best_y:
    #         best_y = new_y
    #         scalarized_best_y = scalarize(new_y).item()

    #     # Update progress bar
    #     if args.fix_args['method'] == "random":
    #         pbar.set_description(
    #             f"[Best scal'd f(x) = {scalarize(best_y).item():.3f}, "
    #             f"curr scal'd f(x) = {scalarize(new_y).item():.3f}, "
    #             f"curr hypervolume = {volume:.3f}]"
    #         )
    #         start_train, end_train = trace_times(None, None, device)
    #         timing_train.append(trace_times(start_train, end_train, device))
    #     else:
    #         # Update surrogate
    #         start_train, end_train = trace_times(None, None, device)
    #         model = model.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))
    #         timing_train.append(trace_times(start_train, end_train, device))

    #         # Evaluate hypervolume
    #         volume = hv.compute(model.train_Y[is_non_dominated(model.train_Y)])

    #         pbar.set_description(
    #             f"[Best scal'd f(x) = {scalarize(best_y).item():.3f}, "
    #             f"curr scal'd f(x) = {scalarize(new_y).item():.3f}, "
    #             f"curr hypervolume = {volume:.3f}, "
    #             f"test MSE = {test_loss:.3f}]"
    #         )

    #     # Record timing
    #     timing = trace_times(start, end, device)

    #     # Update traces
    #     trace_best_y[i + 1] = [by.item() for by in best_y]
    #     trace_scalarized_best_y[i + 1] = scalarized_best_y
    #     trace_hypervolume[i + 1] = volume
    #     trace_timing[i + 1] = timing

    #     # Wandb logging
    #     if wandb is not None:
    #         wandb.log({"trace_scalarized_best_y": scalarized_best_y}, step=i)
    #         wandb.log({"trace_hypervolume": volume}, step=i)
    #         wandb.log({"trace_timing": timing}, step=i)
    #         wandb.log({"trace_timing_train": timing_train[-1]}, step=i)
    #         wandb.log({"trace_timing_pred": timing_preds[-1]}, step=i)

    #         # Log individual objectives
    #         for obj_idx, obj_name in enumerate(OBJ_COLS):
    #             wandb.log({f"trace_best_{obj_name}": trace_best_y[i + 1][obj_idx]}, step=i)

    #     # Early stopping
    #     if args.early_stopping and volume >= max_hypervolume:
    #         print(f"Reached maximum hypervolume at iteration {i+1}")
    #         for j in range(i + 1, args.exp_len + 1):
    #             if wandb is not None:
    #                 wandb.log({"trace_scalarized_best_y": scalarized_best_y}, step=j)
    #                 wandb.log({"trace_hypervolume": volume}, step=j)
    #                 wandb.log({"trace_timing": timing}, step=j)
    #                 wandb.log({"trace_timing_train": timing_train[-1]}, step=j)
    #                 wandb.log({"trace_timing_pred": timing_preds[-1]}, step=j)
    #         break
# Initialize observed_targets with initial train_y
    observed_targets = train_y.clone()

    pbar = tqdm.trange(args.exp_len)
    pbar.set_description(f"[Best scal'd f(x) = {scalarize(observed_targets).max().item():.3f}]")

    for i in pbar:
        start, end = trace_times(None, None, device)

        # Convert remaining features/targets to tensors
        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        targets = [torch.tensor(t, dtype=torch.float32) for t in targets]

        # === Step 1: Select next point ===
        if args.fix_args['method'] == "random":
            start_pred, end_pred = trace_times(None, None, device)
            idx = np.random.randint(len(features))
            new_x = features.pop(idx)
            new_y = targets.pop(idx)
            timing_preds.append(trace_times(start_pred, end_pred, device))
        else:
            # Surrogate-based acquisition
            dataloader = data_utils.DataLoader(
                data_utils.TensorDataset(torch.stack(features), torch.stack(targets)),
                batch_size=256, shuffle=False,
            )

            preds, uncerts, labels, acq_vals = [], [], [], []

            start_pred, end_pred = trace_times(None, None, device)

            for x_batch, y_batch in dataloader:
                f_mean, f_cov = model.posterior(x_batch, return_mean_cov_only=True)
                f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1)

                if args.acqf == "ucb":
                    acq_vals.append(scalarize(ucb(f_mean, f_var)))
                else:
                    samples = thompson_sampling_multivariate(f_mean, f_cov)
                    acq_vals.append(scalarize(samples))

                preds.append(f_mean)
                uncerts.append(f_var.sqrt())
                labels.append(y_batch)

            timing_preds.append(trace_times(start_pred, end_pred, device))

            acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
            preds, uncerts, labels = (
                torch.cat(preds, dim=0).cpu(),
                torch.cat(uncerts, dim=0).cpu(),
                torch.cat(labels, dim=0)
            )

            test_loss = torch.nn.MSELoss()(preds, labels).item()
            idx_best = torch.argmax(acq_vals).item()
            new_x, new_y = features.pop(idx_best), targets.pop(idx_best)

        # === Step 2: Update observed_targets and model ===
        observed_targets = torch.cat([observed_targets, new_y.unsqueeze(0)], dim=0)
        model = model.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))
        start_train, end_train = trace_times(None, None, device)
        timing_train.append(trace_times(start_train, end_train, device))

        # === Step 3: Update Pareto front and hypervolume ===
        pareto_mask = is_non_dominated(observed_targets)
        Y_pareto_current = observed_targets[pareto_mask]
        current_hv = hv.compute(Y_pareto_current)

        # === Step 4: Update best scalarized value from Pareto front ===
        scalarized_vals = torch.tensor([scalarize(pt).item() for pt in Y_pareto_current])
        idx_best_pareto = scalarized_vals.argmax()
        current_best_point = Y_pareto_current[idx_best_pareto]
        scalarized_best_y = scalarized_vals[idx_best_pareto].item()
        best_y = current_best_point

        #Record timing
        timing = trace_times(start, end, device)

        # Update traces
        trace_best_y[i + 1] = [by.item() for by in best_y]
        trace_scalarized_best_y[i + 1] = scalarized_best_y
        trace_hypervolume[i + 1] = current_hv
        trace_timing[i + 1] = timing

        # Wandb logging
        if wandb is not None:
            wandb.log({"trace_scalarized_best_y": scalarized_best_y}, step=i)
            wandb.log({"trace_hypervolume": current_hv}, step=i)
            wandb.log({"trace_timing": timing}, step=i)
            wandb.log({"trace_timing_train": timing_train[-1]}, step=i)
            wandb.log({"trace_timing_pred": timing_preds[-1]}, step=i)

            # Log individual objectives
            for obj_idx, obj_name in enumerate(OBJ_COLS):
                wandb.log({f"trace_best_{obj_name}": trace_best_y[i + 1][obj_idx]}, step=i)

        # Early stopping
        if args.early_stopping and current_hv >= max_hypervolume:
            print(f"Reached maximum hypervolume at iteration {i+1}")
            for j in range(i + 1, args.exp_len + 1):
                if wandb is not None:
                    wandb.log({"trace_scalarized_best_y": scalarized_best_y}, step=j)
                    wandb.log({"trace_hypervolume": current_hv}, step=j)
                    wandb.log({"trace_timing": timing}, step=j)
                    wandb.log({"trace_timing_train": timing_train[-1]}, step=j)
                    wandb.log({"trace_timing_pred": timing_preds[-1]}, step=j)
            break
    # Save results
    save_results_multiobj(
        args, mat_runner.mat_bench, timing_train, timing_preds,
        trace_hypervolume, trace_scalarized_best_y, trace_best_y, trace_timing
    )

    return trace_hypervolume, trace_scalarized_best_y
