#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Sat Jul  5 19:20:25 2025
# File Name: fix_feature.py
# Description:
"""
import numpy as np
import torch
import torch.utils.data as data_utils
import tqdm
import os
from gpytorch.kernels import ScaleKernel, MaternKernel
from src.bayesopt.kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.likelihoods import GaussianLikelihood
from src.bayesopt.gp_baselines import MLLGP
from src.bayesopt.laplace_botorch import LaplaceBoTorch
from src.bayesopt.tpe import TPESelector
from src.bayesopt.acqf import ucb, ei, thompson_sampling
from src.utils import helpers
from src.utils.helpers import trace_times
import math

device = helpers.check_device()
print(f"Using device: {device}")


def save_results_fix(args, mat_bench, timing_train, timing_preds,\
                           trace_best_acqval, trace_best_y, trace_y_his, trace_timing):

    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/{args.algorithm}"
    if not os.path.exists(path):
        os.makedirs(path)
    suffix = f"{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.fix_args['method']}_{args.seed}"
    np.save(f"{path}/timing_train_{suffix}.npy", timing_train)
    np.save(f"{path}/timing_preds_{suffix}.npy", timing_preds)
    np.save(f"{path}/trace_best_acqval_{suffix}.npy", trace_best_acqval)
    np.save(f"{path}/trace_best_y_{suffix}.npy", trace_best_y)
    np.save(f"{path}/trace_y_his_{suffix}.npy", trace_y_his)
    np.save(f"{path}/trace_timing_{suffix}.npy", trace_timing)


# ============================= run BO on cached features ===========================


def run_fix_ft(args, mat_runner, wandb=None):
    """
        use cached feature as the search region    
    """
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    mat_bench = mat_runner.mat_bench
    feature_dim = mat_bench.dataset['features'][0].shape[-1]
    print("feature dim:", feature_dim)
    MAXIMIZATION = mat_bench.maximization

    if mat_bench.feature_type in ["fingerprints", "molformer", "gin", "schnet"]:
        print(f"Test Function: {mat_bench.data_name}; Feature Type: {mat_bench.feature_type}; Randseed: {args.seed}")
    else:
        print(f"Test Function: {mat_bench.data_name}; Foundation LLM: {mat_bench.feature_type};\
               Prompt Type: {args.prompt_type}; Reduction: {args.feature_reduction}; Randseed: {args.seed}")
    print("--" * 10)
    #
    # mat_runner = MATExpRunner(mat_bench, seed, args.normalize_y)
    true_best = mat_bench.ground_truth_max
    dataset_train = mat_runner.generate_initialization(args.n_init_data)
    print(dataset_train)
    features, targets = dataset_train["features"].tolist(), dataset_train["targets_transformed"].tolist()
    # print(features, targets)
    train_x, train_y = [], []
    while len(train_x) < args.n_init_data:
        idx = np.random.randint(len(features))
        train_x.append(torch.tensor(features.pop(idx)))
        train_y.append(torch.tensor(targets.pop(idx)))
    print(type(train_x))
    train_x, train_y = torch.stack(train_x), torch.stack(train_y).view(-1, 1)

    def get_net():
        activation = (torch.nn.Tanh \
                    if mat_bench.feature_name == "fingerprints" \
                    and mat_bench.data_name != "photoswitch"\
                    else torch.nn.ReLU)
        return torch.nn.Sequential(torch.nn.Linear(feature_dim, 50), \
                                    activation(), torch.nn.Linear(50, 50), \
                                    activation(), torch.nn.Linear(50, 1),)

    if args.fix_args['method'] == "laplace":
        model = LaplaceBoTorch(
            get_net,
            train_x,
            train_y,
            noise_var=0.001,
            hess_factorization="kron" if mat_bench.feature_type != "llama-2-7b" else "diag",
        )
        model = model.to(device) if model is not None else model
    elif args.fix_args['method'] == "gp":
        kernel = TanimotoKernel() if mat_bench.feature_type == "fingerprints" else MaternKernel()
        if args.length_scale and mat_bench.feature_type != "fingerprints":
            print("Using \sqrt{d} as length scale")
            ls = torch.ones_like(kernel.lengthscale) * math.sqrt(feature_dim)
            kernel._set_lengthscale(ls)

        print(train_y)

        model = MLLGP(train_x, train_y, ScaleKernel(kernel), GaussianLikelihood())

    elif args.fix_args['method'] == "tpe":
        model = TPESelector(gamma=0.25, maximize=MAXIMIZATION)
        print(f"Using TPE optimizer with gamma={0.25}, maximize={MAXIMIZATION}")
    else:  # Random search
        model = None

    best_y = train_y.max().item()
    print(best_y)

    trace_best_y = [helpers.y_transform(mat_bench.ground_truth_max, MAXIMIZATION)] * (args.exp_len + 1)
    trace_y_his = [helpers.y_transform(mat_bench.ground_truth_max, MAXIMIZATION)] * (args.exp_len + 1)
    trace_timing = [0.0] * (args.exp_len + 1)
    trace_best_acqval = []

    timing_train = []
    timing_preds = []

    features = mat_runner.mat_bench.dataset['features'].to_list()
    targets = mat_runner.mat_bench.dataset['targets_transformed'].to_list()

    pbar = tqdm.trange(args.exp_len)
    pbar.set_description(f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}]")
    for i in pbar:
        # Each BO round
        # record the start time
        start, end = trace_times(None, None, device)
        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        targets = [torch.tensor(t, dtype=torch.float32) for t in targets]
        if args.fix_args['method'] == "random":
            start_pred, end_pred = trace_times(None, None, device)
            idx = np.random.randint(len(features))
            new_x = features.pop(idx)
            new_y = targets.pop(idx)
            timing_preds.append(trace_times(start_pred, end_pred, device))
        elif args.fix_args['method'] == "tpe":
            # TPE uses suggest method with candidates
            start_pred, end_pred = trace_times(None, None, device)

            # Convert features to numpy array for TPE
            features_np = [f.numpy() if isinstance(f, torch.Tensor) else np.array(f) for f in features]

            # Get suggestion from TPE
            idx = model.suggest(features_np)

            new_x = features.pop(idx)
            new_y = targets.pop(idx)
            timing_preds.append(trace_times(start_pred, end_pred, device))

            # For TPE, we don't compute acquisition values in the same way
            trace_best_acqval.append(0.0)  # Placeholder
        else:
            ### estimate acq func with a batch data to select candidate point
            dataloader = data_utils.DataLoader(
                data_utils.TensorDataset(torch.stack(features),
                                         torch.stack(targets).view(-1, 1)),
                batch_size=256,
                shuffle=False,
            )

            preds, uncerts, labels = [], [], []
            acq_vals = []
            # record the start time for predictions
            start_pred, end_pred = trace_times(None, None, device)

            for x, y in dataloader:
                posterior = model.posterior(x)
                f_mean, f_var = posterior.mean, posterior.variance
                if args.acqf == "ei":
                    acq_vals.append(ei(f_mean, f_var, best_y))
                elif args.acqf == "ucb":
                    acq_vals.append(ucb(f_mean, f_var))
                else:
                    acq_vals.append(thompson_sampling(f_mean, f_var))

                preds.append(f_mean)
                uncerts.append(f_var.sqrt())
                labels.append(y)
            # record the end time for predictions
            timing_preds.append(trace_times(start_pred, end_pred, device))
            acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
            preds, uncerts, labels = (torch.cat(preds, dim=0).cpu(), torch.cat(uncerts, dim=0).cpu(), torch.cat(labels, dim=0))
            test_loss = torch.nn.MSELoss()(preds, labels).item()
            print(">>>>>> acq values >>>>>>")
            print(acq_vals)
            # Pick a molecule (a row in the current dataset) that maximizes the acquisition
            idx_best = torch.argmax(acq_vals).item()
            new_x, new_y = features.pop(idx_best), targets.pop(idx_best)  # remove visited data point
            print(">>>>>>> best idx >>>>>")
            print(idx_best)
            print(">>>>> selected y >>>>>")
            print(new_y)
            print(len(features), len(targets))

            trace_best_acqval.append(torch.max(acq_vals).item())
        # new_y = new_y.item()
        # Update the current best y
        if new_y.item() > best_y:
            best_y = new_y.item()

        if args.fix_args['method'] == "random":
            pbar.set_description(f"[Current Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, "\
                               + f"True Best f(x) = {helpers.y_transform(true_best, MAXIMIZATION):.3f},"
                               + f"curr f(x) = {helpers.y_transform(new_y.item(), MAXIMIZATION):.3f}]")
            start_train, end_train = trace_times(start=None, end=None, device=device)
            # record the end time for training
            timing_train.append(trace_times(start_train, end_train, device))

        elif args.fix_args['method'] == "tpe":
            pbar.set_description(f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, "\
                               + f"True Best f(x) = {helpers.y_transform(true_best, MAXIMIZATION):.3f},"
                               + f"curr f(x) = {helpers.y_transform(new_y.item(), MAXIMIZATION):.3f}]")
            start_train, end_train = trace_times(start=None, end=None, device=device)
            # TPE requires updating the history with the new observation
            new_x_np = new_x.numpy() if isinstance(new_x, torch.Tensor) else np.array(new_x)
            new_y_val = new_y.item() if isinstance(new_y, torch.Tensor) else float(new_y)
            model.condition_on_observations(new_x_np, new_y_val)
            # Update train_x and train_y for tracking purposes
            train_x = torch.cat([train_x, new_x.unsqueeze(0)], dim=0)
            train_y = torch.cat([train_y, new_y.view(1, 1)], dim=0)
            timing_train.append(trace_times(start_train, end_train, device))
        else:
            pbar.set_description(f"[Best f(x) = {helpers.y_transform(best_y, MAXIMIZATION):.3f}, "\
                               + f"True Best f(x) = {helpers.y_transform(true_best, MAXIMIZATION):.3f},"
                               + f"curr f(x) = {helpers.y_transform(new_y.item(), MAXIMIZATION):.3f}, test MSE = {test_loss:.3f}]")

            # Update surrogate
            # record the start time for training
            start_train, end_train = trace_times(start=None, end=None, device=device)
            model = model.condition_on_observations(new_x.unsqueeze(0), new_y.unsqueeze(0))
            # record the end time for training
            timing_train.append(trace_times(start_train, end_train, device))

        # Housekeeping
        # record the end time
        timing = trace_times(start, end, device)
        trace_best_y[i + 1] = helpers.y_transform(best_y, MAXIMIZATION)
        trace_y_his[i + 1] = helpers.y_transform(new_y, MAXIMIZATION)
        trace_timing[i + 1] = timing

        if wandb is not None:
            wandb.log({"trace_best_y": trace_best_y[i + 1]}, step=i)
            wandb.log({"trace_y_his": trace_y_his[i + 1]}, step=i)
            wandb.log({"trace_timing": timing}, step=i)
            wandb.log({"trace_timing_train": timing_train[-1]}, step=i)
            wandb.log({"trace_timing_pred": timing_preds[-1]}, step=i)
            if args.fix_args['method'] not in ["random", "smac", "tpe"]:
                wandb.log({"trace_acqvals": acq_vals[idx_best].item()}, step=i)
            elif args.fix_args['method'] in ["smac", "tpe"]:
                wandb.log({"trace_acqvals": 0.0}, step=i)  # Placeholder for SMAC/TPE
            if mat_bench.maximization:
                regret = mat_bench.ground_truth_opt - np.mean(trace_y_his[1:i + 2])
            else:
                regret = np.mean(trace_y_his[1:i + 2]) - mat_bench.ground_truth_opt
            #GAP = (yi− y0) / (y∗− y0),
            # GAP = (best_y - initial_best_y) / (mat_bench.ground_truth_opt - initial_best_y)
            y_0 = trace_best_y[1]
            y_t = trace_best_y[i + 1]
            GAP = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
            wandb.log({"trace_regret": regret}, step=i)
            wandb.log({"trace_gap": GAP}, step=i)
            # print("regret:", regret)
        if args.early_stopping:
            # Early stopping if we already got the max
            if best_y >= mat_bench.ground_truth_max:
                print("find the best")
                print(best_y, mat_bench.ground_truth_max)
                for j in range(i + 1, args.exp_len + 1):
                    wandb.log({"trace_best_y": trace_best_y[i + 1]}, step=j)
                    wandb.log({"trace_y_his": trace_y_his[i + 1]}, step=j)
                    wandb.log({"trace_timing": timing}, step=j)
                    wandb.log({"trace_timing_train": timing_train[-1]}, step=j)
                    wandb.log({"trace_timing_pred": timing_preds[-1]}, step=j)
                    if args.fix_args['method'] != "random":
                        wandb.log({"trace_acqvals": 0}, step=j)
                    if mat_bench.maximization:
                        regret = mat_bench.ground_truth_opt - np.mean(trace_y_his[1:j + 2])
                    else:
                        regret = np.mean(trace_y_his[1:j + 2]) - mat_bench.ground_truth_opt
                    # GAP = (best_y - initial_best_y) / (mat_bench.ground_truth_opt - initial_best_y)
                    y_0 = trace_best_y[1]
                    y_t = trace_best_y[j]
                    GAP = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                    wandb.log({"trace_gap": GAP}, step=j)
                    wandb.log({"trace_regret": regret}, step=j)
                break

    save_results_fix(args, mat_runner.mat_bench, timing_train,\
                          timing_preds, trace_best_acqval, trace_best_y, trace_y_his, trace_timing)
    return trace_best_y, None
