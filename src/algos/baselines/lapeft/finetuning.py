#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Sat Jul  5 19:20:34 2025
# File Name: finetuning.py
# Description:
"""

import numpy as np
import torch
import tqdm
import os
from src.bayesopt.acqf import ucb, ei, thompson_sampling
from src.utils import helpers

from src.algos.baselines.lapeft.foundation_models import MolFormerRegressor, get_molformer_tokenizer
from src.algos.baselines.lapeft.foundation_models import RobertaRegressor, get_roberta_tokenizer
from src.algos.baselines.lapeft.foundation_models import GPT2Regressor, get_gpt2_tokenizer
from src.algos.baselines.lapeft.foundation_models import Llama2Regressor, get_llama2_tokenizer
from src.algos.baselines.lapeft.foundation_models import T5Regressor, get_t5_tokenizer

from src.algos.baselines.lapeft.peft.lora import LoRALLMBayesOpt
from src.bayesopt.acqf import ucb, ei, thompson_sampling
from src.utils import helpers
from src.utils.configs import LaplaceConfig, LLMFeatureType
from peft import LoraConfig, get_peft_model
import math
from src.utils.helpers import trace_times

device = helpers.check_device()
print(f"Using device: {device}")


def save_results_finetuning(args, mat_bench, timing_train, timing_preds,\
        trace_acqvals, trace_best_y, trace_y_his, trace_timing):
    print(mat_bench.dataset_name)
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/{args.algorithm}"

    if not os.path.exists(path):
        os.makedirs(path)

    suffix = f"{args.n_init_data}_{args.acqf}_{args.laplace_type}_{args.seed}"
    np.save(f"{path}/timing_train_{suffix}.npy", timing_train)
    np.save(f"{path}/timing_preds_{suffix}.npy", timing_preds)
    np.save(f"{path}/trace_acqvals_{suffix}.npy", trace_acqvals)
    np.save(f"{path}/trace_best_y_{suffix}.npy", trace_best_y)
    np.save(f"{path}/trace_y_his_{suffix}.npy", trace_y_his)
    np.save(f"{path}/trace_timing_{suffix}.npy", trace_timing)


def get_model(mat_bench, token=False):
    print("llm_model:" + mat_bench.foundation_model)
    if mat_bench.foundation_model == "molformer":
        tokenizer = get_molformer_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = MolFormerRegressor(tokenizer)
        target_modules = ["query", "value"]
    elif "roberta" in mat_bench.foundation_model:
        tokenizer = get_roberta_tokenizer(mat_bench.foundation_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = RobertaRegressor(
            kind=mat_bench.foundation_model,
            tokenizer=tokenizer,
            reduction=LLMFeatureType.AVERAGE,
        )
        target_modules = ["query", "value"]
    elif "gpt2" in mat_bench.foundation_model:
        tokenizer = get_gpt2_tokenizer(mat_bench.foundation_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Regressor(
            kind=mat_bench.foundation_model,
            tokenizer=tokenizer,
            reduction=LLMFeatureType.AVERAGE,
        )
        target_modules = ["c_attn"]
    elif "llama-2" in mat_bench.foundation_model:
        tokenizer = get_llama2_tokenizer(mat_bench.foundation_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Llama2Regressor(
            kind=mat_bench.foundation_model,
            tokenizer=tokenizer,
            reduction=LLMFeatureType.AVERAGE,
        )
        target_modules = ["q_proj", "v_proj"]
    elif "t5" in mat_bench.foundation_model:
        if "chem" in mat_bench.foundation_model:
            foundation_model_real = "GT4SD/multitask-text-and-chemistry-t5-base-augm"
            tokenizer = get_t5_tokenizer(foundation_model_real)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = T5Regressor(
                kind=foundation_model_real,
                tokenizer=tokenizer,
                reduction=LLMFeatureType.AVERAGE,
            )
        else:
            tokenizer = get_t5_tokenizer(mat_bench.foundation_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = T5Regressor(
                kind=mat_bench.foundation_model,
                tokenizer=tokenizer,
                reduction=LLMFeatureType.AVERAGE,
            )
        target_modules = ["q", "v"]
    else:
        raise NotImplementedError

    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["head"],
    )
    lora_model = get_peft_model(model, config)
    for p in lora_model.base_model.head.original_module.parameters():
        p.requires_grad = False
    # for n, p in lora_model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    if token:
        return lora_model, tokenizer
    else:
        return lora_model


# ============================= run fintunning BO ===========================
def run_finetuning(args, mat_runner, wandb=None):
    """
        finetuning the LLM for Bayesian Optimization
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # MAXIMIZATION = mat_bench.maximization
    mat_bench = mat_runner.mat_bench
    lora_model, tokenizer = get_model(mat_bench, True)
    data_processor = mat_bench.get_data_processor(tokenizer)
    # if MAXIMIZATION is false, return a dataset with -y
    dataset_train = mat_runner.generate_initialization(args.n_init_data)
    print("dataset_train:", dataset_train)
    print("dataset_train colums:", dataset_train.columns)
    ground_truth_max = mat_runner.ground_truth_max_transformed  #dataset[mat_runner.OBJ_COL].max()
    ground_truth_max_ori = mat_bench.ground_truth_opt  #dataset[mat_runner.OBJ_COL_ORI].max()
    # MAXIMIZATION = True
    # Train + Laplace
    # ======== Configuration for Laplace approximation =============
    if args.laplace_type == "all_layer":
        config = LaplaceConfig(
            n_epochs=50,
            noise_var=0.001,
            hess_factorization="kron",
            subset_of_weights="all",
            marglik_mode="posthoc",
            prior_prec_structure="layerwise",
        )
    else:
        config = LaplaceConfig(
            n_epochs=30,
            noise_var=0.001,
            hess_factorization="full",
            subset_of_weights="last_layer",
        )
    if mat_bench.dataset_name == "photoswitch":
        config.lr = 1e-2
        config.lr_lora = 3e-3
    # =============================================================

    APPEND_EOS = mat_bench.foundation_model != "molformer" and ("t5" not in mat_bench.foundation_model)
    print(dataset_train)
    print("======== init Lora ==============")
    model = LoRALLMBayesOpt(
        lora_model,
        dataset_train,
        data_processor,
        dtype="float32",
        laplace_config=config,
        device=device,
        append_eos=APPEND_EOS,
    )
    print("--------finish init (initial model trained) ----------")
    target_col_transformed = mat_bench.target_col_transformed
    target_col = mat_bench.target_col
    id_max_observed = dataset_train[target_col_transformed].idxmax()
    best_y = dataset_train.loc[id_max_observed][target_col_transformed]
    best_y_ori = dataset_train.loc[id_max_observed][target_col]

    pbar = tqdm.trange(args.exp_len, position=0, colour="green", leave=True)
    pbar.set_description(f"[Best f(x) = {best_y_ori:.3f}]")

    trace_best_y = [ground_truth_max_ori] * (args.exp_len + 1)
    trace_y_his = [ground_truth_max_ori] * (args.exp_len + 1)
    trace_timing = [0.0] * (args.exp_len + 1)
    trace_acqvals = [-math.inf] * (args.exp_len + 1)
    timing_train = []
    timing_preds = []

    for i in pbar:
        # Timing
        start, end = trace_times(None, None, device)
        print("==========")
        # BO iteration
        dataset = mat_bench.dataset
        print("=========unobserved dataset ==========")
        print(dataset)
        dataloader = data_processor.get_dataloader(
            dataset,
            batch_size=args.finetuning_args["batch_size"],  #default 16
            shuffle=False,
            append_eos=APPEND_EOS,
        )  # redundant colums removed

        preds, uncerts, labels = [], [], []
        acq_vals = []
        sub_pbar = tqdm.tqdm(dataloader, position=1, colour="blue", \
                             desc="[Prediction over dataset]", leave=False,)
        start_pred, end_pred = trace_times(None, None, device)

        for data in sub_pbar:
            # print(data)
            posterior = model.posterior(data)
            f_mean, f_var = posterior.mean, posterior.variance
            if args.acqf == "ei":
                acq_vals.append(ei(f_mean, f_var, best_y))
            elif args.acqf == "ucb":
                acq_vals.append(ucb(f_mean, f_var))
            else:
                acq_vals.append(thompson_sampling(f_mean, f_var))

            preds.append(f_mean)
            uncerts.append(f_var.sqrt())
            labels.append(data["labels"].unsqueeze(-1))

        # Record the end time for predictions
        timing_preds.append(trace_times(start_pred, end_pred, device))
        acq_vals = torch.cat(acq_vals, dim=0).cpu().squeeze()
        preds, uncerts, labels = (torch.cat(preds, dim=0).cpu(),\
                                  torch.cat(uncerts, dim=0).cpu(),\
                                  torch.cat(labels, dim=0).cpu(),)
        test_loss = torch.nn.MSELoss()(preds, labels).item()

        _, idx = acq_vals.topk(k=10)

        for l, p, u, a in zip(labels[idx], preds[idx], uncerts[idx], acq_vals[idx]):
            print(f"True: {l.item():.3f}, Mean: {p.item():.3f}, Std: {u.item():.3f}, Acqf: {a.item():.3f}")
        # input()

        # Pick a molecule (a row in the current dataset) that maximizes the acquisition
        idx_best = torch.argmax(acq_vals).item()
        candidates_idx = dataset['Entry Number'].tolist()
        index = dataset[dataset['Entry Number'] == candidates_idx[idx_best]].index[0]
        new_data = helpers.pop_df(mat_bench.dataset, index)
        print("======== New data selected ========")
        print(new_data)
        print(">>>>>>> best idx >>>>>")
        print(index)
        print(">>>>> selected y >>>>>")
        print(new_data[target_col_transformed])
        print(labels[idx_best].item())
        # assert round(new_data[target_col_transformed], 8) == round(labels[idx_best].item(), 8), "Selected data does not match the label"
        # top_ys, _ = labels.topk(k=10)
        # print()
        # Update the current best y
        if new_data[target_col_transformed] > best_y:
            best_y = new_data[target_col_transformed]
            best_y_ori = new_data[target_col]
            print(best_y_ori)

        # record the start time for training
        start_train, end_train = trace_times(start=None, end=None, device=device)
        # ======== Update surrogate =========
        print("========Update model with newly added data========")
        model = model.condition_on_observations(new_data)
        # record the end time for training
        timing_train.append(trace_times(start_train, end_train, device))

        pbar.set_description(f"[Best f(x) = {best_y_ori:.3f}, "\
                           + f"True Best f(x) = {mat_bench.ground_truth_opt:.3f},"\
                           + f"curr f(x) = {new_data[target_col]:.3f}, "\
                           + f"test MSE: {test_loss:.3f}]")

        # Save results
        # record the end time
        timing = trace_times(start, end, device)
        trace_best_y[i + 1] = best_y_ori
        trace_y_his[i + 1] = new_data[target_col]
        trace_timing[i + 1] = timing
        if wandb is not None:
            wandb.log({"trace_best_y": trace_best_y[i + 1]}, step=i)
            wandb.log({"trace_y_his": trace_y_his[i + 1]}, step=i)
            wandb.log({"trace_timing": timing}, step=i)
            wandb.log({"trace_timing_train": timing_train[-1]}, step=i)
            wandb.log({"trace_timing_pred": timing_preds[-1]}, step=i)
            wandb.log({"trace_acqvals": acq_vals[idx_best].item()}, step=i)
            if mat_bench.maximization:
                regret = mat_bench.ground_truth_opt - np.mean(trace_y_his[1:i + 2])
                wandb.log({"trace_regret": regret}, step=i)
            else:
                regret = np.mean(trace_y_his[1:i + 2]) - mat_bench.ground_truth_opt
                wandb.log({"trace_regret": regret}, step=i)
            # print("regret:", regret)
            y_0 = trace_best_y[1]
            y_t = trace_best_y[i + 1]
            GAP = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
            #(y_t - y_0) / (mat_bench.ground_truth_opt - y_0)
            wandb.log({"trace_gap": GAP}, step=i)
        if args.early_stopping:
            # Early stopping if we already got the max
            if best_y >= ground_truth_max:
                for j in range(i + 1, args.exp_len + 1):
                    wandb.log({"trace_best_y": trace_best_y[i + 1]}, step=j)
                    wandb.log({"trace_y_his": trace_y_his[i + 1]}, step=j)
                    wandb.log({"trace_timing": timing}, step=j)
                    wandb.log({"trace_timing_train": timing_train[-1]}, step=j)
                    wandb.log({"trace_timing_pred": timing_preds[-1]}, step=j)
                    wandb.log({"trace_acqvals": 0}, step=j)
                    if mat_bench.maximization:
                        regret = mat_bench.ground_truth_opt - np.mean(trace_y_his[1:j + 2])
                    else:
                        regret = np.mean(trace_y_his[1:j + 2]) - mat_bench.ground_truth_opt
                    wandb.log({"trace_regret": regret}, step=j)
                    # GAP = (best_y - initial_best_y) / (mat_bench.ground_truth_opt - initial_best_y)
                    y_0 = trace_best_y[1]
                    y_t = trace_best_y[j]
                    GAP = np.nan_to_num((y_t - y_0) / (mat_bench.ground_truth_opt - y_0), nan=1.0)
                    wandb.log({"trace_gap": GAP}, step=j)
                break
    print("Finish BO")
    save_results_finetuning(args, mat_bench, timing_train, timing_preds, trace_acqvals, trace_best_y, trace_y_his, trace_timing)
    return trace_best_y, None
