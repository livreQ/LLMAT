#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Tue Aug 19 17:10:05 2025
# File Name: load_results.py
# Description:
"""
import numpy as np
import os
import pandas as pd
from benchmarks.MAT.material_benchmarks import MATBench


def get_ours_suffix(data_name, f_model, seed, n_init_data=10, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.005, tree_depth=3, finetune=True):
    if f_model == "t5-base-chem":
        if finetune:
            gamma = 0.5
            lmbda = 0.5
            eta = 0.005
            if data_name == "redoxmer":
                lora_lr = "5e-5"
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            elif data_name == "solvation":
                lora_lr = "3e-5"
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            # elif data_name == "kinase":
            #     lora_lr = "3e-5"
            #     lora_reg = 0
            #     suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            # elif data_name == "laser":
            #     lora_lr = "5e-5" # "1e-5"
            #     lora_reg = 0 # 0.001
            #     suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            elif data_name == "photoswitch":
                eta = 0.01
                lora_lr = "0"
                # lora_lr = "1e-7"
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            # elif data_name == "pce":
            #     eta = 0.01
            #     lora_lr = "3e-5" # "1e-7"
            #     lora_reg = 0
            #     suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            else:
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
        else:
            # gamma = 0.5
            # lmbda = 0.5
            eta = 0.01
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
        return suffix
    elif f_model == "molformer":
        if finetune:
            gamma = 0.5
            lmbda = 0.5
            eta = 0.005
            if data_name == "redox-mer":
                lora_lr = "3e-5"
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            elif data_name == "solvation":
                lora_lr = "3e-5"
                tree_depth = 4
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            elif data_name == "kinase":
                lora_lr = "3e-5"
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            elif data_name == "laser":
                lora_lr = "3e-5"  # "1e-5"
                lora_reg = 0  # 0.001
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            elif data_name == "photoswitch":
                eta = 0.005
                lora_lr = "1e-6"
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            elif data_name == "pce":
                eta = 0.005
                lora_lr = "1e-7"  # "1e-7"
                lora_reg = 0
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{lora_lr}_{lora_reg}_{seed}"
            else:
                suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
        else:
            # gamma = 0.5
            # lmbda = 0.5
            eta = 0.005
            suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
        return suffix
    else:
        suffix = f"{n_init_data}_{acqf}_{gamma}_{eta}_{tree_depth}_{seed}"
        return suffix


def load_ours(mat_bench, f_model, r_type, seed, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.01, tree_depth=3, finetune=True):

    # print(mat_bench.dataset_name)
    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/llmat"
    clustering = mat_bench.dataset_name.split("/")[-2]
    path = f"{path}/{clustering}"
    data_name = mat_bench.dataset_name.split("/")[0]
    suffix = get_ours_suffix(data_name, f_model, seed, n_init_data=10, acqf="ei", gamma=0.5, lmbda=0.5, eta=0.005, tree_depth=3, finetune=finetune)
    if not os.path.exists(f"{path}/trace_best_y_{suffix}.npy"):
        print(f"{path}/trace_best_y_{suffix}.npy")
        return None
    if r_type == "best_y":
        ret = np.load(f"{path}/trace_best_y_{suffix}.npy")
    elif r_type == "trace_time":
        ret = np.load(f"{path}/trace_timing_{suffix}.npy")
    elif r_type == "train_time":
        ret = np.load(f"{path}/timing_train_{suffix}.npy")
    elif r_type == "pred_time":
        ret = np.load(f"{path}/timing_preds_{suffix}.npy")
    elif r_type == "trace_acqvals":
        ret = np.load(f"{path}/trace_acqvals_{suffix}.npy")
    elif r_type == "trace_y_his":
        ret = np.load(f"{path}/trace_y_his_{suffix}.npy")
    else:
        ret = None
    return ret


def get_baselines_suffix(seed, n_init_data=10, acqf="ts", laplace_type="all_layer", base_line="gp", finetune=False):

    if not finetune:
        suffix = f"{n_init_data}_{acqf}_{laplace_type}_{base_line}_{seed}"
    else:
        suffix = f"{n_init_data}_{acqf}_{laplace_type}_{seed}"
    return suffix


def load_baselines(mat_bench, r_type, seed, acqf="ts", laplace_type="all_layer", base_line="gp", finetune=False):

    prefix = "/".join(mat_bench.dataset_name.split("/")[:-2])
    path = f"results/{prefix}/lapeft"
    suffix = get_baselines_suffix(seed, base_line=base_line, finetune=finetune)
    # print(suffix)
    if not os.path.exists(f"{path}/trace_best_y_{suffix}.npy"):
        print(f"{path}/trace_best_y_{suffix}.npy")
        return None
    else:
        if r_type == "best_y":
            ret = np.load(f"{path}/trace_best_y_{suffix}.npy")
        elif r_type == "trace_time":
            ret = np.load(f"{path}/trace_timing_{suffix}.npy")
        elif r_type == "train_time":
            ret = np.load(f"{path}/timing_train_{suffix}.npy")
        elif r_type == "pred_time":
            ret = np.load(f"{path}/timing_preds_{suffix}.npy")
        elif r_type == "trace_acqvals":
            ret = np.load(f"{path}/trace_acqvals_{suffix}.npy")
        elif r_type == "trace_y_his":
            ret = np.load(f"{path}/trace_y_his_{suffix}.npy")
        else:
            ret = None
        return ret


if __name__ == "__main__":
    # Example usage
    mat_bench = MATBench(
        data_name="redox-mer",
        run_subset_only=False,
        feature_type="t5-base-chem",
        finetuning=True,
        iupac=False,
        prompt_type="default",
        randseed=42,
        clustering_type="kmeans",
        feature_reduction=None,
    )
    seed = 42
    r_type = "best_y"
    f_model = "t5-base-chem"

    best_y = load_ours(mat_bench, f_model, r_type, seed)
    print(best_y)

    baseline_best_y = load_baselines(mat_bench, r_type, seed)
    print(baseline_best_y)
