#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.edu
# Created Time : 2025
# File Name: run_multiobj_benchmark.py
# Description: Run multi-objective Bayesian optimization benchmarks.
"""
import argparse
import os
import sys
import numpy as np
import torch
from src.utils.wandb_helpers import wandb_wrapper


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.main.args import parse_args
from benchmarks.MAT.multiobj_material_benchmarks import MultiObjMATBench


def obtain_llm_bo_object(bo_type, benchmark='multiobj_mat'):
    # if bo_type == 'llambo':
    #     from src.algos.baselines.llambo.bo_llambo import bo_llambo as bo_llm

    if bo_type == 'multiobj_lapeft':
        from src.algos.baselines.lapeft.multibo_lapeft import multibo_lapeft as bo_llm
    elif bo_type == 'multiobj_llmat':
        from src.algos.llmat.multiobj_llmat_bo import multiobj_llmat_bo as bo_llm
    else:
        pass
    return bo_llm

def main():
    args = parse_args()

    print("args.benchmark: ", args.benchmark)
    if args.benchmark == 'multiobj_mat':
        print("Running multi-objective benchmark")
    else:
        print("Running single-objective benchmark")
        exit()

    # --- Run Experiments ---
    list_f_model = args.benchmarks[args.benchmark]['feature_models']
    list_data = args.datasets  # not data id???

    print("list_f_model: ", list_f_model)
    print("list_data: ", list_data)
 
    for data_name in list_data:
        for f_model in list_f_model:
            if args.finetuning and f_model == "fingerprints":
                continue
            print("feature_model: ", f_model)
            iupac = args.iupac
            if data_name not in ["redox-mer"]:
                assert iupac == False
            print(f"Running experiment with seed: {args.seed}")
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            # Initialize benchmark
            mat_bench = MultiObjMATBench(
                data_name=data_name,
                run_subset_only=args.run_subset_only,
                feature_type=f_model,
                finetuning=False,  # This script is for fixed features
                prompt_type=args.prompt_type,
                randseed=args.seed,
                clustering_type=args.clustering_type,
                feature_reduction=args.feature_reduction,)
   
            if args.algorithm in ['multiobj_lapeft', 'multiobj_llmat']:
                llm_bo = obtain_llm_bo_object(args.algorithm, benchmark='multiobj_mat')
                wandb = wandb_wrapper(args, run=args.seed, data_name=data_name, model_name=f_model)
                print("========== start bo ======")
                final_y, all_metrics = llm_bo(args, mat_bench, wandb)
                print("========== end bo ======")
            else:
                raise (NotImplementedError)


if __name__ == "__main__":
    main()
