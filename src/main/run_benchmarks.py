#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author:
# Created Time : Sun Jul 21 02:08:54 2024
# File Name: benchmark.py
# Description: 
"""

import numpy as np
from src.utils.utils_templates import load_json
from src.utils.wandb_helpers import wandb_wrapper


def obtain_llm_bo_object(bo_type, benchmark='mat'):
    # if bo_type == 'llambo':
    #     from src.algos.baselines.llambo.bo_llambo import bo_llambo as bo_llm

    if bo_type == 'lapeft':
        from src.algos.baselines.lapeft.bo_lapeft import bo_lapeft as bo_llm
    elif bo_type == 'llmat' or bo_type == 'lfbo' or bo_type == 'bore':
        if benchmark == 'toy':
            from src.algos.llmat.llmat_bo_toy import llmat_bo_toy as bo_llm
        else:
            from src.algos.llmat.llmat_bo import llmat_bo as bo_llm
    elif bo_type == 'decoupled_mcts_lfbo':
        from src.algos.llmat.decoupled_mcts_lfbo_bo import decoupled_mcts_lfbo_bo as bo_llm
    else:
        pass
    return bo_llm


def run_mat(args, prompt_ablation=False, api_ablation=False):
    if api_ablation:
        from benchmarks.MAT.api_material_benchmarks import MATBench
    elif prompt_ablation:
        from benchmarks.MAT.prompt_material_benchmarks import MATBench
    else:
        from benchmarks.MAT.material_benchmarks import MATBench
    list_f_model = args.benchmarks[args.benchmark]['feature_models']
    list_data = args.datasets  # not data id???
    for data_name in list_data:
        for f_model in list_f_model:
            if args.finetuning and f_model in ("fingerprints", "gin", "schnet"):
                continue
            print("feature_model: ", f_model)
            iupac = args.iupac
            if data_name not in ["redox-mer", "solvation"]:
                assert iupac == False
            if api_ablation:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=args.run_subset_only,
                    feature_type=f_model,
                    finetuning=args.finetuning,
                    iupac=iupac,
                    prompt_type=args.prompt_type,
                    randseed=args.seed,
                    clustering_type=args.clustering_type,
                    feature_reduction=args.feature_reduction,
                    api_type=args.api_type,
                )
            elif prompt_ablation:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=args.run_subset_only,
                    feature_type=f_model,
                    finetuning=args.finetuning,
                    iupac=iupac,
                    prompt_type=args.prompt_type,
                    randseed=args.seed,
                    clustering_type=args.clustering_type,
                    feature_reduction=args.feature_reduction,
                    prompt_id=args.prompt_id,
                )
            else:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=args.run_subset_only,
                    feature_type=f_model,
                    finetuning=args.finetuning,
                    iupac=iupac,
                    prompt_type=args.prompt_type,
                    randseed=args.seed,
                    clustering_type=args.clustering_type,
                    feature_reduction=args.feature_reduction,
                )
            # run BO
            #all_final_y = []
            #all_metrics_pd = []

            if args.algorithm in ['llmat', 'lapeft', 'lfbo', 'bore', 'decoupled_mcts_lfbo']:
                llm_bo = obtain_llm_bo_object(args.algorithm, benchmark='mat')
                wandb = wandb_wrapper(args, run=args.seed, data_name=data_name, model_name=f_model)
                final_y, all_metrics = llm_bo(args, mat_bench, wandb)
            else:
                raise (NotImplementedError)
            #all_final_y += [np.array(final_y)]
            #all_metrics_pd += [all_metrics]
            #return all_final_y, all_metrics_pd


def run_toy(args):
    from benchmarks.TOY.toy_benchmarks import ToyBench
    list_f_model = args.benchmarks[args.benchmark]['feature_models']
    list_data = args.datasets

    # Get toy-specific parameters with defaults
    dims = getattr(args, 'toy_dims', 20)
    n_samples = getattr(args, 'toy_n_samples', 1000)

    for data_name in list_data:
        for f_model in list_f_model:
            print("feature_model: ", f_model)
            toy_bench = ToyBench(
                data_name=data_name,
                run_subset_only=args.run_subset_only,
                feature_type=f_model,
                finetuning=args.finetuning,
                iupac=False,  # Not applicable for toy benchmarks
                prompt_type=args.prompt_type,
                randseed=args.seed,
                clustering_type=args.clustering_type,
                feature_reduction=args.feature_reduction,
                dims=dims,
                n_samples=n_samples,
            )
            # run BO

            if args.algorithm in ['llmat', 'lapeft', 'lfbo', 'bore']:
                llm_bo = obtain_llm_bo_object(args.algorithm, benchmark='toy')
                wandb = wandb_wrapper(args, run=args.seed, data_name=data_name, model_name=f_model)
                final_y, all_metrics = llm_bo(args, toy_bench, wandb)
            else:
                raise (NotImplementedError)
