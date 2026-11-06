#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Wed Jul 10 16:50:33 2024
# File Name: args.py
# Description:
"""

import argparse
from typing import List, Optional, Union
import yaml
from pdb import set_trace
import doctest
from ConfigSpace import hyperparameters as CSH


def parse_args():
    """
    configurations for datasets, algorithms, etc...
    """
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=help_formatter)
    ############################ general args ###############################
    group = parser.add_argument_group("General Settings")
    benchmarks = ["mat", "multiobj_mat", "toy"]
    group.add_argument('--benchmark', choices=benchmarks, default='mat', help='Name of the benchmarks.')
    group.add_argument('--datasets', type=str, default=None, help='specify list of datasets to evaluate')
    group.add_argument('--folder', type=str, default='benchmarks', help='Path to the folder the data is downloaded to.')

    group.add_argument('--output', type=str, default="output", help='Path to the output folder')
    group.add_argument('--seed', type=int, default=100, help='Seed')
    group.add_argument('--n_seeds', type=int, default=1, help='number of repetitions')
    group.add_argument('--n_trials', type=int, default=None, help='number of trials for BO')
    group.add_argument('--n_inits', type=int, default=None, help='number of initial points')
    group.add_argument('--early_stopping', action='store_true', help='Early stop if optimal value is found')
    group.add_argument('--finetuning', action='store_true', help='Finetuning the LLM for BO')

    ############################ wandb configurations #############################
    group = parser.add_argument_group("Wandb Settings")
    group.add_argument('--wandb', type=str, default=None, help='Wandb project name. If None, no wandb logging')
    group.add_argument('--run_group', type=str, default=None, help='Wandb run group.')
    group.add_argument('--run_name', type=str, default=None, help='Wandb run name. If None, name will be random')
    group.add_argument('--wandb_key', type=str, default=None, help='Wandb token key for login. If None, shell login assumed')

    ########################## algorithms configurations ########################
    group = parser.add_argument_group("Algorithms Settings")
    group.add_argument('-a', '--algorithm', type=str, default='lapeft', choices=['llmat', 'lapeft', 'multiobj_llmat', 'multiobj_lapeft', "lfbo", "bore", "decoupled_mcts_lfbo"])

    # ------------------- baselines -----------------
    # lapeft -- finetuning llm for bo
    group.add_argument("--feature_type", default="molformer", \
                       choices=["molformer", "roberta-large",
                                "t5-base", "t5-base-chem",
                                "gpt2-medium", "gpt2-large","llama-2-7b","qwen3-8b",
                                "fingerprints", "gin", "schnet",],)
    group.add_argument(
        "--prompt_type",
        choices=["single-number", "just-smiles", "completion"],
        default="just-smiles",
    )
    group.add_argument("--laplace_type", choices=["last_layer", "all_layer"], default="all_layer")
    # group.add_argument("--acqf", choices=["ei", "ucb", "pi"], default="ei")
    group.add_argument("--alpha", type=float, default=1.0, help="EI weight power")
    # group.add_argument("--finetuning", action='store_true', help="fintuning or use pretained feature")
    group.add_argument("--length_scale", action='store_true', help="length scale for the kernel, if true, use set value")
    group.add_argument("--method", choices=["gp", "random", "laplace", "tpe"], help="fixed feature lapeft approach", default="gp")
    # ------------------------------------------------
    # ------------------ ours ------------------------
    group.add_argument("--clustering_type", choices=["kmeans", "llms"], default="kmeans")
    group.add_argument("--prompt_id", type=int, default=0, help="Prompt ID for LLM clustering ablation (0-4)")
    group.add_argument("--api_type", type=str, choices=["gpt", "gemini", "qwen3"], default="gpt", help="LLM API type for clustering ablation")
    # major hyper-parameters to tune
    group.add_argument("--lmbda", type=float, default=None, help="lambda UCB for llmat")
    group.add_argument("--gamma", type=float, default=None, help="gamma quantile for llmat")
    group.add_argument("--eta", type=float, default=None, help="meta learning rate for llmat")
    group.add_argument("--p_val", type=float, default=None, help="pval for llmat")
    group.add_argument("--tree_depth", type=int, default=None, help="tree depth for llmat")
    group.add_argument("--head_nepoch", type=int, default=None, help="number of epochs for head finetuning")
    group.add_argument("--head_lr", type=float, default=None, help="learning rate for head finetuning")
    group.add_argument("--lora_nepoch", type=int, default=None, help="number of epochs for lora finetuning")
    group.add_argument("--lora_lr", type=float, default=None, help="learning rate for lora finetuning")
    group.add_argument("--threshold", type=float, default=None, help="thredhold for partitioning")
    group.add_argument("--selection_type", choices=["uct", "greedy"], default="uct", help="MCTS node selection: uct (exploration+exploitation) or greedy (exploitation only)")

    args = parser.parse_args()
    # model_args, train_args, prior_args, bo_args = None, None, None, None
    # if args.model_config is None:
    if args.algorithm == 'decoupled_mcts_lfbo':
        algo_config = "configs/llmat/decoupled_mcts_lfbo.yaml"
    elif args.algorithm == 'llmat' or args.algorithm == 'lfbo' or args.algorithm == 'bore' \
        or args.algorithm == 'multiobj_llmat':
        algo_config = f"configs/llmat/llmat_{args.clustering_type}.yaml"
    elif args.algorithm == 'lapeft' and not args.finetuning:
        algo_config = f"configs/{args.algorithm}/{args.algorithm}_{args.method}.yaml"
    else:
        algo_config = f"configs/{args.algorithm}/{args.algorithm}.yaml"

    print("====================load algorithm config file ===============")
    print(algo_config)
    with open(algo_config) as f:
        algo_args = yaml.load(f, Loader=yaml.UnsafeLoader)
        algo_args_dict = vars(algo_args)
        cli_overrides = {
            "lmbda": args.lmbda, "eta": args.eta, "gamma": args.gamma,
            "p_val": args.p_val, "tree_depth": args.tree_depth,
            "selection_type": args.selection_type,
        }
        for param, val in cli_overrides.items():
            if val is not None:
                algo_args_dict.pop(param, None)
        print(algo_args_dict)
        # overwrite the default values with the values from the file.
        args_dict = vars(args)
        args_dict.update(algo_args_dict)

        args = argparse.Namespace(**args_dict)
    # if args.n_inits is not None, set args.n_init_data to its value
    if args.n_inits is not None:
        args.n_init_data = args.n_inits
    if args.n_trials is not None:
        args.exp_len = args.n_trials

    if args.finetuning:
        if args.head_nepoch is not None:
            args.finetuning_args["head"]["n_epochs"] = args.head_nepoch
        if args.head_lr is not None:
            args.finetuning_args["head"]["lr"] = args.head_lr
        if args.lora_nepoch is not None:
            args.finetuning_args["lora"]["n_epochs"] = args.lora_nepoch
        if args.lora_lr is not None:
            args.finetuning_args["lora"]["lr"] = args.lora_lr
    else:
        if args.head_nepoch is not None:
            args.fix_args["head"]["n_epochs"] = args.head_nepoch
        if args.head_lr is not None:
            args.fix_args["head"]["lr"] = args.head_lr
    print("======= Did not use values from the config file ======")
    # ------------------------------------------------

    ################### benchmark configurations ##################
    data_config = f"configs/data_config/data.yaml"
    print("====================load benchmarks config file ===============")
    print(data_config)
    with open(data_config) as f:
        data_args = yaml.load(f, Loader=yaml.UnsafeLoader)
        print(data_args)
        # overwrite the default values with the values from the file.
        args_dict = vars(args)
        args_dict.update(vars(data_args))
        args = argparse.Namespace(**args_dict)
        if args.datasets is None:
            args.datasets = args.benchmarks[args.benchmark]['datasets']
        else:
            args.datasets = args.datasets.split(';')

    return args


if __name__ == "__main__":
    doctest.testmod()
