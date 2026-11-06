#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Wed Jun  4 18:11:35 2025
# File Name: wandb_helpers.py
# Description:
"""
import numpy as np
import os


def is_connected(host='http://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host)  #Python 3.x
        return True
    except:
        return False


def wandb_wrapper(args, wandb=None, run=None, data_name=None, model_name=None):
    if wandb is None:
        if not is_connected():
            print('no internet connection. Going in dry')
            os.environ['WANDB_MODE'] = 'dryrun'
        import wandb
        if args.wandb_key is not None:
            wandb.login(key=args.wandb_key)

    if args.run_name is None:
        prefix = data_name + "_" + model_name + "_finetuning_" + str(int(args.finetuning))\
            + "_ninits_" + str(args.n_init_data) + "_ynormalize_" + str(int(args.normalize_y))\
            + "_earlystop_" + str(int(args.early_stopping)) + "_seed_" + str(args.seed)
        if args.algorithm == 'lapeft':
            if args.finetuning:
                args.run_name = prefix + "_lr_" + str(args.finetuning_args["lora"]["lr"]) + "_laplacetype_" + args.laplace_type
            else:
                args.run_name = prefix + "_method_" + args.fix_args["method"] + "_acq_" + args.acqf
                if "laplace" == args.fix_args["method"]:
                    args.run_name = args.run_name + "_laplacetype_" + args.laplace_type
        elif args.algorithm == 'llmat':
            args.run_name = prefix + "_clustering_" + args.clustering_type + "_acq_" + args.acqf + "_lmbda_" + str(args.lmbda)\
                + "_eta_" + str(args.eta) + "_treedepth_" + str(args.tree_depth) + "_pval_" + str(args.p_val) + "_gamma_" + str(args.gamma)
            if args.finetuning:
                args.run_name += "_headepochs_" + str(args.finetuning_args["head"]["n_epochs"]) + "_headlr_" + str(args.finetuning_args["head"]["lr"])\
                    + "_loraepochs_" + str(args.finetuning_args["lora"]["n_epochs"]) + "_loralr_" + str(args.finetuning_args["lora"]["lr"])
            else:
                args.run_name += "_headepochs_" + str(args.fix_args["head"]["n_epochs"]) + "_headlr_" + str(args.fix_args["head"]["lr"])
        else:
            args.run_name = prefix

    wandb.init(project=args.wandb, name=args.run_name, group=args.run_group, tags=[data_name, model_name], reinit=True)
    wandb.config.update(args)
    return wandb
