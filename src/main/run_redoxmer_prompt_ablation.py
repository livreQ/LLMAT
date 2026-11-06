#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: run_redoxmer_prompt_ablation.py
# Description: Simple script to run redoxmer with specific prompt_id for parallel execution
"""

from src.main.args import parse_args
from src.main.run_benchmarks import run_mat


def main():
    """
    Run redoxmer experiment with specified prompt_id.
    This is designed to be called multiple times in parallel on a cluster.
    """

    # Parse arguments (now includes --prompt_id)
    args = parse_args()

    # Override dataset to redox-mer
    args.datasets = ['redox-mer']
    args.benchmark = 'mat'

    print(f"\n{'='*70}")
    print(f"Running Redoxmer with Prompt ID: {args.prompt_id}")
    print(f"Clustering type: {args.clustering_type}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Feature type: {args.feature_type}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Run the experiment
    run_mat(args, prompt_ablation=True)

    print(f"\n{'='*70}")
    print(f"Completed Prompt ID: {args.prompt_id}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
