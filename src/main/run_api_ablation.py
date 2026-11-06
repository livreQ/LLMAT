#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: run_api_ablation.py
# Description: Script to run API ablation (GPT vs Gemini vs Qwen 3) for parallel execution
"""

from src.main.args import parse_args
from src.main.run_benchmarks import run_mat


def main():
    """
    Run experiment with specified api_type for LLM clustering ablation.
    This is designed to be called multiple times in parallel on a cluster,
    once per API type (gpt, gemini, qwen3).

    Usage:
        python -m src.main.run_api_ablation --api_type gemini --clustering_type llms --feature_type molformer
        python -m src.main.run_api_ablation --api_type qwen3 --clustering_type llms --feature_type molformer
        python -m src.main.run_api_ablation --api_type gpt --clustering_type llms --feature_type molformer
    """

    args = parse_args()

    args.benchmark = 'mat'

    print(f"\n{'='*70}")
    print(f"Running API Ablation")
    print(f"API type: {args.api_type}")
    print(f"Datasets: {args.datasets}")
    print(f"Clustering type: {args.clustering_type}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Feature type: {args.feature_type}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Run the experiment with api_ablation=True
    run_mat(args, api_ablation=True)

    print(f"\n{'='*70}")
    print(f"Completed API Ablation: {args.api_type}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
