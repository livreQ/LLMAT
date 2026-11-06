#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.edu
# Created Time : 2025-11-27
# File Name: run_noise_ablation.py
# Description: Run ablation study for noise robustness on toy benchmarks
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import itertools

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Only import what we need to avoid dependency issues
try:
    from src.utils.wandb_helpers import wandb_wrapper
except ImportError:
    # Provide a dummy wandb wrapper if not available
    class wandb_wrapper:
        def __init__(self, *args, **kwargs):
            self.enabled = False
        def log(self, *args, **kwargs):
            pass
        def finish(self):
            pass

from benchmarks.TOY.toy_benchmarks_noisy import NoisyToyBench
from src.algos.llmat.llmat_bo_toy import llmat_bo_toy


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_args_from_config(config, function, noise_type, noise_scale, seed):
    """Create argparse.Namespace from config dict with specific parameters"""

    # Convert config to namespace
    args = argparse.Namespace()

    # Copy all non-ablation keys
    for key, value in config.items():
        if key != 'ablation':
            setattr(args, key, value)

    # Set specific parameters for this run
    args.seed = seed
    args.noise_type = noise_type
    args.noise_scale = noise_scale
    args.noise_seed = seed + 1000  # Different from data seed

    # Set benchmark-specific parameters
    args.benchmark = 'toy'
    args.algorithm = 'llmat'
    args.datasets = [function]
    args.finetuning = False

    # Get ablation config
    ablation = config.get('ablation', {})
    args.toy_dims = ablation.get('dims', [10])[0]
    args.toy_n_samples = ablation.get('n_samples', 1000)

    # Set feature model (simple for toy benchmarks)
    args.benchmarks = {
        'toy': {
            'feature_models': ['x_original']
        }
    }

    # Add wandb-related attributes expected by wandb_wrapper
    args.wandb_key = None
    args.run_name = None
    args.early_stopping = False
    args.wandb = "LLMAT_Noise_Ablation"
    args.run_group = f"noise_{noise_type}_{noise_scale}"

    return args


def run_single_experiment(config, function, noise_type, noise_scale, seed, results_dir):
    """Run a single BO experiment with specified noise configuration"""

    print(f"\n{'='*80}")
    print(f"Running: {function} | Noise: {noise_type}({noise_scale}) | Seed: {seed}")
    print(f"{'='*80}\n")

    # Create args from config
    args = create_args_from_config(config, function, noise_type, noise_scale, seed)

    # Get ablation config for dims and n_samples
    ablation = config.get('ablation', {})
    dims = ablation.get('dims', [10])[0]
    n_samples = ablation.get('n_samples', 1000)

    # Create noisy toy benchmark
    toy_bench = NoisyToyBench(
        data_name=function,
        run_subset_only=args.run_subset_only,
        feature_type='x_original',
        finetuning=False,
        iupac=False,
        prompt_type=args.prompt_type,
        randseed=seed,
        clustering_type=args.clustering_type,
        feature_reduction=args.feature_reduction,
        dims=dims,
        n_samples=n_samples,
        noise_type=noise_type,
        noise_scale=noise_scale,
        noise_seed=args.noise_seed,
    )

    # Setup wandb
    wandb = wandb_wrapper(args, run=seed, data_name=function,
                         model_name=f'x_original_noise-{noise_type}-{noise_scale}')

    # Run BO
    try:
        final_y, all_metrics = llmat_bo_toy(args, toy_bench, wandb)

        # Get clean optimum for true gap calculation
        clean_opt = toy_bench.get_clean_optimum() if hasattr(toy_bench, 'get_clean_optimum') else toy_bench.ground_truth_opt

        # Compute trajectory metrics
        initial_opt_y = final_y[0] if len(final_y) > 0 else 0
        trace_gap = []
        trace_regret = []

        for t, best_y in enumerate(final_y):
            # GAP: normalized improvement
            if clean_opt != initial_opt_y:
                gap = (best_y - initial_opt_y) / (clean_opt - initial_opt_y)
            else:
                gap = 1.0
            trace_gap.append(np.nan_to_num(gap, nan=1.0))

            # Regret: distance to true optimum
            regret = abs(best_y - clean_opt)
            trace_regret.append(regret)

        # Save results
        result = {
            'function': function,
            'noise_type': noise_type,
            'noise_scale': noise_scale,
            'seed': seed,
            'dims': dims,
            'n_samples': n_samples,
            'final_best_y': final_y[-1] if len(final_y) > 0 else None,
            'ground_truth_opt': toy_bench.ground_truth_opt,  # Noisy optimum (what BO sees)
            'clean_opt': clean_opt,  # True clean optimum (for evaluation)
            'final_gap': (final_y[-1] - clean_opt) if len(final_y) > 0 else None,  # Gap to TRUE optimum
            'trace_y': final_y,
            'trace_gap': trace_gap,  # Trajectory of GAP
            'trace_regret': trace_regret,  # Trajectory of regret
            'metrics': all_metrics,
        }

        # Save individual result
        result_file = os.path.join(
            results_dir,
            f"{function}_{noise_type}_{noise_scale}_seed{seed}.pkl"
        )
        pd.to_pickle(result, result_file)
        print(f"\nSaved result to {result_file}")

        return result

    except Exception as e:
        print(f"ERROR in experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ablation_study(config_path, results_dir=None):
    """Run full ablation study across all configurations"""

    # Load config
    config = load_config(config_path)
    ablation = config.get('ablation', {})

    # Create results directory
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/ablation/noise_{timestamp}"

    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    # Save config to results directory
    config_save_path = os.path.join(results_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # Get all configurations
    functions = ablation.get('functions', ['levy', 'ackley'])
    noise_types = ablation.get('noise_types', ['gaussian'])
    noise_scales = ablation.get('noise_scales', [0.0, 0.1, 0.2])
    seeds = ablation.get('seeds', [42, 123, 456])

    # Generate all combinations
    all_configs = list(itertools.product(functions, noise_types, noise_scales, seeds))
    total_runs = len(all_configs)

    print(f"\n{'='*80}")
    print(f"Starting ablation study with {total_runs} total runs")
    print(f"Functions: {functions}")
    print(f"Noise types: {noise_types}")
    print(f"Noise scales: {noise_scales}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}\n")

    # Run all experiments
    all_results = []
    for idx, (function, noise_type, noise_scale, seed) in enumerate(all_configs):
        print(f"\nProgress: {idx+1}/{total_runs}")
        result = run_single_experiment(config, function, noise_type, noise_scale,
                                      seed, results_dir)
        if result is not None:
            all_results.append(result)

    # Save aggregated results
    results_df = pd.DataFrame([{
        'function': r['function'],
        'noise_type': r['noise_type'],
        'noise_scale': r['noise_scale'],
        'seed': r['seed'],
        'dims': r['dims'],
        'final_best_y': r['final_best_y'],
        'ground_truth_opt': r['ground_truth_opt'],
        'final_gap': r['final_gap'],
    } for r in all_results])

    results_csv = os.path.join(results_dir, "all_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved aggregated results to {results_csv}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    summary = results_df.groupby(['function', 'noise_type', 'noise_scale']).agg({
        'final_gap': ['mean', 'std', 'min', 'max'],
        'final_best_y': ['mean', 'std']
    }).round(4)

    print(summary)

    # Save summary
    summary_csv = os.path.join(results_dir, "summary_statistics.csv")
    summary.to_csv(summary_csv)
    print(f"\nSaved summary statistics to {summary_csv}")

    return results_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run noise ablation study for LLMAT')
    parser.add_argument('--config', type=str,
                       default='configs/ablation/noise_ablation.yaml',
                       help='Path to ablation config file')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory to save results (default: auto-generated with timestamp)')
    parser.add_argument('--single', action='store_true',
                       help='Run single experiment (for testing)')
    parser.add_argument('--function', type=str, default='levy',
                       help='Function for single run')
    parser.add_argument('--noise_type', type=str, default='gaussian',
                       help='Noise type for single run')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                       help='Noise scale for single run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for single run')

    args = parser.parse_args()

    if args.single:
        # Run single experiment for testing
        config = load_config(args.config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = args.results_dir or f"results/ablation/noise_test_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        result = run_single_experiment(
            config, args.function, args.noise_type,
            args.noise_scale, args.seed, results_dir
        )
        print(f"\nSingle experiment completed!")
        if result:
            print(f"Final gap: {result['final_gap']:.6f}")
    else:
        # Run full ablation study
        results_df, summary = run_ablation_study(args.config, args.results_dir)
        print(f"\nAblation study completed! Total runs: {len(results_df)}")
