#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Run baseline BO algorithms on toy benchmarks for comparison with LLMAT
Supports: GP, Laplace
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse
import itertools
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from benchmarks.TOY.toy_benchmarks_noisy import NoisyToyBench
from src.algos.baselines.toy_baselines import run_baseline_bo


def run_single_baseline(config, baseline_model, function, noise_type, noise_scale, seed, results_dir):
    """
    Run a single baseline BO experiment

    Args:
        config: Configuration dictionary
        baseline_model: 'gp' or 'laplace'
        function: Function name ('levy', 'ackley')
        noise_type: Type of noise
        noise_scale: Scale of noise
        seed: Random seed
        results_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"Running {baseline_model.upper()} on {function} with {noise_type} noise (scale={noise_scale}, seed={seed})")
    print(f"{'='*80}")

    try:
        # Create benchmark
        toy_bench = NoisyToyBench(
            data_name=function,
            run_subset_only=False,
            finetuning=False,
            iupac=False,
            prompt_type=None,
            randseed=seed,
            feature_type='x_original',
            clustering_type='kmeans',
            feature_reduction=None,
            dims=config['toy']['dims'],
            n_samples=config['toy']['n_samples'],
            noise_type=noise_type,
            noise_scale=noise_scale,
            noise_seed=seed
        )

        # Prepare targets for maximization
        toy_bench.prepare_targets_for_maximization()

        # Run baseline BO
        device = config['training'].get('device', 'cpu')
        n_init = config['toy'].get('n_init', 40)
        n_iters = config['toy'].get('n_iters', 100)
        acq_func = config['baselines'].get('acq_func', 'ei')

        trace_best_y, trace_y_his = run_baseline_bo(
            baseline_model=baseline_model,
            toy_bench=toy_bench,
            n_init=n_init,
            n_iters=n_iters,
            acq_func=acq_func,
            device=device,
            seed=seed
        )

        # Create results directory
        result_subdir = os.path.join(
            results_dir,
            f"{function}",
            f"noise-{noise_type}-{noise_scale}",
            baseline_model
        )
        os.makedirs(result_subdir, exist_ok=True)

        # Save traces
        suffix = f"{seed}"
        np.save(f"{result_subdir}/trace_best_y_{suffix}.npy", trace_best_y)
        np.save(f"{result_subdir}/trace_y_his_{suffix}.npy", trace_y_his)

        # Compute metrics
        clean_opt = toy_bench.get_clean_optimum() if hasattr(toy_bench, 'get_clean_optimum') else toy_bench.ground_truth_opt
        initial_opt_y = trace_best_y[0]
        final_best_y = trace_best_y[-1]

        # Compute final GAP
        if clean_opt != initial_opt_y:
            final_gap = (initial_opt_y - final_best_y) / (initial_opt_y - clean_opt)
        else:
            final_gap = 1.0
        final_gap = np.nan_to_num(final_gap, nan=1.0)

        # Compute final regret
        final_regret = abs(final_best_y - clean_opt)

        print(f"Initial best: {initial_opt_y:.4f}")
        print(f"Final best: {final_best_y:.4f}")
        print(f"Clean optimum: {clean_opt:.4f}")
        print(f"Final GAP: {final_gap:.4f}")
        print(f"Final Regret: {final_regret:.4f}")

        return {
            'baseline_model': baseline_model,
            'function': function,
            'noise_type': noise_type,
            'noise_scale': noise_scale,
            'seed': seed,
            'dims': toy_bench.dims,
            'final_best_y': final_best_y,
            'clean_opt': clean_opt,
            'final_gap': final_gap,
            'final_regret': final_regret
        }

    except Exception as e:
        print(f"Error in experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Run baseline BO on toy benchmarks')
    parser.add_argument('--config', type=str, default='configs/ablation/noise_ablation.yaml',
                       help='Path to config file')
    parser.add_argument('--baseline', type=str, default='gp',
                       choices=['gp', 'laplace', 'all'],
                       help='Which baseline to run (gp, laplace, or all)')
    parser.add_argument('--results_dir', type=str, default='results/baselines_toy',
                       help='Directory to save results')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results will be saved to: {args.results_dir}")

    # Save config
    config_save_path = os.path.join(args.results_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # Get configurations
    ablation = config.get('ablation', {})
    functions = ablation.get('functions', ['levy'])

    # For baselines, only use clean data (no noise)
    noise_types = ['gaussian']
    noise_scales = [0.0]  # Only clean data, no noise
    seeds = ablation.get('seeds', [42, 123, 456])

    print("NOTE: Baselines will only run on clean data (noise_scale=0.0)")

    # Determine which baselines to run
    if args.baseline == 'all':
        baselines = ['gp', 'laplace']
    else:
        baselines = [args.baseline]

    # Generate all combinations
    all_configs = list(itertools.product(baselines, functions, noise_types, noise_scales, seeds))
    total_runs = len(all_configs)

    print(f"\n{'='*80}")
    print(f"Starting baseline BO comparison with {total_runs} total runs")
    print(f"Baselines: {baselines}")
    print(f"Functions: {functions}")
    print(f"Noise types: {noise_types}")
    print(f"Noise scales: {noise_scales}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}\n")

    # Run all experiments
    all_results = []
    for idx, (baseline_model, function, noise_type, noise_scale, seed) in enumerate(all_configs):
        print(f"\nProgress: {idx+1}/{total_runs}")
        result = run_single_baseline(
            config, baseline_model, function, noise_type, noise_scale, seed, args.results_dir
        )
        if result is not None:
            all_results.append(result)

    # Save aggregated results
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(args.results_dir, "all_baseline_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved aggregated results to {results_csv}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"{'='*80}")
    for baseline in baselines:
        baseline_results = results_df[results_df['baseline_model'] == baseline]
        if len(baseline_results) > 0:
            print(f"\n{baseline.upper()}:")
            print(f"  Mean GAP: {baseline_results['final_gap'].mean():.4f} ± {baseline_results['final_gap'].std():.4f}")
            print(f"  Mean Regret: {baseline_results['final_regret'].mean():.4f} ± {baseline_results['final_regret'].std():.4f}")


if __name__ == "__main__":
    main()
