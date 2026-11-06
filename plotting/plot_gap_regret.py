#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
Quick script to plot GAP and regret curves from ablation results
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add project root to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plotting.visualize_ablation import plot_gap_and_regret_trajectories

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot GAP and regret trajectories')
    parser.add_argument('results_dir', type=str,
                       help='Directory containing ablation results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for figure (default: results_dir/figures/gap_regret.png)')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        output_dir = os.path.join(args.results_dir, "figures")
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, "gap_regret_trajectories.png")

    print(f"Plotting GAP and regret trajectories from: {args.results_dir}")
    print(f"Output will be saved to: {args.output}")

    plot_gap_and_regret_trajectories(args.results_dir, save_path=args.output)

    print(f"\nDone! Plot saved to: {args.output}")
