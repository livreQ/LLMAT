#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: analyze_prompt_ablation.py
# Description: Analyze and visualize results from redoxmer prompt ablation study
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class PromptAblationAnalyzer:
    """
    Analyze results from prompt ablation experiments.
    """

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True)

        # Load results summary
        self.results_summary = self._load_results_summary()

        # Prompt variant names
        self.prompt_names = {
            0: "Original",
            1: "Detailed",
            2: "Simplified",
            3: "Quantitative",
            4: "Example-based"
        }

        print(f"Loaded results from: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")

    def _load_results_summary(self):
        """Load results summary JSON file."""
        summary_file = self.results_dir / "results_summary.json"

        if not summary_file.exists():
            raise FileNotFoundError(f"Results summary not found: {summary_file}")

        with open(summary_file, 'r') as f:
            return json.load(f)

    def load_bo_results(self):
        """
        Load BO results for each experiment.
        This reads the trace files from the results directories.
        """
        print("\nLoading BO traces...")

        all_results = []

        for idx in range(len(self.results_summary['status'])):
            if self.results_summary['status'][idx] != 'SUCCESS':
                continue

            prompt_id = self.results_summary['prompt_id'][idx]
            algorithm = self.results_summary['algorithm'][idx]
            feature_model = self.results_summary['feature_model'][idx]
            seed = self.results_summary['seed'][idx]

            # Construct path to results
            # Path structure: results/redox-mer/molformer/{clustering_type}/
            results_path = Path(f"results/redox-mer/{feature_model}/llms/")

            # Look for trace files matching the seed and algorithm
            trace_pattern = f"trace_best_y_*_{algorithm}_*_{seed}.npy"

            trace_files = list(results_path.glob(trace_pattern))

            if trace_files:
                # Load the first matching trace
                trace_file = trace_files[0]
                try:
                    trace_y = np.load(trace_file)

                    all_results.append({
                        'prompt_id': prompt_id,
                        'prompt_name': self.prompt_names.get(prompt_id, f"Prompt {prompt_id}"),
                        'algorithm': algorithm,
                        'feature_model': feature_model,
                        'seed': seed,
                        'trace_y': trace_y,
                        'final_best': trace_y[-1] if len(trace_y) > 0 else np.nan,
                        'n_trials': len(trace_y)
                    })
                except Exception as e:
                    print(f"Error loading {trace_file}: {e}")
            else:
                print(f"Warning: No trace file found for prompt={prompt_id}, "
                      f"algo={algorithm}, model={feature_model}, seed={seed}")

        print(f"Loaded {len(all_results)} successful experiments")
        return all_results

    def compute_statistics(self, results):
        """Compute summary statistics for each prompt variant."""
        print("\nComputing statistics...")

        stats_by_prompt = {}

        for prompt_id in sorted(set(r['prompt_id'] for r in results)):
            prompt_results = [r for r in results if r['prompt_id'] == prompt_id]

            if not prompt_results:
                continue

            final_bests = [r['final_best'] for r in prompt_results if not np.isnan(r['final_best'])]

            stats_by_prompt[prompt_id] = {
                'prompt_name': self.prompt_names.get(prompt_id, f"Prompt {prompt_id}"),
                'n_experiments': len(prompt_results),
                'mean_final_best': np.mean(final_bests) if final_bests else np.nan,
                'std_final_best': np.std(final_bests) if final_bests else np.nan,
                'median_final_best': np.median(final_bests) if final_bests else np.nan,
                'min_final_best': np.min(final_bests) if final_bests else np.nan,
                'max_final_best': np.max(final_bests) if final_bests else np.nan,
            }

        return stats_by_prompt

    def plot_convergence_curves(self, results):
        """Plot convergence curves for each prompt variant."""
        print("\nPlotting convergence curves...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Group by prompt
        for prompt_id in sorted(set(r['prompt_id'] for r in results)):
            prompt_results = [r for r in results if r['prompt_id'] == prompt_id]

            if not prompt_results:
                continue

            # Get all traces for this prompt
            traces = [r['trace_y'] for r in prompt_results]

            # Ensure all traces have same length (pad if necessary)
            max_len = max(len(t) for t in traces)
            padded_traces = []
            for t in traces:
                if len(t) < max_len:
                    # Pad with last value
                    padded = np.concatenate([t, np.full(max_len - len(t), t[-1])])
                else:
                    padded = t
                padded_traces.append(padded)

            traces_array = np.array(padded_traces)

            # Compute mean and std
            mean_trace = np.mean(traces_array, axis=0)
            std_trace = np.std(traces_array, axis=0)

            prompt_name = self.prompt_names.get(prompt_id, f"Prompt {prompt_id}")

            # Plot
            x = np.arange(max_len)
            ax.plot(x, mean_trace, label=prompt_name, linewidth=2)
            ax.fill_between(x, mean_trace - std_trace, mean_trace + std_trace, alpha=0.2)

        ax.set_xlabel('BO Iteration', fontsize=12)
        ax.set_ylabel('Best Objective Value', fontsize=12)
        ax.set_title('BO Convergence by Prompt Variant', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "convergence_curves.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / "convergence_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved convergence plot to {self.output_dir}")

    def plot_final_performance(self, stats_by_prompt):
        """Plot final performance comparison across prompts."""
        print("\nPlotting final performance...")

        prompt_ids = sorted(stats_by_prompt.keys())
        prompt_names = [stats_by_prompt[pid]['prompt_name'] for pid in prompt_ids]
        means = [stats_by_prompt[pid]['mean_final_best'] for pid in prompt_ids]
        stds = [stats_by_prompt[pid]['std_final_best'] for pid in prompt_ids]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(prompt_ids))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')

        ax.set_xlabel('Prompt Variant', fontsize=12)
        ax.set_ylabel('Final Best Objective Value', fontsize=12)
        ax.set_title('Final BO Performance by Prompt Variant', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std, f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / "final_performance.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / "final_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved performance plot to {self.output_dir}")

    def plot_performance_distribution(self, results):
        """Plot distribution of final performance across prompts."""
        print("\nPlotting performance distributions...")

        # Prepare data
        data_for_plot = []
        for r in results:
            if not np.isnan(r['final_best']):
                data_for_plot.append({
                    'Prompt': r['prompt_name'],
                    'Final Best': r['final_best']
                })

        df = pd.DataFrame(data_for_plot)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Box plot
        sns.boxplot(data=df, x='Prompt', y='Final Best', ax=ax)
        sns.stripplot(data=df, x='Prompt', y='Final Best', color='black', alpha=0.3, ax=ax)

        ax.set_xlabel('Prompt Variant', fontsize=12)
        ax.set_ylabel('Final Best Objective Value', fontsize=12)
        ax.set_title('Distribution of Final Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_distribution.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / "performance_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved distribution plot to {self.output_dir}")

    def generate_report(self, stats_by_prompt):
        """Generate analysis report."""
        print("\nGenerating report...")

        report = f"""
{'='*70}
REDOXMER PROMPT ABLATION STUDY - ANALYSIS REPORT
{'='*70}

Analysis Directory: {self.results_dir}

SUMMARY STATISTICS BY PROMPT:
{'-'*70}
"""

        for prompt_id in sorted(stats_by_prompt.keys()):
            stats = stats_by_prompt[prompt_id]
            report += f"""
Prompt {prompt_id}: {stats['prompt_name']}
  Number of experiments: {stats['n_experiments']}
  Mean final best:       {stats['mean_final_best']:.6f} ± {stats['std_final_best']:.6f}
  Median final best:     {stats['median_final_best']:.6f}
  Min final best:        {stats['min_final_best']:.6f}
  Max final best:        {stats['max_final_best']:.6f}
"""

        # Ranking
        report += f"\nRANKING (by mean final best):\n{'-'*70}\n"

        ranked = sorted(stats_by_prompt.items(),
                       key=lambda x: x[1]['mean_final_best'],
                       reverse=True)  # Higher is better for maximization

        for rank, (prompt_id, stats) in enumerate(ranked, 1):
            report += f"{rank}. {stats['prompt_name']:20s} {stats['mean_final_best']:.6f}\n"

        report += f"\n{'='*70}\n"

        print(report)

        # Save to file
        report_file = self.output_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"Saved report to {report_file}")

        return report

    def save_results_table(self, results):
        """Save detailed results to CSV."""
        print("\nSaving results table...")

        data = []
        for r in results:
            data.append({
                'prompt_id': r['prompt_id'],
                'prompt_name': r['prompt_name'],
                'algorithm': r['algorithm'],
                'feature_model': r['feature_model'],
                'seed': r['seed'],
                'final_best': r['final_best'],
                'n_trials': r['n_trials']
            })

        df = pd.DataFrame(data)
        csv_file = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False)

        print(f"Saved detailed results to {csv_file}")

    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*70)
        print("RUNNING PROMPT ABLATION ANALYSIS")
        print("="*70)

        # Load BO results
        results = self.load_bo_results()

        if not results:
            print("\nNo results found to analyze!")
            return

        # Compute statistics
        stats_by_prompt = self.compute_statistics(results)

        # Generate plots
        self.plot_convergence_curves(results)
        self.plot_final_performance(stats_by_prompt)
        self.plot_performance_distribution(results)

        # Generate report
        self.generate_report(stats_by_prompt)

        # Save detailed results
        self.save_results_table(results)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze redoxmer prompt ablation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to ablation results directory'
    )

    args = parser.parse_args()

    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Run analysis
    analyzer = PromptAblationAnalyzer(args.results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
