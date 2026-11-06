#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: prompt_cluster_ablation.py
# Description: Ablation study on LLM clustering prompts for redox-mer dataset
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prompt_cluster import PromptCluster


class PromptClusterAblation:
    """
    Ablation study for different LLM clustering prompts.
    Tests sensitivity of clustering to prompt variations.
    """

    def __init__(self, data_name="redox-mer", molecule_col="SMILES", obj_cols=["Ered"]):
        self.data_name = data_name
        self.molecule_col = molecule_col
        self.obj_cols = obj_cols

        # Define alternative prompts for ablation study
        self.prompts = {
            "llms_prompt0": self._get_original_prompt(),
            "llms_prompt1": self._get_variant1_detailed(),
            "llms_prompt2": self._get_variant2_simplified(),
            "llms_prompt3": self._get_variant3_quantitative(),
            # "variant4_examples": self._get_variant4_examples(),
        }

    def _get_original_prompt(self):
        """Original prompt from prompt_cluster_multiobj.py"""
        return """You are a chemist, please use the molecules provided, group them into five clusters based on their redox potential (extremely low : 0, low: 1, medium: 2, high: 3, extremely high :4). Analyze the following features for each molecule: Number and type of electron-withdrawing groups (e.g., CF₃, NO₂, CN, halogens). Number and type of electron-donating groups (e.g., alkyl, methoxy, hydroxyl). Positioning of substituents on aromatic rings (meta, para, ortho). Presence of sulfur-containing functional groups (e.g., SCF₃, S=O). Degree of molecular polarity (based on F, O, or N atoms). Use these criteria to evaluate the redox potential qualitatively and cluster the molecules accordingly."""

    def _get_variant1_detailed(self):
        """Variant 1: More detailed chemical reasoning"""
        return """Act as a chemist. Using the given molecules, assign each one to a redox-potential cluster: 0 = extremely low, 1 = low, 2 = medium, 3 = high, 4 = extremely high. Evaluate each molecule based on: 
    - Electron-withdrawing substituents (CF₃, NO₂, CN, halogens, etc.)
    - Electron-donating substituents (alkyl, OMe, OH, etc.)
    - Substituent positions on aromatic systems (ortho/meta/para)
    - Sulfur-containing functional groups (SCF₃, sulfoxides, etc.)
    - Overall polarity from heteroatoms (F/O/N) 
    Use these qualitative features to estimate redox strength and cluster the molecules."""

    def _get_variant2_simplified(self):
        """Variant 2: Simplified, more direct instructions"""
        return """You are an expert computational chemist. For each provided molecule, perform the following steps: Identify electron-withdrawing and electron-donating groups. Determine substituent positions on aromatic cores. Note any sulfur-based functionalities (e.g., SCF₃, S=O). Assess molecular polarity based on heteroatom composition. Infer the molecule’s qualitative redox potential from these structural features.Then group all molecules into five redox-potential clusters (0–4) from extremely low to extremely high."""

    def _get_variant3_quantitative(self):
        """Variant 3: More quantitative structural guidelines"""
        return """As a chemist specializing in structure–property relationships, analyze each molecule and justify its qualitative redox potential. Consider:
    - Strength and number of EWG vs EDG
    - Aromatic substitution pattern and resonance effects
    - Sulfur-functional group contributions
    - Polarity arising from heteroatoms
    Based on the identified features, explain your reasoning and categorize each molecule into one of five redox classes (0: extremely low → 4: extremely high)."""

    def _get_variant4_examples(self):
        """Variant 4: Include example-based guidance"""
        return """You are a redox-chemistry expert. Examine the structures provided and discuss how their functional groups, electronic effects (EWG/EDG), ring substitution patterns, sulfur-containing moieties, and heteroatom-driven polarity influence their electron-donating or electron-accepting tendencies. Using these qualitative insights, assign each molecule to one of five redox-potential clusters (0–4)."""

    def cluster_with_prompt(self, dataset, prompt_key, api_key, prompt_id=0):
        """
        Perform clustering with a specific prompt variant.

        Args:
            dataset: DataFrame with molecules
            prompt_key: Key for prompt variant
            api_key: OpenAI API key

        Returns:
            Dataset with cluster labels
        """
        print(f"\n{'='*60}")
        print(f"Clustering with prompt: {prompt_key}")
        print(f"{'='*60}\n")

        # Create custom clusterer with specific prompt
        clusterer = PromptCluster(data_name=self.data_name, molecule_col=self.molecule_col)

        dataset_clustered = clusterer.gpt_clustering(dataset.copy(), prompt_id=prompt_id)

        # Rename cluster column to include prompt variant
        cluster_col = f"llm_cluster_{prompt_key}"
        dataset_clustered.rename(columns={"llm_cluster": cluster_col}, inplace=True)

        return dataset_clustered

    def run_ablation_study(self, dataset, api_key, output_dir="benchmarks/MAT/data/ablation_results/"):
        """
        Run ablation study across all prompt variants.

        Args:
            dataset: DataFrame with molecules and objectives
            api_key: OpenAI API key
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Cluster with each prompt variant
        results = {}
        all_clusters_df = dataset.copy()
        prompt_id = 0
        for prompt_key in self.prompts.keys():
            print(f"\n{'='*70}")
            print(f"Processing prompt variant: {prompt_key}")
            print(f"{'='*70}")

            clustered_data = self.cluster_with_prompt(dataset, prompt_key, api_key, prompt_id=prompt_id)
            cluster_col = f"llm_cluster_{prompt_key}"

            # Merge cluster labels
            all_clusters_df[cluster_col] = clustered_data[cluster_col]

            # Store results
            results[prompt_key] = {
                'cluster_col': cluster_col,
                'distribution': clustered_data[cluster_col].value_counts().to_dict(),
            }
            prompt_id += 1

        # Save all cluster labels
        all_clusters_df.to_csv(output_dir + "all_cluster_labels.csv", index=False)
        print(f"\nSaved all cluster labels to: {output_dir}all_cluster_labels.csv")

        # Analyze results
        self.analyze_clustering_stability(all_clusters_df, results, output_dir)

        return all_clusters_df, results

    def analyze_clustering_stability(self, df, results, output_dir):
        """
        Analyze stability and consistency of clustering across prompts.

        Args:
            df: DataFrame with all cluster labels
            results: Dictionary with clustering results
            output_dir: Directory to save analysis
        """
        print(f"\n{'='*70}")
        print("CLUSTERING STABILITY ANALYSIS")
        print(f"{'='*70}\n")

        cluster_cols = [results[k]['cluster_col'] for k in results.keys()]

        # 1. Cluster distribution statistics
        print("1. CLUSTER DISTRIBUTION STATISTICS")
        print("-" * 70)

        dist_stats = {}
        for prompt_key in results.keys():
            cluster_col = results[prompt_key]['cluster_col']
            clusters = df[cluster_col].values

            dist_stats[prompt_key] = {
                'mean_cluster': np.mean(clusters),
                'std_cluster': np.std(clusters),
                'median_cluster': np.median(clusters),
                'distribution': df[cluster_col].value_counts().sort_index().to_dict(),
            }

            print(f"\n{prompt_key}:")
            print(f"  Mean cluster: {dist_stats[prompt_key]['mean_cluster']:.3f}")
            print(f"  Std cluster:  {dist_stats[prompt_key]['std_cluster']:.3f}")
            print(f"  Median:       {dist_stats[prompt_key]['median_cluster']:.1f}")
            print(f"  Distribution: {dist_stats[prompt_key]['distribution']}")

        # 2. Objective statistics per cluster
        print(f"\n2. OBJECTIVE STATISTICS PER CLUSTER")
        print("-" * 70)

        obj_stats = {}
        for prompt_key in results.keys():
            cluster_col = results[prompt_key]['cluster_col']
            obj_stats[prompt_key] = {}

            print(f"\n{prompt_key}:")
            for cluster_id in range(5):
                cluster_mask = df[cluster_col] == cluster_id
                if cluster_mask.sum() == 0:
                    continue

                cluster_data = df[cluster_mask]
                obj_stats[prompt_key][cluster_id] = {}

                print(f"\n  Cluster {cluster_id} (n={cluster_mask.sum()}):")
                for obj_col in self.obj_cols:
                    mean_val = cluster_data[obj_col].mean()
                    std_val = cluster_data[obj_col].std()
                    obj_stats[prompt_key][cluster_id][obj_col] = {'mean': mean_val, 'std': std_val}
                    print(f"    {obj_col}: {mean_val:.4f} ± {std_val:.4f}")

        # 3. Label change analysis
        print(f"\n3. CLUSTER LABEL CHANGE ANALYSIS")
        print("-" * 70)

        # Use original as reference
        reference_col = results['llms_prompt0']['cluster_col']
        reference_labels = df[reference_col].values

        change_stats = {}
        for prompt_key in results.keys():
            if prompt_key == 'llms_prompt0':
                continue

            cluster_col = results[prompt_key]['cluster_col']
            variant_labels = df[cluster_col].values

            # Calculate changes
            changed_mask = reference_labels != variant_labels
            n_changed = changed_mask.sum()
            pct_changed = 100 * n_changed / len(df)

            # Average magnitude of change
            label_diff = np.abs(variant_labels - reference_labels)
            avg_change_magnitude = label_diff[changed_mask].mean() if n_changed > 0 else 0

            change_stats[prompt_key] = {
                'n_changed': n_changed,
                'pct_changed': pct_changed,
                'avg_magnitude': avg_change_magnitude,
                'changed_indices': np.where(changed_mask)[0],
            }

            print(f"\n{prompt_key} vs original:")
            print(f"  Molecules with changed labels: {n_changed} / {len(df)} ({pct_changed:.2f}%)")
            print(f"  Average change magnitude: {avg_change_magnitude:.3f} clusters")

            # Show distribution of changes
            change_counts = {}
            for diff in label_diff:
                change_counts[diff] = change_counts.get(diff, 0) + 1
            print(f"  Change distribution: {dict(sorted(change_counts.items()))}")

        # 4. Pairwise agreement matrix
        print(f"\n4. PAIRWISE AGREEMENT MATRIX")
        print("-" * 70)

        agreement_matrix = np.zeros((len(results), len(results)))
        prompt_keys = list(results.keys())

        for i, key1 in enumerate(prompt_keys):
            for j, key2 in enumerate(prompt_keys):
                labels1 = df[results[key1]['cluster_col']].values
                labels2 = df[results[key2]['cluster_col']].values
                agreement = 100 * (labels1 == labels2).sum() / len(df)
                agreement_matrix[i, j] = agreement

        print("\nAgreement (%) between prompts:")
        print("        ", "  ".join([f"{k[:10]:>10s}" for k in prompt_keys]))
        for i, key in enumerate(prompt_keys):
            print(f"{key[:10]:10s}", "  ".join([f"{agreement_matrix[i, j]:10.2f}" for j in range(len(prompt_keys))]))

        # Save statistics
        stats_summary = {
            'distribution_stats': dist_stats,
            'objective_stats': obj_stats,
            'change_stats': change_stats,
            'agreement_matrix': agreement_matrix.tolist(),
            'prompt_keys': prompt_keys,
        }

        import json
        with open(output_dir + "ablation_statistics.json", 'w') as f:
            # json.dump(stats_summary, f, indent=2)
            json.dump(stats_summary, f, indent=2, default=str)

        print(f"\nSaved statistics to: {output_dir}ablation_statistics.json")

        # Generate visualizations
        self.plot_ablation_results(df, results, agreement_matrix, prompt_keys, output_dir)

    def plot_ablation_results(self, df, results, agreement_matrix, prompt_keys, output_dir):
        """Generate visualizations for ablation study."""

        # 1. Cluster distribution comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, prompt_key in enumerate(prompt_keys):
            cluster_col = results[prompt_key]['cluster_col']
            distribution = df[cluster_col].value_counts().sort_index()

            axes[idx].bar(distribution.index, distribution.values, color='steelblue', alpha=0.7)
            axes[idx].set_xlabel('Cluster', fontsize=10)
            axes[idx].set_ylabel('Count', fontsize=10)
            axes[idx].set_title(f'{prompt_key}', fontsize=11, fontweight='bold')
            axes[idx].set_xticks(range(5))
            axes[idx].grid(True, alpha=0.3)

        # # Hide empty subplot
        # if len(prompt_keys) < 6:
        #     axes[-1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir + "cluster_distributions.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir + "cluster_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Agreement heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlGnBu', xticklabels=[k[:15] for k in prompt_keys], yticklabels=[k[:15] for k in prompt_keys], vmin=0, vmax=100,
                    cbar_kws={'label': 'Agreement (%)'})
        plt.title('Pairwise Cluster Label Agreement', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir + "agreement_heatmap.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir + "agreement_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Objective values by cluster for each prompt in one row
        n_prompts = len(prompt_keys)
        n_obj = len(self.obj_cols)
        
        # Create a subplot for each prompt in a single row
        fig, axes = plt.subplots(n_obj, n_prompts, figsize=(5 * n_prompts, 4 * n_obj), squeeze=False, sharey=True)

        for idx, prompt_key in enumerate(prompt_keys):
            cluster_col = results[prompt_key]['cluster_col']

            for obj_idx, obj_col in enumerate(self.obj_cols):
                ax = axes[obj_idx, idx]

                # Box plot of objective values per cluster
                data_to_plot = [df[df[cluster_col] == c][obj_col].values for c in range(5) if (df[cluster_col] == c).sum() > 0]
                labels_to_plot = [str(c) for c in range(5) if (df[cluster_col] == c).sum() > 0]

                if not data_to_plot:
                    ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                else:
                    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')

                if idx == 0:  # First column
                    ax.set_ylabel(obj_col, fontsize=10)
                
                ax.set_xlabel('Cluster', fontsize=10)
                ax.set_title(f'{prompt_key}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir + "objectives_by_cluster.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir + "objectives_by_cluster.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nSaved visualizations to: {output_dir}")


class PromptClusterCustom(PromptCluster):
    """Custom version that allows injecting a custom prompt."""

    def __init__(self, data_name, molecule_col, obj_cols, custom_prompt, api_key):
        super().__init__(data_name, molecule_col, obj_cols)
        self.custom_prompt = custom_prompt
        self.set_api_key(api_key=api_key)

    def get_prefix_prompt(self):
        """Override to return custom prompt."""
        return self.custom_prompt


if __name__ == "__main__":
    print("=" * 70)
    print("LLM Clustering Prompt Ablation Study for Redox-Mer Dataset")
    print("=" * 70)

    # Load dataset
    data_path = "benchmarks/MAT/data/redox_mer.csv"
    dataset = pd.read_csv(data_path)

    print(f"\nLoaded dataset: {len(dataset)} molecules")
    print(f"Objectives: Ered, Gsol")

    # Initialize ablation study
    ablation = PromptClusterAblation(data_name="redox-mer", molecule_col="SMILES", obj_cols=["Ered"])

    # Set API key
    api_key = __import__("os").environ["OPENAI_API_KEY"]

    # Run ablation study
    all_clusters_df, results = ablation.run_ablation_study(dataset=dataset, api_key=api_key, output_dir="benchmarks/MAT/data/ablation_results/")

    print("\n" + "=" * 70)
    print("Ablation study complete!")
    print("=" * 70)
