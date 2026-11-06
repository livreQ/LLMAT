#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : 2025
# File Name: generate_prompt_datasets.py
# Description: Generate LLM-clustered datasets for all prompt variants
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.MAT.prompt_material_benchmarks import MATBench


def main():
    """Generate datasets for all prompt variants."""

    data_name = "redox-mer"
    f_models = ["molformer"]  # Can add more: ["t5-base-chem", "gpt2-medium", ...]
    prompt_ids = range(5)  # 0-4

    print("="*70)
    print("Generating LLM-Clustered Datasets for Prompt Ablation")
    print("="*70)
    print(f"Dataset: {data_name}")
    print(f"Feature models: {f_models}")
    print(f"Prompt IDs: {list(prompt_ids)}")
    print("="*70)

    for prompt_id in prompt_ids:
        print(f"\n{'='*70}")
        print(f"Processing Prompt ID: {prompt_id}")
        print(f"{'='*70}\n")

        for f_model in f_models:
            print(f"Feature model: {f_model}")

            try:
                mat_bench = MATBench(
                    data_name=data_name,
                    run_subset_only=False,
                    feature_type=f_model,
                    finetuning=True,
                    iupac=False,
                    prompt_type="just-smiles",
                    randseed=666,
                    clustering_type="llms",
                    feature_reduction="average",
                    prompt_id=prompt_id,
                )

                print(f"✓ Successfully generated dataset for prompt_id={prompt_id}, model={f_model}")
                print(f"  Dataset saved to: benchmarks/MAT/data/dataset/{data_name}/{f_model}/llms/dataset{prompt_id}.pkl")

            except Exception as e:
                print(f"✗ Error generating dataset for prompt_id={prompt_id}, model={f_model}")
                print(f"  Error: {str(e)}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*70)
    print("Dataset Generation Complete!")
    print("="*70)
    print("\nYou can now run BO experiments with:")
    print("  bash scripts/run_prompt_ablation.sh <seed> <prompt_id>")


if __name__ == "__main__":
    main()
