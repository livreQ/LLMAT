#!/bin/bash
# Prompt-variation ablation for LLM clustering on Redox-mer.
# Usage: bash scripts/run_prompt_ablation.sh <seed> <prompt_id>
seed=$1
prompt_id=$2
python src/main/run_redoxmer_prompt_ablation.py --wandb "LLMAT_prompt_ablation" \
    --wandb_key "${WANDB_API_KEY:-YOUR_WANDB_API_KEY}" --early_stopping --algorithm llmat \
    --clustering_type "llms" --seed "$seed" --run_group llmat_llms --prompt_id "$prompt_id"
