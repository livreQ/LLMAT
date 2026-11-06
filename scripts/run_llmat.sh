#!/bin/bash
# Run LLMAT on the molecular benchmarks.
# Usage: bash scripts/run_llmat.sh <seed> <clustering_type: kmeans|llms> [extra args, e.g. --finetuning]
# Optional: export WANDB_API_KEY=... (or run `wandb login` once)
seed=$1
clustering_type=$2
shift 2
python src/main/run_all.py --wandb "LLMAT" --wandb_key "${WANDB_API_KEY:-YOUR_WANDB_API_KEY}" --early_stopping \
    --algorithm llmat --clustering_type "$clustering_type" --seed "$seed" --run_group "llmat_$clustering_type" "$@"
