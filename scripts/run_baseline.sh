#!/bin/bash
# Run a baseline on the molecular benchmarks.
# Usage:
#   bash scripts/run_baseline.sh lapeft <seed> <method>            # method: GP / Laplace variants, e.g. gp, tpe
#   bash scripts/run_baseline.sh lfbo   <seed> <clustering_type>   # LFBO baseline
#   bash scripts/run_baseline.sh bore   <seed> <clustering_type>   # BORE baseline
#   bash scripts/run_baseline.sh decoupled_mcts_lfbo <seed> <selection_type>
# Add --finetuning (as a trailing argument) for the LoRA-finetuned variants.
algorithm=$1
seed=$2
arg=$3
shift 3
KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"
case "$algorithm" in
  lapeft)
    python src/main/run_all.py --wandb "LLMAT_baselines" --wandb_key "$KEY" --early_stopping \
        --algorithm lapeft --seed "$seed" --method "$arg" "$@" ;;
  lfbo|bore)
    python src/main/run_all.py --wandb "LLMAT_baselines" --wandb_key "$KEY" --early_stopping \
        --algorithm "$algorithm" --clustering_type "$arg" --seed "$seed" --run_group "${algorithm}_$arg" "$@" ;;
  decoupled_mcts_lfbo)
    python src/main/run_all.py --wandb "LLMAT_baselines" --wandb_key "$KEY" --early_stopping \
        --algorithm decoupled_mcts_lfbo --selection_type "$arg" --seed "$seed" \
        --run_group "decoupled_mcts_lfbo_$arg" "$@" ;;
  *) echo "unknown algorithm: $algorithm"; exit 1 ;;
esac
