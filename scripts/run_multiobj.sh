#!/bin/bash
# Multi-objective experiments (Redox-mer, redox potential + solvation energy).
# Usage:
#   bash scripts/run_multiobj.sh llmat  <seed> <clustering_type>
#   bash scripts/run_multiobj.sh lapeft <seed> <method>
algorithm=$1
seed=$2
arg=$3
KEY="${WANDB_API_KEY:-YOUR_WANDB_API_KEY}"
if [ "$algorithm" = "llmat" ]; then
  python src/main/run_multiobj_benchmark.py --feature_type t5-base-chem --seed "$seed" --wandb "Multiobj" \
      --wandb_key "$KEY" --early_stopping --algorithm multiobj_llmat --benchmark multiobj_mat \
      --clustering_type "$arg" --run_group "llmat_$arg"
else
  python src/main/run_multiobj_benchmark.py --feature_type t5-base-chem --seed "$seed" \
      --wandb_key "$KEY" --early_stopping --algorithm multiobj_lapeft --benchmark multiobj_mat --method "$arg"
fi
