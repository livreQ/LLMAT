#!/bin/bash
# Run visualization for toy BO experiments
# Uses configuration from data.yaml for toy benchmark parameters

python -m plotting.visualize_toy_bo \
    --config configs/llmat/llmat_kmeans.yaml \
    --data_config configs/data_config/data.yaml \
    --n_iterations 10 \
    --visualize_every 1 \
    --seed 5 \
    --dims 1 \
    --n_samples 1000 \
    --save_dir visualizations/toy_levy_1d


# python -m plotting.visualize_toy_bo \
#     --config configs/llmat/llmat_kmeans.yaml \
#     --data_config configs/data_config/data.yaml \
#     --n_iterations 100 \
#     --visualize_every 10 \
#     --seed 0 \
#     --dims 1 \
#     --n_samples 1000 \
#     --data_name ackley \
#     --save_dir visualizations/toy_ackley_1d

# To run with Ackley function instead:
# python -m plotting.visualize_toy_bo --data_name ackley --save_dir visualizations/toy_ackley_1d

# To run with 10D (as specified in data.yaml):
# python -m plotting.visualize_toy_bo --dims 10 --n_samples 1000
