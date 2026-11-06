#!/bin/bash
# Cache foundation-model features for a dataset.
# Usage: bash scripts/cache_feature.sh <dataset> [feature_type, default llama-2-7b]
# Models are downloaded from Hugging Face; for gated models (e.g. Llama-2) run `huggingface-cli login`
# or export HUGGING_FACE_HUB_TOKEN=YOUR_HF_TOKEN beforehand.
data=$1
feature_type=${2:-llama-2-7b}
python benchmarks/MAT/cache_features.py --feature_type "$feature_type" --problem "$data"
