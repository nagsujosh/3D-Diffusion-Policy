#!/bin/bash

# Training script for Stacking Cups with 3D Diffusion Policy
# Task: Stacking Cups with Franka Robot
# Usage: bash scripts/train_stacking_cups.sh [gpu_id] [seed]

# Default parameters
GPU_ID=${1:-0}
SEED=${2:-42}
CONFIG="train_stacking_cups"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Training Stacking Cups Policy"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

# Change to 3D-Diffusion-Policy directory and run training
cd "$(dirname "$0")/../3D-Diffusion-Policy"

# Run training
python3 train.py \
    --config-name=${CONFIG} \
    training.seed=${SEED} \
    training.device="cuda:0" \
    exp_name="stacking_cups"

echo "Training completed!"
