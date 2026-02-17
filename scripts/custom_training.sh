#!/bin/bash

# Training script for Dual-Camera 3D Diffusion Policy
# Usage: bash scripts/custom_training.sh <task_name> <zarr_path> [gpu_id] [seed]
#
# Example:
#   bash scripts/custom_training.sh placing_bell_pepper ../data/placing_bell_pepper_dual_view.zarr 0 42

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash scripts/custom_training.sh <task_name> <zarr_path> [gpu_id] [seed]"
    echo ""
    echo "Arguments:"
    echo "  task_name   Name of your task (e.g. placing_bell_pepper, stacking_cups)"
    echo "  zarr_path   Path to zarr dataset relative to 3D-Diffusion-Policy/ dir"
    echo "  gpu_id      GPU device ID (default: 0)"
    echo "  seed        Random seed (default: 42)"
    echo ""
    echo "Example:"
    echo "  bash scripts/custom_training.sh placing_bell_pepper ../data/placing_bell_pepper_dual_view.zarr 0 42"
    exit 1
fi

TASK_NAME=$1
ZARR_PATH=$2
GPU_ID=${3:-0}
SEED=${4:-42}
CONFIG="train_custom"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Training DP3 Policy"
echo "=========================================="
echo "Task: $TASK_NAME"
echo "Zarr: $ZARR_PATH"
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
    task.name=${TASK_NAME} \
    task.task_name=${TASK_NAME} \
    task.dataset.zarr_path=${ZARR_PATH} \
    exp_name="${TASK_NAME}"

echo "Training completed!"
