#!/bin/bash
# Multi-GPU inference script using torchrun
# Usage: bash run_multi_gpu.sh [NUM_GPUS]

# Set the number of GPUs (default: 5, or use command line argument)
NUM_GPUS=${1:-5}

# Activate conda environment
source /home/group/cad_codebased/cad_llm3/bin/activate

# Set CUDA visible devices (optional, comment out to use all GPUs)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Run with torchrun
echo "Starting multi-GPU inference with $NUM_GPUS GPUs..."
echo "=========================================="

torchrun --nproc_per_node=$NUM_GPUS \
    /home/group/cad_codebased/eval_ckpt/generate_step_multi_gpu.py

echo "=========================================="
echo "Multi-GPU inference completed!"

