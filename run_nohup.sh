#!/bin/bash

# Initialize Conda hook
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate Training

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training script in the background
nohup python -u train_math_on_policy_distill.py \
    --k_samples 16 \
    --resume_from_checkpoint runs/qwen2.5_gsm8k_distill/ckpt_interrupted_40 \
    > nohup.out 2>&1 &

# Get the PID of the last background process
PID=$!

echo "Training started with PID: $PID"
echo "Logs are being written to nohup.out"
echo "To monitor the logs, run: tail -f nohup.out"
