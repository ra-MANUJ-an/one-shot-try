#!/bin/bash

# One-Shot RLVR Training Script
# Based on "Reinforcement Learning for Reasoning in Large Language Models with One Training Example"

echo "Starting One-Shot RLVR Training..."
echo "Paper: https://arxiv.org/abs/2504.20571"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your setup

# Training with π_1 (1-shot RLVR)
python3 -m main_one_shot_rlvr \
  data.num_shots=1 \
  data.train_dataset=one_shot_rlvr \
  data.train_split=pi1 \
  data.test_dataset=math500 \
  data.test_split=test \
  data.batch_size=128 \
  data.max_prompt_length=1024 \
  model.name='gemma2-2b-it' \
  model.mesh_config='[[1, 4], ["fsdp", "tp"]]' \
  algorithm.temperature=0.6 \
  algorithm.top_p=0.95 \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.num_batches=64 \
  trainer.checkpoint_dir=./checkpoints/one_shot_rlvr/pi1/ \
  trainer.intermediate_ckpt_dir=./intermediate_ckpt/ \
  trainer.metrics_log_dir=./logs/tensorboard/one_shot_rlvr/pi1/ \
  experiment.name="one_shot_rlvr_pi1_gemma2_2b"

echo "One-Shot RLVR training completed!"