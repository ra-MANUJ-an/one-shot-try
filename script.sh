#!/bin/bash

echo "Starting One-Shot RLVR Training..."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Get absolute path
WORK_DIR=$(pwd)

python3 -m main_one_shot_rlvr \
  data.num_shots=1 \
  data.train_dataset=one_shot_rlvr \
  data.train_split=pi1 \
  data.test_dataset=math500 \
  data.test_split=test \
  data.batch_size=8 \
  data.max_prompt_length=384 \
  model.name='qwen3-0.6b' \
  model.mesh_config='[[1, 4], ["fsdp", "tp"]]' \
  model.use_gradient_checkpointing=true \
  algorithm.total_generation_steps=128 \
  algorithm.num_generations=2 \
  algorithm.temperature=0.6 \
  algorithm.top_p=0.95 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.num_batches=64 \
  trainer.gradient_accumulation_steps=16 \
  trainer.checkpoint_dir="${WORK_DIR}/checkpoints/one_shot_rlvr/pi1/" \
  trainer.intermediate_ckpt_dir="${WORK_DIR}/intermediate_ckpt/" \
  trainer.metrics_log_dir="${WORK_DIR}/logs/tensorboard/one_shot_rlvr/pi1/" \
  experiment.name="one_shot_rlvr_pi1_qwen3-0.6b"

echo "One-Shot RLVR training completed!"
