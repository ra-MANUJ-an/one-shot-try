#!/bin/bash

echo "Starting One-Shot RLVR Training with Qwen2.5-1.5B..."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

WORK_DIR=$(pwd)

python3 -m main_one_shot_rlvr \
  data.num_shots=1 \
  data.train_dataset=one_shot_rlvr \
  data.train_split=pi1 \
  data.test_dataset=math500 \
  data.test_split=test \
  data.batch_size=2 \
  data.max_prompt_length=1024 \
  model.name='qwen2.5-1.5b' \
  model.mesh_config='[[1, 4], ["fsdp", "tp"]]' \
  model.use_gradient_checkpointing=true \
  algorithm.total_generation_steps=1024 \
  algorithm.num_generations=4 \
  algorithm.temperature=0.01 \
  algorithm.top_p=1.0 \
  algorithm.top_k=null \
  trainer.total_epochs=25 \
  trainer.val_before_train=false \
  trainer.num_batches=32 \
  trainer.gradient_accumulation_steps=8 \
  trainer.checkpoint_dir="${WORK_DIR}/checkpoints/one_shot_rlvr/qwen2.5-1.5b/" \
  trainer.intermediate_ckpt_dir="${WORK_DIR}/intermediate_ckpt/" \
  trainer.metrics_log_dir="${WORK_DIR}/logs/tensorboard/one_shot_rlvr/qwen2.5-1.5b/" \
  experiment.name="one_shot_rlvr_qwen2.5-1.5b"

echo "One-Shot RLVR training completed!"
