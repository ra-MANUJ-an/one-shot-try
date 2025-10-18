#!/bin/bash

echo "Starting One-Shot RLVR Training on 32 chips (4 hosts)..."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps"

# Get the worker index from gcloud metadata
WORKER_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number" -H "Metadata-Flavor: Google")
export JAX_PROCESS_INDEX=${WORKER_ID}

echo "Worker ID: ${JAX_PROCESS_INDEX}"

# Run training
python3 -m main_one_shot_rlvr \
  --config-name=one_shot_config_32chip \
  data.num_shots=1 \
  model.mesh_config='[[4, 8], ["fsdp", "tp"]]'

echo "Training completed on worker ${JAX_PROCESS_INDEX}!"
