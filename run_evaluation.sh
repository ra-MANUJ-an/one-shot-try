#!/bin/bash

echo "=========================================="
echo "QWEN2.5-MATH-1.5B EVALUATION"
echo "=========================================="

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 evaluate_base_model.py

echo ""
echo "=========================================="
echo "✓ EVALUATION COMPLETE!"
echo "=========================================="