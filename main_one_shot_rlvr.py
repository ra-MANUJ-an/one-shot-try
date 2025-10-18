#!/usr/bin/env python3
"""
main_one_shot_rlvr.py - WITH PROPER MULTI-HOST INITIALIZATION
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import subprocess 
import time
from pathlib import Path

# CRITICAL: Initialize JAX distributed BEFORE importing any JAX-heavy modules
import jax
import jax.numpy as jnp

# Only initialize if not already initialized AND we have multiple hosts
if jax.process_count() > 1 and not hasattr(jax.config, '_distributed_initialized'):
    print("Initializing JAX distributed for multi-host setup...")
    
    try:
        jax.distributed.initialize()
        print("✓ JAX distributed initialized successfully")
    except Exception as e:
        print(f"⚠️  JAX distributed initialization failed: {e}")
        print("   Continuing with single-host mode...")
else:
    print("✓ Using single-host mode or already initialized")

# Check if we need multi-host setup
# NOW we can safely call device_count AFTER initialization
num_devices = jax.device_count()
num_local_devices = jax.local_device_count()

print(f"JAX Initialization:")
print(f"  Total devices: {num_devices}")
print(f"  Local devices: {num_local_devices}")
print(f"  Process index: {jax.process_index()}")
print(f"  Process count: {jax.process_count()}")

# NOW import heavy modules
from one_shot_rlvr_trainer import OneShotRLVRTrainer

def start_tensorboard(logdir, port=6006):
    """Start TensorBoard in background"""
    os.makedirs(logdir, exist_ok=True)
    
    # Only start TensorBoard on the primary host
    if jax.process_index() == 0:
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", logdir,
            "--port", str(port),
            "--bind_all"
        ]
        
        print(f"Starting TensorBoard on primary host...")
        print(f"Available at: http://localhost:{port}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print(f"TensorBoard started (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"Failed to start TensorBoard: {e}")
            return None
    else:
        print(f"Skipping TensorBoard on worker host (process {jax.process_index()})")
        return None

@hydra.main(version_base=None, config_path="", config_name="one_shot_config")
def main(cfg: DictConfig) -> None:
    """Main One-Shot RLVR training function"""
    
    # Only print on primary host to avoid spam
    if jax.process_index() == 0:
        print("Starting One-Shot RLVR Training Pipeline")
        print("=" * 60)
        print("Paper: 'Reinforcement Learning for Reasoning in Large Language Models with One Training Example'")
        print("=" * 60)
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)
    
    # Initialize trainer
    trainer = OneShotRLVRTrainer(cfg)
    
    # Start TensorBoard (only on primary host)
    tb_port = cfg.trainer.get("tensorboard_port", 6006)
    tb_process = start_tensorboard(trainer.metrics_log_dir, port=tb_port)
    
    if tb_process:
        time.sleep(2)
    
    try:
        # Create dataloaders (all hosts need to do this)
        if jax.process_index() == 0:
            print("Creating One-Shot RLVR dataloaders...")
        trainer.create_one_shot_dataloaders()
        
        # Load model (all hosts need to do this)
        if jax.process_index() == 0:
            print("Loading model...")
        components = trainer.load_model(create_policy_copy=True)
        
        if jax.process_index() == 0:
            print("One-Shot RLVR Model Setup:")
            print(f"  Training examples: {trainer.num_shots}")
            print(f"  Model: {trainer.model_name}")
            print(f"  Total devices: {num_devices}")
            print(f"  Mesh shape: {trainer.mesh.shape}")
            print(f"  Policy model: Full model copy (trainable)")
            print(f"  Reference model: Base model (frozen)")
        
        # Run evaluation before training if requested
        if cfg.trainer.get("val_before_train", False):
            if jax.process_index() == 0:
                print("Running initial evaluation...")
            try:
                eval_results = trainer.evaluate(temperature=0.6, top_p=0.95)
                if jax.process_index() == 0:
                    print(f"Initial accuracy: {eval_results[2]:.2f}%")
                    print(f"Correct: {eval_results[0]}/{eval_results[1]}")
            except Exception as e:
                if jax.process_index() == 0:
                    print(f"Initial evaluation failed: {e}")
        
        # Train the model
        if jax.process_index() == 0:
            print("Starting One-Shot RLVR training...")
        training_result = trainer.fit()
        
        if jax.process_index() == 0:
            print("One-Shot RLVR training completed!")
        
        # Run final evaluation (only on primary host)
        if cfg.trainer.get("test_freq", 1) > 0 and jax.process_index() == 0:
            print("Running final evaluation...")
            try:
                eval_results = trainer.evaluate(temperature=0.6, top_p=0.95)
                print(f"Final accuracy: {eval_results[2]:.2f}%")
                print(f"Correct: {eval_results[0]}/{eval_results[1]}")
            except Exception as e:
                print(f"Final evaluation failed: {e}")
    
    except Exception as e:
        if jax.process_index() == 0:
            print(f"One-Shot RLVR pipeline failed: {e}")
            import traceback
            traceback.print_exc()
        return
    
    if jax.process_index() == 0:
        print("One-Shot RLVR pipeline completed successfully!")
        print("Paper: https://arxiv.org/abs/2504.20571")


if __name__ == "__main__":
    main()
