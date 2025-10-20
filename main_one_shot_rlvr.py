#!/usr/bin/env python3
"""
Main script for One-Shot RLVR training.
Based on "Reinforcement Learning for Reasoning in Large Language Models with One Training Example"
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import subprocess 
import time
from pathlib import Path

from one_shot_rlvr_trainer import OneShotRLVRTrainer

def start_tensorboard(logdir, port=6006):
    """Start TensorBoard in background"""
    os.makedirs(logdir, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", logdir,
        "--port", str(port),
        "--bind_all"
    ]
    
    print(f"Starting TensorBoard...")
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
        print("You can manually start it with:")
        print(f"tensorboard --logdir {logdir} --port {port}")
        return None

@hydra.main(version_base=None, config_path="", config_name="one_shot_config")
def main(cfg: DictConfig) -> None:
    """Main One-Shot RLVR training function"""
    
    print("Starting One-Shot RLVR Training Pipeline")
    print("=" * 60)
    print("Paper: 'Reinforcement Learning for Reasoning in Large Language Models with One Training Example'")
    print("=" * 60)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Initialize One-Shot RLVR trainer
    trainer = OneShotRLVRTrainer(cfg)
    
    # Start TensorBoard
    tb_port = cfg.trainer.get("tensorboard_port", 6006)
    tb_process = start_tensorboard(trainer.metrics_log_dir, port=tb_port)
    time.sleep(2) 
    
    try:
        # Create One-Shot RLVR dataloaders
        print("Creating One-Shot RLVR dataloaders...")
        trainer.create_one_shot_dataloaders()
        
        # Load model
        print("Loading model...")
        components = trainer.load_model(create_policy_copy=True)
        
        print("One-Shot RLVR Model Setup:")
        print(f"  Training examples: {trainer.num_shots}")
        print(f"  Model: {trainer.model_name}")
        print(f"  Policy model: Full model copy (trainable)")
        print(f"  Reference model: Base model (frozen)")
        print(f"  Evaluation protocol: Mathematical equivalence checking")
        
        # Run evaluation before training if requested
        if cfg.trainer.get("val_before_train", False):
            print("Running initial evaluation...")
            try:
                eval_results = trainer.evaluate(temperature=0.6, top_p=0.95)  # Paper settings
                print(f"Initial accuracy: {eval_results[2]:.2f}%")
                print(f"Correct: {eval_results[0]}/{eval_results[1]}")
            except Exception as e:
                print(f"Initial evaluation failed: {e}")
        
        # Train the model using One-Shot RLVR
        print("Starting One-Shot RLVR training...")
        training_result = trainer.fit()
        
        print("One-Shot RLVR training completed!")
        
        # Run final evaluation
        if cfg.trainer.get("test_freq", 1) > 0:
            print("Running final evaluation...")
            try:
                eval_results = trainer.evaluate(temperature=0.6, top_p=0.95)  # Paper settings
                print(f"Final accuracy: {eval_results[2]:.2f}%")
                print(f"Correct: {eval_results[0]}/{eval_results[1]}")
                
                print("\nOne-Shot RLVR Results Summary:")
                print(f"  Training examples: {trainer.num_shots}")
                print(f"  Model: {trainer.model_name}")
                print(f"  Final accuracy: {eval_results[2]:.2f}%")
                print(f"  Improvement method: Few-shot reinforcement learning")
                
            except Exception as e:
                print(f"Final evaluation failed: {e}")
    
    except Exception as e:
        print(f"One-Shot RLVR pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("One-Shot RLVR pipeline completed successfully!")
    print("Paper: https://arxiv.org/abs/2504.20571")


if __name__ == "__main__":
    main()
