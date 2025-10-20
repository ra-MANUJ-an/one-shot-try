#!/usr/bin/env python3
"""
One-Shot RLVR Trainer - Implementation of the paper:
"Reinforcement Learning for Reasoning in Large Language Models with One Training Example"
"""

import functools
import gc
import os
from pprint import pprint
import re
import time
import json
from typing import List, Dict, Any, Optional, Tuple

from flax import nnx
import grain
import humanize
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from omegaconf import DictConfig, OmegaConf

from datasets import load_dataset
from transformers import AutoTokenizer
from utils import *
# from tunix.models.gemma import gemma as gemma_lib
# from tunix.models.gemma import params as gemma_params_lib
# from tunix.examples.data import translation_dataset as data_lib
from tunix.models.qwen3 import model as qwen3_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params_lib


import re

def extract_answer_robust(passage: str) -> str:
    if not passage:
        return None
        
    # Pattern 1: Look for \boxed{...} with proper matching braces
    # This handles nested braces like \boxed{\frac{1}{2}}
    stack = []
    i = passage.find('\\boxed')
    if i != -1:
        i += 6  # Skip '\boxed'
        # Skip whitespace
        while i < len(passage) and passage[i].isspace():
            i += 1
        if i < len(passage) and passage[i] == '{':
            i += 1
            start = i
            brace_count = 1
            while i < len(passage) and brace_count > 0:
                if passage[i] == '{':
                    brace_count += 1
                elif passage[i] == '}':
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                answer = passage[start:i-1]
                return answer.strip()
    
    # Pattern 2: Lenient matching - extract up to common terminators
    patterns = [
        r'\\boxed\{([^}]+)\}',  # Standard
        r'boxed\{([^}]+)\}',     # Missing backslash
        r'\\boxed\s*\{(.+?)(?:\.\s|\)\.|\.$)',  # Ends with period
        r'final answer is.*?\\boxed\{([^}]+)',  # "final answer is"
        r'answer is.*?\\boxed\{([^}]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, passage, re.IGNORECASE | re.DOTALL)
        if matches:
            answer = matches[-1].strip()
            # Clean up
            answer = answer.rstrip('.,;:)')
            # Try to fix common LaTeX issues
            if '\\frac' in answer:
                # Count braces - each \frac needs 2 pairs
                open_braces = answer.count('{')
                close_braces = answer.count('}')
                if open_braces > close_braces:
                    answer += '}' * (open_braces - close_braces)
            return answer
    
    # Pattern 3: Super lenient - just find anything after boxed{
    super_lenient = r'boxed\s*\{([^\n]{1,200})'
    matches = re.findall(super_lenient, passage, re.IGNORECASE)
    if matches:
        answer = matches[-1]
        # Find the first reasonable endpoint
        for char in ['.', ')', '\n', 'The ', 'Thus', 'Therefore']:
            if char in answer:
                answer = answer[:answer.index(char)]
                break
        return answer.strip().rstrip('.,;:)')
    
    return None

class OneShotRLVRTrainer:
    """
    Trainer for One-Shot RLVR as described in the paper.
    Key differences from standard RLVR:
    1. Trains on very few examples (1-16 shots)
    2. Uses more sophisticated reward functions
    3. Focuses on mathematical reasoning tasks
    """
    
    def __init__(self, cfg: DictConfig):
        """Initialize One-Shot RLVR trainer with configuration."""
        self.cfg = cfg
        self._setup_config()
        self._setup_model_configs()
        
        # Core components (initialized later)
        self.base_model = None      
        self.policy_model = None
        self.tokenizer = None
        self.sampler = None
        self.mesh = None
        self.model_config = None
        
        # HuggingFace tokenizer for prompt formatting
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            self.model_configs[self.model_name]["huggingface_name"]
        )

    def _setup_config(self):
        """Extract and setup configuration parameters."""
        # Model and data config
        self.model_name = self.cfg.model.name
        self.mesh_config = self.cfg.model.get("mesh_config", [(1, 4), ("fsdp", "tp")])
        
        # One-Shot RLVR specific parameters
        self.num_shots = self.cfg.data.get("num_shots", 1)  # Key parameter from paper
        self.duplicate_factor = self.cfg.data.get("duplicate_factor", 128)  # Duplicate examples to fill batch
        
        # Training data
        self.train_dataset_name = self.cfg.data.train_dataset
        self.train_dataset_split = self.cfg.data.train_split
        self.test_dataset_name = self.cfg.data.test_dataset
        self.test_dataset_split = self.cfg.data.test_split
        
        # Data directories
        self.train_data_dir = self.cfg.data.get("train_data_dir", "./data/train")
        self.test_data_dir = self.cfg.data.get("test_data_dir", "./data/test")
        
        # Training parameters
        self.batch_size = self.cfg.data.batch_size
        self.num_batches = self.cfg.trainer.get("num_batches", 64)
        self.num_test_batches = self.cfg.data.get("num_test_batches", 100)
        self.num_epochs = self.cfg.trainer.get("total_epochs", 1)
        
        # Algorithm parameters
        self.max_prompt_length = self.cfg.data.get("max_prompt_length", 256)
        self.total_generation_steps = self.cfg.algorithm.get("total_generation_steps", 768)
        self.temperature = self.cfg.algorithm.get("temperature", 0.6)
        self.top_p = self.cfg.algorithm.get("top_p", 0.95)  # Paper uses 0.95
        self.top_k = self.cfg.algorithm.get("top_k", 50)
        self.num_generations = self.cfg.algorithm.get("num_generations", 2)
        self.num_iterations = self.cfg.algorithm.get("num_iterations", 1)
        self.beta = self.cfg.algorithm.get("beta", 0.08)
        self.epsilon = self.cfg.algorithm.get("epsilon", 0.2)
        
        # Calculate max steps based on One-Shot RLVR approach
        self.max_steps = int(self.num_batches * self.num_iterations * self.num_epochs)
        
        # Trainer parameters
        self.learning_rate = self.cfg.trainer.get("learning_rate", 3e-6)
        self.b1 = self.cfg.trainer.get("b1", 0.9)
        self.b2 = self.cfg.trainer.get("b2", 0.99)
        self.weight_decay = self.cfg.trainer.get("weight_decay", 0.1)
        self.warmup_steps_ratio = self.cfg.trainer.get("warmup_steps_ratio", 0.1)
        self.max_grad_norm = self.cfg.trainer.get("max_grad_norm", 0.1)
        
        # Checkpointing and logging
        self.checkpoint_dir = self.cfg.trainer.get("checkpoint_dir", "./checkpoints/")
        self.save_interval_steps = self.cfg.trainer.get("save_freq", 20)
        self.max_to_keep = self.cfg.trainer.get("max_to_keep", 4)
        self.metrics_log_dir = self.cfg.trainer.get("metrics_log_dir", "./logs/tensorboard/")
        self.eval_every_n_steps = self.cfg.trainer.get("test_freq", 10)
        self.intermediate_ckpt_dir = self.cfg.trainer.get("intermediate_ckpt_dir", "./intermediate_ckpt/")

    # def _setup_model_configs(self):
    #     """Setup model-specific configurations."""
    #     self.model_configs = {
    #         "gemma2-2b-it": {
    #             "kaggle_path": "google/gemma-2/flax/gemma2-2b-it",
    #             "config_fn": gemma_lib.TransformerConfig.gemma2_2b,
    #             "model_class": gemma_lib.Transformer,
    #             "params_loader": gemma_params_lib.load_and_format_params,
    #             "tokenizer_class": data_lib.GemmaTokenizer,
    #             "tokenizer_path": "./tokenizers/tokenizer_gemma2.model",
    #             "huggingface_name": "google/gemma-2-2b-it"
    #         }
    #     }
    def _setup_model_configs(self):
        """Setup model-specific configurations."""
        self.model_configs = {
            "gemma2-2b-it": {
                "kaggle_path": "google/gemma-2/flax/gemma2-2b-it",
                # "config_fn": gemma_lib.TransformerConfig.gemma2_2b,
                # "model_class": gemma_lib.Transformer,
                # "params_loader": gemma_params_lib.load_and_format_params,
                # "tokenizer_class": data_lib.GemmaTokenizer,
                "tokenizer_path": "./tokenizers/tokenizer_gemma2.model",
                "huggingface_name": "google/gemma-2-2b-it"
            },
            "qwen3-0.6b": {
                "kaggle_path": "qwen-lm/qwen-3/transformers/0.6b",
                "config_fn": qwen3_lib.ModelConfig.qwen3_0_6b,  
                "model_class": qwen3_lib.Qwen3,
                "params_loader": qwen3_params_lib.create_model_from_safe_tensors,
                "tokenizer_class": "transformers",  
                "tokenizer_path": None,
                "huggingface_name": "Qwen/Qwen3-0.6B"
            },
            "qwen2.5-1.5b": {
                "kaggle_path": None,  # Will download from HF
                "config_fn": qwen2_lib.ModelConfig.qwen2_5_1_5b,
                "model_class": qwen2_lib.Qwen2,
                "params_loader": qwen2_params_lib.create_model_from_safe_tensors,
                "tokenizer_class": "transformers",
                "tokenizer_path": None,
                "huggingface_name": "Qwen/Qwen2.5-1.5B-Instruct"
            },

        }
    # def create_one_shot_dataloaders(self):
    #     """
    #     Create dataloaders for One-Shot RLVR.
    #     Key difference: Only use very few examples, duplicate to fill batches.
    #     """
    #     print(f"Creating One-Shot RLVR dataloaders with {self.num_shots} shots...")
        
    #     # Load the specified training examples
    #     if self.train_dataset_name == "one_shot_rlvr":
    #         # Load the specific examples from the paper
    #         train_examples = self._load_one_shot_examples()
    #     else:
    #         # Load from regular dataset but limit to num_shots
    #         train_dataset_raw = self._load_dataset(
    #             self.train_dataset_name, 
    #             self.train_data_dir, 
    #             self.train_dataset_split
    #         )
    #         train_examples = list(train_dataset_raw[:self.num_shots])
        
    #     print(f"Loaded {len(train_examples)} training examples")
    #     print(f"Example training data:")
    #     pprint(train_examples[0] if train_examples else "No examples found")
        
    #     # Duplicate examples to fill training batches (key One-Shot RLVR technique)
    #     duplicated_examples = []
    #     target_size = self.num_batches * self.batch_size
        
    #     for i in range(target_size):
    #         duplicated_examples.append(train_examples[i % len(train_examples)])
        
    #     print(f"Duplicated {len(train_examples)} examples to {len(duplicated_examples)} for training")
        
    #     # Create grain datasets
    #     train_dataset = (
    #         grain.MapDataset.source(duplicated_examples)
    #         .batch(self.batch_size)[:self.num_batches]
    #     )
        
    #     # Test dataset (normal loading)
    #     test_dataset_raw = self._load_dataset(
    #         self.test_dataset_name, 
    #         self.test_data_dir, 
    #         self.test_dataset_split
    #     )
    #     test_dataset = test_dataset_raw.batch(self.batch_size)[:self.num_test_batches]
        
    #     self.train_dataset = train_dataset.repeat(self.num_epochs) 
    #     self.test_dataset = test_dataset
        
    #     print("One-Shot RLVR Datasets Created")
    #     print(f"Train batches: {len(self.train_dataset)}")
    #     print(f"Test batches: {len(self.test_dataset)}")

    def create_one_shot_dataloaders(self):
        """
        Create dataloaders for One-Shot RLVR - CORRECTED VERSION.
        Key: Duplicate the single example duplicate_factor times.
        """
        print(f"Creating One-Shot RLVR dataloaders with {self.num_shots} shots...")
        print(f"Duplicate factor: {self.duplicate_factor}")
        
        # Load the specified training examples
        if self.train_dataset_name == "one_shot_rlvr":
            train_examples = self._load_one_shot_examples()
        else:
            train_dataset_raw = self._load_dataset(
                self.train_dataset_name, 
                self.train_data_dir, 
                self.train_dataset_split
            )
            train_examples = list(train_dataset_raw[:self.num_shots])
        
        print(f"Loaded {len(train_examples)} unique training examples")
        
        duplicated_examples = []
        for example in train_examples:
            for _ in range(self.duplicate_factor):
                duplicated_examples.append(example.copy())  # Use .copy() to avoid reference issues
        
        print(f"✓ Duplicated {len(train_examples)} unique examples × {self.duplicate_factor} = {len(duplicated_examples)} total examples")
        
        # Verify duplication
        assert len(duplicated_examples) == len(train_examples) * self.duplicate_factor, \
            f"Duplication failed! Expected {len(train_examples) * self.duplicate_factor}, got {len(duplicated_examples)}"
        
        # IMPORTANT: With batch_size=128 and 128 total examples, we have exactly 1 batch
        # So num_batches should be 1, not 64!
        expected_batches = len(duplicated_examples) // self.batch_size
        print(f"Total examples: {len(duplicated_examples)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Expected batches per epoch: {expected_batches}")
        
        # Create grain datasets
        train_dataset = (
            grain.MapDataset.source(duplicated_examples)
            .batch(self.batch_size)
        )
        
        # Test dataset (normal loading)
        test_dataset_raw = self._load_dataset(
            self.test_dataset_name, 
            self.test_data_dir, 
            self.test_dataset_split
        )
        test_dataset = test_dataset_raw.batch(self.batch_size)[:self.num_test_batches]
        
        # Paper trains for ~2000 steps, so we repeat the dataset
        self.train_dataset = train_dataset.repeat(self.num_epochs) 
        self.test_dataset = test_dataset
        
        print("✓ One-Shot RLVR Datasets Created")
        print(f"  Train batches per epoch: {expected_batches}")
        print(f"  Total training steps: {expected_batches * self.num_epochs}")
        print(f"  Test batches: {len(self.test_dataset)}")
        
        # Verification: Check first batch
        first_batch = train_dataset[0]
        print(f"\n✓ Verification - First batch:")
        print(f"  Batch size: {len(first_batch['question'])}")
        print(f"  First question (truncated): {first_batch['question'][0][:100]}...")
        print(f"  All questions same? {len(set(first_batch['question'])) == 1}")

    def _load_one_shot_examples(self) -> List[Dict[str, Any]]:
        """
        Load the specific training examples from the One-Shot RLVR paper.
        These are the π_1, π_4, π_16 examples mentioned in the paper.
        """
        # Example π_1 from the paper
        pi_1 = {
            "prompts": format_prompt_with_chat_template(
                self.hf_tokenizer,
                "The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{}."
            ),
            "question": "The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let's think step by step and output the final answer within \\boxed{}.",
            "answer": "12.8"
        }
        
        examples = [pi_1]
        
        # Add more examples based on num_shots
        if self.num_shots > 1:
            # You would load π_4, π_16, etc. from your data files
            # For now, we'll duplicate π_1 until we have the exact examples
            pass
            
        return examples[:self.num_shots]

    def _load_dataset(self, dataset_name: str, data_dir: str = None, split: str = "train") -> grain.MapDataset:
        """Load dataset with One-Shot RLVR formatting."""
        hf_dataset_name, config_name, dataset_config = get_data(dataset_name)
        
        if dataset_config["use_tfds"]:
            # Handle TFDS datasets (like GSM8K)
            if data_dir and not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            data = tfds.data_source(
                hf_dataset_name,
                split=split,
                data_dir=data_dir,
                builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
                download=True,
            )
            
            def process_tfds_item(x):
                question_text = x[dataset_config["question_field"]].decode("utf-8")
                answer_text = x[dataset_config["answer_field"]].decode("utf-8")
                
                if dataset_config["answer_extractor"]:
                    answer = dataset_config["answer_extractor"](answer_text)
                else:
                    answer = answer_text
                
                return {
                    "prompts": format_prompt_with_chat_template(self.hf_tokenizer, question_text),
                    "question": question_text,
                    "answer": answer,
                }
            
            dataset = (
                grain.MapDataset.source(data)
                .shuffle(seed=42)
                .map(process_tfds_item)
            )
        
        else:
            # Handle HuggingFace datasets
            if config_name:
                dataset_hf = load_dataset(hf_dataset_name, config_name, split=split)
            else:
                dataset_hf = load_dataset(hf_dataset_name, split=split)
            
            data_list = list(dataset_hf)
            
            def process_hf_item(item):
                question_text = item[dataset_config["question_field"]]
                answer_text = item[dataset_config["answer_field"]]
                
                if dataset_config["answer_extractor"]:
                    answer = dataset_config["answer_extractor"](answer_text)
                    if answer is None:
                        answer = answer_text
                else:
                    answer = answer_text
                
                return {
                    "prompts": format_prompt_with_chat_template(self.hf_tokenizer, question_text),
                    "question": question_text,
                    "answer": answer,
                }
            
            dataset = (
                grain.MapDataset.source(data_list)
                .shuffle(seed=42)
                .map(process_hf_item)
            )
        
        return dataset

    def load_model(self, create_policy_copy=True, cache_config=None):
        """Load model components for One-Shot RLVR training."""
        print(f"Loading model for One-Shot RLVR: {self.model_name}")
        
        try:
            ckpt_path = self._download_and_prepare_checkpoint()
            self.model_checkpoint_path = ckpt_path

            self._load_base_model(ckpt_path)
            print("Reference model (base model) loaded - stays frozen")

            if create_policy_copy:
                self._create_policy_model()
                print("Policy model (full model copy) created - will be trained")

            self._create_tokenizer()
            self._create_sampler()
            
            return {
                "policy_model": self.policy_model,
                "reference_model": self.base_model,
                "tokenizer": self.tokenizer,
                "sampler": self.sampler,
                "mesh": self.mesh,
                "model_config": self.model_config,
            }
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise e

    def _download_and_prepare_checkpoint(self):
        """Download and prepare model checkpoint."""
        if self.model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        config = self.model_configs[self.model_name]
        
        # For Qwen2.5, download from HuggingFace instead of Kaggle
        if self.model_name.startswith("qwen2.5"):
            from huggingface_hub import snapshot_download
            print(f"Downloading {self.model_name} from HuggingFace...")
            ckpt_path = snapshot_download(repo_id=config["huggingface_name"])
            print(f"Model downloaded to: {ckpt_path}")
            return ckpt_path
        
        # Original Kaggle download logic for other models
        if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
            kagglehub.login()
        
        print(f"Downloading {self.model_name} from Kaggle...")
        kaggle_ckpt_path = kagglehub.model_download(config["kaggle_path"])
        time.sleep(30)

        if self.model_name.startswith("qwen3"):
            return kaggle_ckpt_path
        else:
            # Gemma conversion logic
            params = config["params_loader"](
                os.path.join(kaggle_ckpt_path, self.model_name)
            )
            model = config["model_class"].from_params(params, version=self.model_name.replace("gemma", ""))
            os.makedirs(self.intermediate_ckpt_dir, exist_ok=True)
            ckpt_path = os.path.abspath(os.path.join(self.intermediate_ckpt_dir, "state"))
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            checkpointer = ocp.StandardCheckpointer()
            _, state = nnx.split(model)
            if not os.path.exists(ckpt_path):
                checkpointer.save(ckpt_path, state)
                print("Waiting for checkpoint to save...")
                
            time.sleep(120) 
            del params, model, state
            gc.collect()
            
            return ckpt_path

    # def _download_and_prepare_checkpoint(self):
    #     """Download and prepare model checkpoint."""
    #     if self.model_name not in self.model_configs:
    #         raise ValueError(f"Unsupported model: {self.model_name}")
        
    #     config = self.model_configs[self.model_name]
        
    #     if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    #         kagglehub.login()
        
    #     print(f"Downloading {self.model_name} from Kaggle...")
    #     kaggle_ckpt_path = kagglehub.model_download(config["kaggle_path"])
    #     time.sleep(30)

    #     if self.model_name.startswith("qwen3"):
    #         return kaggle_ckpt_path
    #     else:
    #         params = config["params_loader"](
    #             os.path.join(kaggle_ckpt_path, self.model_name)
    #         )
    #         model = config["model_class"].from_params(params, version=self.model_name.replace("gemma", ""))
    #         os.makedirs(self.intermediate_ckpt_dir, exist_ok=True)
    #         ckpt_path = os.path.abspath(os.path.join(self.intermediate_ckpt_dir, "state"))
    #         os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    #         checkpointer = ocp.StandardCheckpointer()
    #         _, state = nnx.split(model)
    #         if not os.path.exists(ckpt_path):
    #             checkpointer.save(ckpt_path, state)
    #             print("Waiting for checkpoint to save...")
                
    #         time.sleep(120) 
    #         del params, model, state
    #         gc.collect()
            
    #         return ckpt_path

    def _load_base_model(self, ckpt_path):
        """Load the base model."""
        config = self.model_configs[self.model_name]
        self.mesh = jax.make_mesh(*self.mesh_config)
        self.model_config = config["config_fn"]()

        # Both qwen3 and qwen2.5 use the same loading pattern
        if self.model_name.startswith("qwen3") or self.model_name.startswith("qwen2.5"):
            print("\n\n\n\n\n\n\nI'm using QWEN\n\n\n\n\n\n\n")
            self.base_model = config["params_loader"](
                file_dir=ckpt_path, 
                config=self.model_config,
                mesh=self.mesh,
                dtype=jnp.float32
            )
        else:
            # Gemma loading logic
            abs_model = nnx.eval_shape(
                lambda: config["model_class"](self.model_config, rngs=nnx.Rngs(params=0))
            )
            abs_state = nnx.state(abs_model)
            abs_state = jax.tree.map(
                lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
                abs_state,
                nnx.get_named_sharding(abs_state, self.mesh),
            )
            checkpointer = ocp.StandardCheckpointer()
            restored_params = checkpointer.restore(ckpt_path, target=abs_state)
            
            graph_def, _ = nnx.split(abs_model)
            self.base_model = nnx.merge(graph_def, restored_params)

    # def _load_base_model(self, ckpt_path):
    #     """Load the base model."""
    #     config = self.model_configs[self.model_name]
    #     self.mesh = jax.make_mesh(*self.mesh_config)
    #     self.model_config = config["config_fn"]()

    #     if self.model_name.startswith("qwen3"):
    #         self.base_model = config["params_loader"](
    #             file_dir=ckpt_path, 
    #             config=self.model_config,
    #             mesh=self.mesh
    #         )
    #     else:
    #         abs_model = nnx.eval_shape(
    #             lambda: config["model_class"](self.model_config, rngs=nnx.Rngs(params=0))
    #         )
    #         abs_state = nnx.state(abs_model)
    #         abs_state = jax.tree.map(
    #             lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
    #             abs_state,
    #             nnx.get_named_sharding(abs_state, self.mesh),
    #         )
    #         checkpointer = ocp.StandardCheckpointer()
    #         restored_params = checkpointer.restore(ckpt_path, target=abs_state)
            
    #         graph_def, _ = nnx.split(abs_model)
    #         self.base_model = nnx.merge(graph_def, restored_params)

    def _create_policy_model(self):
        """Create policy model (trainable copy)."""
        if self.base_model is None:
            raise ValueError("Base model must be loaded first")
        
        # Both qwen3 and qwen2.5 use the same pattern
        if self.model_name.startswith("qwen3") or self.model_name.startswith("qwen2.5"):
            config = self.model_configs[self.model_name]
            self.policy_model = config["params_loader"](
                file_dir=self.model_checkpoint_path,
                config=self.model_config,
                mesh=self.mesh,
                dtype=jnp.float32
            )
        else:
            # Gemma loading logic
            config = self.model_configs[self.model_name]
            policy_model = config["model_class"](self.model_config, rngs=nnx.Rngs(params=0))
            
            base_state = nnx.state(self.base_model)
            nnx.update(policy_model, base_state)
            self.policy_model = policy_model
            
            with self.mesh:
                state = nnx.state(self.policy_model)
                pspecs = nnx.get_partition_spec(state)
                sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
                nnx.update(self.policy_model, sharded_state)

    # def _create_policy_model(self):
    #     """Create policy model (trainable copy)."""
    #     if self.base_model is None:
    #         raise ValueError("Base model must be loaded first")
        
    #     if self.model_name.startswith("qwen3"):
    #         config = self.model_configs[self.model_name]
    #         self.policy_model = config["params_loader"](
    #             file_dir=self.model_checkpoint_path,
    #             config=self.model_config,
    #             mesh=self.mesh
    #         )
    #     else:
    #         config = self.model_configs[self.model_name]
    #         policy_model = config["model_class"](self.model_config, rngs=nnx.Rngs(params=0))
            
    #         base_state = nnx.state(self.base_model)
    #         nnx.update(policy_model, base_state)
    #         self.policy_model = policy_model
            
    #         with self.mesh:
    #             state = nnx.state(self.policy_model)
    #             pspecs = nnx.get_partition_spec(state)
    #             sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    #             nnx.update(self.policy_model, sharded_state)

    def _create_tokenizer(self):
        """Create tokenizer."""
        config = self.model_configs[self.model_name]
        
        if config["tokenizer_path"] is not None:
            self.tokenizer = config["tokenizer_class"](
                model_path=config["tokenizer_path"]
            )
        else:
            self.tokenizer = self.hf_tokenizer
        
        return self.tokenizer

    def _create_sampler(self, cache_config=None):
        """Create sampler for generation."""
        if cache_config is None:
            cache_config = sampler_lib.CacheConfig(
                cache_size=self.max_prompt_length + self.total_generation_steps + 256,
                num_layers=self.model_config.num_layers,
                num_kv_heads=self.model_config.num_kv_heads,
                head_dim=self.model_config.head_dim,
            )
        
        self.sampler = sampler_lib.Sampler(
            transformer=self.policy_model,
            tokenizer=self.tokenizer,
            cache_config=cache_config,
        )
        return self.sampler

    def fit(self):
        """
        Train the model using One-Shot RLVR with the correct Tunix API.
        """
        print(f"Starting One-Shot RLVR training with {self.num_shots} shots...")
        warmup_steps = int(self.warmup_steps_ratio * self.max_steps)

        print(f"One-Shot RLVR Training Configuration:")
        print(f"  Number of training examples: {self.num_shots}")
        print(f"  Duplication factor: {self.duplicate_factor}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Temperature: {self.temperature} (paper uses 0.6)")
        print(f"  Top-p: {self.top_p} (paper uses 0.95)")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpointing_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.save_interval_steps, 
            max_to_keep=self.max_to_keep
        )
        
        os.makedirs(self.metrics_log_dir, exist_ok=True)
        metrics_logging_options = metrics_logger.MetricsLoggerOptions(
            log_dir=self.metrics_log_dir, 
            flush_every_n_steps=20
        )

        # Setup optimizer
        self.optimizer = optax.adamw(
            learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=self.max_steps,
                end_value=0.0,
            ),
            b1=self.b1,
            b2=self.b2,
            weight_decay=self.weight_decay,
        )

        if self.max_grad_norm is not None:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_norm=self.max_grad_norm),
                self.optimizer,
            )

        # Use the new Tunix GRPO API (like in demo)
        cluster_config = rl_cluster_lib.ClusterConfig(
            role_to_mesh={
                rl_cluster_lib.Role.ACTOR: self.mesh,
                rl_cluster_lib.Role.REFERENCE: self.mesh,
                rl_cluster_lib.Role.ROLLOUT: self.mesh,
            },
            rollout_engine='vanilla',
            offload_to_cpu=False,
            training_config=rl_cluster_lib.RLTrainingConfig(
                actor_optimizer=self.optimizer,
                eval_every_n_steps=self.eval_every_n_steps,
                max_steps=self.max_steps,
                # gradient_accumulation_steps=1,
                # metrics logging
                metrics_logging_options=metrics_logging_options,
                # checkpoint saving
                checkpoint_root_directory=self.checkpoint_dir,
                checkpointing_options=checkpointing_options,
            ),
            rollout_config=base_rollout.RolloutConfig(
                max_tokens_to_generate=self.total_generation_steps,
                max_prompt_length=self.max_prompt_length,
                kv_cache_size=self.max_prompt_length + self.total_generation_steps + 256,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ),
        )

        grpo_config = GrpoConfig(
            num_generations=self.num_generations,
            num_iterations=self.num_iterations,
            beta=self.beta,
            epsilon=self.epsilon,
        )

        # Create RL cluster
        rl_cluster = rl_cluster_lib.RLCluster(
            actor=self.policy_model,
            reference=self.base_model,
            tokenizer=self.tokenizer,
            cluster_config=cluster_config,
        )

        # Use the sophisticated reward function from the paper
        self.grpo_learner = GrpoLearner(
            rl_cluster=rl_cluster,
            reward_fns=[one_shot_rlvr_reward_function],  # Paper's reward function
            grpo_config=grpo_config,
        )

        with self.mesh:
            self.grpo_learner.train(self.train_dataset)

        return self.grpo_learner

    # def evaluate(self, temperature=0.6, top_k=50, top_p=0.95, num_passes=1):
    #     """Evaluate model using One-Shot RLVR evaluation protocol."""
    #     corr = 0
    #     total = 0
        
    #     for batch in tqdm(self.test_dataset):
    #         answers = batch["answer"]
    #         questions = batch["question"]
            
    #         multiple_call_responses = [[] for _ in range(len(questions))]
    #         for p in range(num_passes):
    #             responses = self.generate(
    #                 questions, self.sampler, temperature, top_k, top_p, seed=p
    #             )
    #             for idx, response in enumerate(responses):
    #                 multiple_call_responses[idx].append(response)

    #         for question, multiple_call_response, answer in zip(
    #             questions, multiple_call_responses, answers
    #         ):
    #             corr_per_question = 0
    #             for response in multiple_call_response:
    #                 # Use the sophisticated evaluation from the paper
    #                 score = compute_score_one_shot_rlvr(response, answer)
    #                 if score > 0:
    #                     corr_per_question = 1
    #                     break

    #             if corr_per_question > 0:
    #                 corr += 1

    #             total += 1
    #             if total % 10 == 0:
    #                 print(f"Progress: {corr}/{total} = {corr / total * 100:.2f}%")
        
    #     accuracy = corr / total * 100 if total > 0 else 0
    #     return corr, total, accuracy
    # def evaluate(self, temperature=0.6, top_k=50, top_p=0.95, num_passes=1):
    #     """Evaluate model using One-Shot RLVR evaluation protocol."""
        
    #     # Create a new sampler with appropriate cache size for evaluation
    #     eval_cache_config = sampler_lib.CacheConfig(
    #         cache_size=1024,  # Larger cache for evaluation
    #         num_layers=self.model_config.num_layers,
    #         num_kv_heads=self.model_config.num_kv_heads,
    #         head_dim=self.model_config.head_dim,
    #     )
        
    #     eval_sampler = sampler_lib.Sampler(
    #         transformer=self.policy_model,
    #         tokenizer=self.tokenizer,
    #         cache_config=eval_cache_config,
    #     )
        
    #     corr = 0
    #     total = 0
        
    #     for batch in tqdm(self.test_dataset):
    #         answers = batch["answer"]
    #         questions = batch["question"]
            
    #         multiple_call_responses = [[] for _ in range(len(questions))]
    #         for p in range(num_passes):
    #             # Use eval_sampler instead of self.sampler
    #             responses = self.generate(
    #                 questions, 
    #                 eval_sampler,  # Changed here
    #                 temperature, 
    #                 top_k, 
    #                 top_p, 
    #                 seed=p
    #             )
    #             for idx, response in enumerate(responses):
    #                 multiple_call_responses[idx].append(response)

    #         for question, multiple_call_response, answer in zip(
    #             questions, multiple_call_responses, answers
    #         ):
    #             corr_per_question = 0
    #             for response in multiple_call_response:
    #                 score = compute_score_one_shot_rlvr(response, answer)
    #                 if score > 0:
    #                     corr_per_question = 1
    #                     break

    #             if corr_per_question > 0:
    #                 corr += 1

    #             total += 1
    #             if total % 10 == 0:
    #                 print(f"Progress: {corr}/{total} = {corr / total * 100:.2f}%")
        
    #     accuracy = corr / total * 100 if total > 0 else 0
    #     return corr, total, accuracy

    def evaluate(self, temperature=0.6, top_k=50, top_p=0.95, num_passes=1):
        """Evaluate model using One-Shot RLVR evaluation protocol."""
        
        # Use larger cache for evaluation
        eval_cache_config = sampler_lib.CacheConfig(
            cache_size=2048,  # Even larger cache
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
        )
        
        eval_sampler = sampler_lib.Sampler(
            transformer=self.policy_model,
            tokenizer=self.tokenizer,
            cache_config=eval_cache_config,
        )
        
        corr = 0
        total = 0
        
        for batch in tqdm(self.test_dataset):
            answers = batch["answer"]
            questions = batch["question"]
            
            # PROCESS ONE QUESTION AT A TIME instead of batch
            for i, (question, answer) in enumerate(zip(questions, answers)):
                multiple_responses = []
                for p in range(num_passes):
                    # Generate one question at a time
                    response = self.generate(
                        question,  # Single question, not list
                        eval_sampler,
                        temperature,
                        top_k,
                        top_p,
                        seed=p
                    )
                    multiple_responses.append(response)
                
                # Check if any response is correct
                corr_per_question = 0
                for response in multiple_responses:
                    score = compute_eval_score(response, answer)
                    if score > 0:
                        corr_per_question = 1
                        break
                
                if corr_per_question > 0:
                    corr += 1
                
                total += 1
                if total % 10 == 0:
                    print(f"Progress: {corr}/{total} = {corr / total * 100:.2f}%")
        
        accuracy = corr / total * 100 if total > 0 else 0
        return corr, total, accuracy

    # def generate(self, question, sampler, temperature=0.6, top_k=50, top_p=0.95, seed=None):
    #     """Generate responses using the model."""
    #     if isinstance(question, str):
    #         input_batch = [format_prompt_with_chat_template(self.hf_tokenizer, question)]
    #     else:
    #         input_batch = [format_prompt_with_chat_template(self.hf_tokenizer, q) for q in question]

    #     max_gen_steps = min(self.total_generation_steps, 512)

    #     out_data = sampler(
    #         input_strings=input_batch,
    #         max_generation_steps=512,
    #         temperature=temperature,
    #         top_k=top_k,
    #         top_p=top_p,
    #         echo=False,
    #         seed=jax.random.PRNGKey(seed) if seed is not None else None,
    #     )

    #     output = out_data.text
    #     if isinstance(question, str):
    #         return output[0]
    #     return output

    # def generate(self, question, sampler, temperature=0.6, top_k=50, top_p=0.95, seed=None):
    #     """Generate responses using the model."""
    #     if isinstance(question, str):
    #         input_batch = [format_prompt_with_chat_template(self.hf_tokenizer, question)]
    #     else:
    #         input_batch = [format_prompt_with_chat_template(self.hf_tokenizer, q) for q in question]

    #     # DEBUG: Check prompt lengths
    #     prompt_lengths = [len(self.hf_tokenizer.encode(prompt)) for prompt in input_batch]
    #     max_prompt_len = max(prompt_lengths)
    #     # print(f"DEBUG: Max prompt length: {max_prompt_len} tokens")
    #     # print(f"DEBUG: Cache size: 1024, Available for generation: {1024 - max_prompt_len}")
        
    #     # Calculate safe generation length
    #     safe_gen_length = min(512, 1024 - max_prompt_len - 50)  # Leave 50 token buffer
    #     # print(f"DEBUG: Using generation length: {safe_gen_length}")

    #     out_data = sampler(
    #         input_strings=input_batch,
    #         max_generation_steps=safe_gen_length,  # Use calculated safe length
    #         temperature=temperature,
    #         top_k=top_k,
    #         top_p=top_p,
    #         echo=False,
    #         seed=jax.random.PRNGKey(seed) if seed is not None else None,
    #     )

    #     output = out_data.text
    #     if isinstance(question, str):
    #         return output[0]
    #     return output

    def generate(self, question, sampler, temperature=0.6, top_k=50, top_p=0.95, seed=None):
        """Generate responses using the model."""
        if isinstance(question, str):
            input_batch = [format_prompt_with_chat_template(self.hf_tokenizer, question)]
        else:
            input_batch = [format_prompt_with_chat_template(self.hf_tokenizer, q) for q in question]

        prompt_lengths = [len(self.hf_tokenizer.encode(prompt)) for prompt in input_batch]
        max_prompt_len = max(prompt_lengths)
        
        safe_gen_length = min(1024, 2048 - max_prompt_len - 50)
        
        # ADD STOP TOKEN (matching evaluation)
        stop_token_id = self.hf_tokenizer.encode('<|im_end|>')[0]

        out_data = sampler(
            input_strings=input_batch,
            max_generation_steps=safe_gen_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            echo=False,
            eos_tokens=[stop_token_id],  # ADD THIS
            seed=jax.random.PRNGKey(seed) if seed is not None else None,
        )

        output = out_data.text
        if isinstance(question, str):
            return output[0]
        return output


import numpy as np

# # One-Shot RLVR Reward Function (based on the paper)
# def one_shot_rlvr_reward_function(prompts, completions, answer, **kwargs):
#     """
#     Reward function for One-Shot RLVR based on the paper.
#     Uses sophisticated mathematical equivalence checking.
#     """
#     print("\n" + "="*80)
#     print("REWARD FUNCTION CALLED")
#     print(f"Number of completions: {len(completions)}")
#     print(f"Number of answers: {len(answer)}")
#     print("="*80)
    
#     scores = []
#     for i, (completion, ground_truth) in enumerate(zip(completions, answer)):
#         # Extract answer from completion
#         from utils import extract_answer, grade_answer_mathd, grade_answer_sympy
        
#         model_answer = extract_answer(completion)
        
#         # Compute score
#         if model_answer is None:
#             score = 0.0
#         else:
#             is_correct = (
#                 grade_answer_mathd(model_answer, ground_truth) or 
#                 grade_answer_sympy(model_answer, ground_truth)
#             )
#             score = 1.0 if is_correct else 0.0
        
#         scores.append(score)
        
#         # Debug first 5 examples
#         if i < 5:
#             print(f"\n--- Example {i} ---")
#             print(f"Ground truth: {ground_truth}")
#             print(f"Completion (first 300 chars): {completion[:300]}")
#             print(f"Extracted answer: {model_answer}")
#             print(f"Score: {score}")
    
#     avg_score = np.mean(scores) if scores else 0.0
#     print(f"\nAverage reward: {avg_score:.3f}")
#     print(f"Score distribution: {np.bincount([int(s) for s in scores]).tolist()}")
#     print("="*80 + "\n")
    
#     return scores

# def one_shot_rlvr_reward_function(prompts, completions, answer, **kwargs):
#     """
#     Reward function with negative rewards for completely wrong answers.
    
#     Reward scheme:
#     +1.0: Correct answer
#      0.0: Attempted answer but wrong
#     -0.5: No answer extracted (didn't follow format)
#     -1.0: Gibberish / very far from correct
#     """
#     print("\n" + "="*80)
#     print("REWARD FUNCTION CALLED")
#     print(f"Number of completions: {len(completions)}")
#     print("="*80)
    
#     scores = []
#     for i, (completion, ground_truth) in enumerate(zip(completions, answer)):
#         from utils import extract_answer, grade_answer_mathd, grade_answer_sympy
#         import re
        
#         model_answer = extract_answer_robust(completion)
        
#         # Reward structure
#         if model_answer is None:
#             # No boxed answer found - penalize for not following format
#             score = -0.5
#             reason = "No \\boxed{} answer found"
#         else:
#             # Check if correct
#             valid_answers = [ground_truth, "12.8", "12.7", "12.699", "12.71", "12.69", "\\sqrt[3]{2048}"]
#             is_correct = False
            
#             for valid_ans in valid_answers:
#                 if grade_answer_mathd(model_answer, valid_ans) or grade_answer_sympy(model_answer, valid_ans):
#                     is_correct = True
#                     break
            
#             if not is_correct:
#                 is_correct = (
#                     grade_answer_mathd(model_answer, ground_truth) or 
#                     grade_answer_sympy(model_answer, ground_truth)
#                 )
            
#             if is_correct:
#                 score = 1.0
#                 reason = "CORRECT!"
#             else:
#                 # Try to parse as number to check if close
#                 try:
#                     model_val = float(model_answer.replace("\\sqrt[3]{", "").replace("}", "").replace(",", ""))
#                     target_val = float(ground_truth)
                    
#                     # Check if close (within 50% of correct answer)
#                     ratio = abs(model_val - target_val) / target_val
                    
#                     if ratio < 0.1:
#                         # Very close - small negative
#                         score = 0.7
#                         reason = f"Close but not exact ({model_val} vs {target_val})"
#                     elif ratio < 0.5:
#                         # Somewhat close - attempted but wrong
#                         score = -0.3
#                         reason = f"Wrong but in ballpark ({model_val} vs {target_val})"
#                     else:
#                         # Very wrong
#                         score = -0.7
#                         reason = f"Very far from correct ({model_val} vs {target_val})"
#                 except:
#                     # Can't parse - probably symbolic or gibberish
#                     # Check if it's at least mathematical notation
#                     if any(c in model_answer for c in ['\\', '^', 'sqrt', 'frac']):
#                         score = -0.3
#                         reason = "Wrong but used math notation"
#                     else:
#                         score = -0.7
#                         reason = "Non-mathematical answer"
        
#         scores.append(score)
        
#         if i < 5:
#             print(f"\n--- Example {i} ---")
#             print(f"Ground truth: {ground_truth}")
#             print(f"Completion (first 200 chars): {completion[:200]}")
#             print(f"Extracted answer: {model_answer}")
#             print(f"Score: {score} ({reason})")
    
#     avg_score = np.mean(scores) if scores else 0.0
#     print(f"\nAverage reward: {avg_score:.3f}")
#     print(f"Score distribution: min={min(scores):.2f}, max={max(scores):.2f}, mean={avg_score:.3f}")
#     print("="*80 + "\n")
    
#     return scores

def one_shot_rlvr_reward_function(prompts, completions, answer, **kwargs):
    """
    Reward function for One-Shot RLVR - PAPER VERSION.
    
    CRITICAL: Paper uses BINARY 0-1 rewards ONLY.
    No negative rewards, no partial credit.
    
    Returns:
        List of 0.0 or 1.0 for each completion
    """
    scores = []
    
    for i, (completion, ground_truth) in enumerate(zip(completions, answer)):
        # Extract answer from completion
        model_answer = extract_answer_robust(completion)
        
        # Binary reward: 1.0 if correct, 0.0 otherwise
        if model_answer is None:
            score = 0.0
        else:
            # Check if correct using paper's evaluation methods
            is_correct = (
                grade_answer_mathd(model_answer, ground_truth) or 
                grade_answer_sympy(model_answer, ground_truth)
            )
            score = 1.0 if is_correct else 0.0
        
        scores.append(score)
    
    # Log statistics every N calls (to avoid spam)
    if len(scores) > 0:
        avg_score = np.mean(scores)
        if np.random.rand() < 0.01:  # Log 1% of the time
            correct_count = sum(1 for s in scores if s > 0)
            print(f"[Reward] Correct: {correct_count}/{len(scores)} ({avg_score:.1%})")
    
    return scores

def compute_eval_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute evaluation score (binary: 0 or 1).
    This is ONLY for evaluation, not for training rewards.
    
    For evaluation, we use strict binary scoring:
    - 1.0 if correct
    - 0.0 if incorrect
    
    This is different from the training reward function which can give
    negative rewards and partial credit.
    """
    from utils import extract_answer, grade_answer_mathd, grade_answer_sympy
    
    # Extract answer from model output
    model_answer = extract_answer_robust(solution_str)
    
    if model_answer is None:
        return 0.0
    
    # Check multiple valid forms of the answer
    valid_answers = [
        ground_truth,  # Original ground truth
    ]
    
    for valid_ans in valid_answers:
        is_correct = (
            grade_answer_mathd(model_answer, valid_ans) or 
            grade_answer_sympy(model_answer, valid_ans)
        )
        if is_correct:
            return 1.0
    
    return 0.0


# # One-Shot RLVR Reward Function (based on the paper)
# def one_shot_rlvr_reward_function(prompts, completions, answer, **kwargs):
#     """
#     Reward function for One-Shot RLVR based on the paper.
#     Uses sophisticated mathematical equivalence checking.
#     """
#     scores = []
#     for completion, ground_truth in zip(completions, answer):
#         score = compute_score_one_shot_rlvr(completion, ground_truth)
#         scores.append(score)
#     return scores


# # Import the actual DeepScaleR evaluation function
# from deepscaler import compute_score

# def compute_score_one_shot_rlvr(solution_str: str, ground_truth: str, use_think: bool = False) -> float:
#     """
#     Compute reward score for One-Shot RLVR using the actual DeepScaleR function.
#     This matches the paper's evaluation exactly.
#     """
#     # Use the actual DeepScaleR compute_score function from the paper
#     return compute_score(
#         data_source="math",  # or appropriate data source
#         solution_str=solution_str,
#         ground_truth=ground_truth,
#         extra_info=None,
#         use_think=use_think
#     )
