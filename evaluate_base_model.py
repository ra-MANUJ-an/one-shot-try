#!/usr/bin/env python3
import os
import time
from typing import Dict, Any, List
from pprint import pprint

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer
from datasets import load_dataset
import grain
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params_lib
from tunix.generate import sampler as sampler_lib
from utils import (
    format_prompt_with_chat_template,
    format_prompt_with_template,
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy
)


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

class Qwen25MathEvaluator:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        mesh_config=None,
        max_prompt_length: int = 1024,  # Increased from 512
        max_generation_steps: int = 1024,  # Increased from 512
    ):
        self.model_path = model_path
        self.max_prompt_length = max_prompt_length
        self.max_generation_steps = max_generation_steps
        
        if mesh_config is None:
            # Default: 4-way tensor parallelism
            mesh_config = [[1, 4], ["fsdp", "tp"]]
        self.mesh = jax.make_mesh(*mesh_config)
        self.tokenizer = None
        self.model = None
        self.sampler = None
        self.model_config = None
        
        print(f"Initializing Qwen2.5-Math-1.5B evaluator")
        print(f"Model path: {model_path}")
        print(f"Mesh config: {mesh_config}")
        print(f"Available devices: {jax.devices()}")
    
    def load_model(self):
        print("Loading model components...")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        print("Setting up model config...")
        self.model_config = qwen2_lib.ModelConfig.qwen2_5_1_5b()
        
        print(f"Downloading model files for {self.model_path}...")
        local_model_path = snapshot_download(repo_id=self.model_path)
        print(f"Model files downloaded to: {local_model_path}")

        print("Loading model from safe tensors...")
        with self.mesh:
            self.model = qwen2_params_lib.create_model_from_safe_tensors(
                file_dir=local_model_path,
                config=self.model_config,
                mesh=self.mesh
            )
        
        print("Model loaded successfully!")
        print("Creating sampler...")
        cache_config = sampler_lib.CacheConfig(
            cache_size=4096,
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
        )
        
        self.sampler = sampler_lib.Sampler(
            transformer=self.model,
            tokenizer=self.tokenizer,
            cache_config=cache_config,
        )
        
        print("Sampler created successfully!")
        
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "sampler": self.sampler,
            "config": self.model_config,
        }

    def load_math500_dataset(self, split: str = "test") -> grain.MapDataset:
        """Load MATH500 dataset"""
        print(f"Loading MATH500 dataset (split: {split})...")
        
        # Load from HuggingFace
        dataset_hf = load_dataset("HuggingFaceH4/MATH-500", split=split)
        data_list = list(dataset_hf)
        
        print(f"Loaded {len(data_list)} examples")
        print("Example data:")
        pprint(data_list[0])
        
        def process_item(item):
            question = item["problem"]
            answer = item["answer"]
            
            # Use the EXACT Qwen2.5-Math prompt format for zero-shot evaluation
            # prompt = (
            #     "<|im_start|>system\n"
            #     "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
            #     "<|im_start|>user\n"
            #     f"{question}<|im_end|>\n"
            #     "<|im_start|>assistant\n"
            # )

            # prompt = self.tokenizer.apply_chat_template(
            #     [
            #         {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            #         {"role": "user", "content": question},
            #     ],
            #     tokenize=False,
            #     add_generation_prompt=True,
            # )

            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": (
                        "Please reason step by step. "
                        "Your final answer must appear inside \\boxed{...} and nothing else."
                    )},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            # prompt += "\nAnswer: "

            
            return {
                "prompt": prompt,
                "question": question,
                "answer": answer,
            }
        
        dataset = grain.MapDataset.source(data_list).map(process_item)
        print("\n" + "="*60)
        print("DEBUG: First formatted prompt:")
        first_item = dataset[0]
        print(first_item["prompt"])
        print("="*60 + "\n")
        
        return dataset
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int = None,
    ) -> str:
        prompt_length = len(self.tokenizer.encode(prompt))
        cache_size = 4096
        safe_gen_length = min(
            self.max_generation_steps,
            cache_size - prompt_length - 100  # 100 token buffer
        )
        if safe_gen_length < 256:
            print(f"WARNING: Short generation length ({safe_gen_length} tokens) due to long prompt ({prompt_length} tokens)")
        
        stop_token_id = self.tokenizer.encode('<|im_end|>')[0]

        # Generate
        out_data = self.sampler(
            input_strings=[prompt],
            max_generation_steps=safe_gen_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            echo=False,
            eos_tokens=[stop_token_id],
            seed=jax.random.PRNGKey(seed) if seed is not None else None,
        )
        
        return out_data.text[0]
    
    def evaluate(
        self,
        batch_size: int = 8,
        num_batches: int = None,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        num_passes: int = 1,
        debug_first_n: int = 3,  # NEW: Debug first N examples
    ) -> Dict[str, Any]:
        print("=" * 60)
        print("Starting Evaluation")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Num batches: {num_batches or 'all'}")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k: {top_k}")
        print(f"  Top-p: {top_p}")
        print(f"  Passes per question: {num_passes}")
        print(f"  Debug first N examples: {debug_first_n}")
        print("=" * 60)
        
        # Load dataset
        dataset = self.load_math500_dataset()
        
        # Create batched dataset
        if num_batches is not None:
            dataset = dataset.batch(batch_size)[:num_batches]
        else:
            dataset = dataset.batch(batch_size)
        
        correct = 0
        total = 0
        results = []
        debug_count = 0
        
        # Evaluate batch by batch
        for batch_idx, batch in enumerate(tqdm(dataset, desc="Evaluating")):
            prompts = batch["prompt"]
            questions = batch["question"]
            answers = batch["answer"]
            
            # Process one question at a time to avoid memory issues
            for i, (prompt, question, answer) in enumerate(zip(prompts, questions, answers)):
                should_debug = debug_count < debug_first_n
                
                if should_debug:
                    print(f"\n{'='*60}")
                    print(f"DEBUG Example {debug_count + 1}/{debug_first_n}")
                    print(f"Question: {question[:]}")
                    print("="*60 + "\n")
                    print(f"Ground truth: {answer}")
                    print("="*60 + "\n")
                    print(f"Prompt (first 300 chars): {prompt[:]}")
                    print(f"Prompt length: {len(self.tokenizer.encode(prompt))} tokens")
                    print("="*60 + "\n")
                
                # Generate multiple passes
                responses = []
                for pass_idx in range(num_passes):
                    response = self.generate(
                        prompt=prompt,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=pass_idx,
                    )
                    responses.append(response)
                    
                    if should_debug:
                        # print(f"\nGeneration {pass_idx + 1}:")
                        print(f"Response: {response[:]}")
                        print("="*120 + "\n")
                
                is_correct = False
                extracted_answers = []
                for response in responses:
                    # Extract answer from response using utils.py function
                    # model_answer = extract_answer(response)
                    model_answer = extract_answer_robust(response)
                    extracted_answers.append(model_answer)
                    
                    if should_debug:
                        print(f"\nExtracted answer: {model_answer}")
                    
                    if model_answer is None:
                        continue
                    
                    # Grade answer using both methods from utils.py
                    is_correct = (
                        grade_answer_mathd(model_answer, answer) or 
                        grade_answer_sympy(model_answer, answer)
                    )
                    if should_debug:
                        print(f"Is correct: {is_correct}")
                    
                    if is_correct:
                        break
                
                if is_correct:
                    correct += 1
                
                if should_debug:
                    print(f"Final result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    print(f"Running accuracy: {correct}/{total+1} = {(correct/(total+1)*100):.2f}%")
                    debug_count += 1
                
                total += 1
                
                # Store result
                results.append({
                    "question": question,
                    "answer": answer,
                    "responses": responses,
                    "extracted_answers": extracted_answers,
                    "correct": is_correct,
                })
                
                # Print progress
                if total % 10 == 0:
                    current_acc = (correct / total * 100) if total > 0 else 0
                    print(f"\nProgress: {correct}/{total} = {current_acc:.2f}%")

        # Calculate final metrics
        accuracy = (correct / total * 100) if total > 0 else 0
        
        eval_results = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_passes": num_passes,
            "detailed_results": results,
        }
        
        return eval_results


def main():
    print("=" * 60)
    print("Qwen2.5-Math-1.5B Evaluation on MATH500")
    print("=" * 60)
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    mesh_config = [[1, 4], ["fsdp", "tp"]]  # 4-way tensor parallelism
    evaluator = Qwen25MathEvaluator(
        model_path=model_path,
        mesh_config=mesh_config,
        max_prompt_length=1024,  # Increased
        max_generation_steps=1024,  # Increased
    )
    
    evaluator.load_model()
    
    print("\nStarting evaluation...")
    results = evaluator.evaluate(
        batch_size=8,
        # num_batches=3,
        temperature=0.0,
        top_k=None,
        top_p=None,
        num_passes=1,
        debug_first_n=5,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: MATH500")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print("=" * 60)
    
    import json
    output_file = "qwen25_math_1_5b_eval_results.json"
    with open(output_file, 'w') as f:
        save_results = {k: v for k, v in results.items() if k != "detailed_results"}
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 60)
    print("Sample Failures (first 3):")
    print("=" * 60)
    failure_count = 0
    for result in results['detailed_results']:
        if not result['correct'] and failure_count < 3:
            print(f"\nQuestion: {result['question'][:150]}...")
            print(f"Ground truth: {result['answer']}")
            print(f"Extracted answers: {result['extracted_answers']}")
            failure_count += 1


if __name__ == "__main__":
    main()
