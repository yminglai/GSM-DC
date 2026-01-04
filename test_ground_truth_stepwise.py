#!/usr/bin/env python
# coding: utf-8
"""
test_ground_truth_stepwise.py
=============================
Test step-wise ground truth evaluation using all_problems.json
This script evaluates models by checking if each step in the solution is correct,
not just the final answer.
"""

import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.tools import tokenizer, fix_seed
from tools.tools_test import true_correct  # Use step-wise validation
from tools.irr_tools_test import true_correct as irr_true_correct  # Use irrelevant-aware validation
from math_gen.problem_gen import Problem
from data_gen.prototype.id_gen import IdGen_PT
from format.format import format_prompt
from test_batch import rebuild_problem_from_json, generate_problem, extract_final_answer

# =============== 环境与全局配置 ===============
device = "cuda" if torch.cuda.is_available() else "cpu"
fix_seed(42)

print("Loading tokenizer...")
model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model_tokenizer.pad_token = model_tokenizer.eos_token

# =============== Generate Response Using LLaMA Model ===============
def generate_response(op, problem, nshots, model):
    input_text = format_prompt(
        True, problem, op=op, nshots=nshots, 
        cur_id_gen=None, tokenizer=tokenizer, 
        generate_problem=generate_problem
    )
    inputs = model_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    input_len = inputs.input_ids.shape[1]
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)

    generated_text = model_tokenizer.decode(
        outputs[0][input_len:], 
        skip_special_tokens=True
    )
    return generated_text.strip()

# =============== Step-wise Evaluation Function ===============
def evaluate_stepwise(
    model_path: str,
    op_range: tuple = (2, 5),
    num_samples_per_condition: int = 10,
    nshots: int = 5
):
    """
    使用 tools_test.true_correct 进行逐步验证评估
    
    Args:
        model_path: 模型路径
        op_range: (start_op, end_op) OP范围
        num_samples_per_condition: 每个condition测试的样本数
        nshots: few-shot examples数量
    """
    
    model_name = model_path.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Step-wise Evaluation: {model_name}")
    print(f"OP Range: {op_range[0]} to {op_range[1]-1}")
    print(f"Samples per condition: {num_samples_per_condition}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    # Load dataset
    with open("./all_problems.json", "r") as file:
        all_data = json.load(file)
    
    # Results storage
    all_results = {
        "model_name": model_name,
        "model_path": model_path,
        "op_results": {}
    }
    
    # Iterate through OPs
    for op in range(op_range[0], op_range[1]):
        print(f"\n{'='*50}")
        print(f"Evaluating OP={op}")
        print(f"{'='*50}")
        
        start_idx = (op - 2) * 300
        end_idx = (op - 1) * 300
        op_data = all_data[start_idx:end_idx]
        
        op_results = {
            "light": {"stepwise_correct": 0, "irr_correct": 0, "final_correct": 0, "total": 0, "details": []},
            "medium": {"stepwise_correct": 0, "irr_correct": 0, "final_correct": 0, "total": 0, "details": []},
            "hard": {"stepwise_correct": 0, "irr_correct": 0, "final_correct": 0, "total": 0, "details": []}
        }
        
        conditions = ["light", "medium", "hard"]
        
        # Process each condition
        for cond_idx, condition in enumerate(conditions):
            print(f"\n  Condition: {condition.upper()}")
            
            # Get samples for this condition
            cond_start = cond_idx * 100
            cond_data = op_data[cond_start:cond_start + num_samples_per_condition]
            
            progress_bar = tqdm(
                cond_data, 
                desc=f"  OP={op} {condition:6s}",
                unit="problem"
            )
            
            for prob_data in progress_bar:
                # Rebuild problem
                problem = rebuild_problem_from_json(prob_data)
                
                # Tokenize problem text
                tokenized_problem = tokenizer.encode(". ".join(problem.problem))
                tokenized_problem[0] = 383
                problem_text = tokenizer.decode(tokenized_problem)
                
                # Generate solution
                predicted_solution = generate_response(op, problem_text, nshots, model)
                
                # Step-wise validation using tools_test.true_correct
                # This checks if each calculation step is correct
                stepwise_correct, stepwise_details = true_correct(
                    predicted_solution, 
                    problem
                )
                
                # Irrelevant-aware validation using irr_tools_test.true_correct
                # This checks correctness without using irrelevant context
                irr_correct, full_correct, irr_details, _ = irr_true_correct(
                    predicted_solution,
                    problem
                )
                
                # Extract final answer for comparison
                pred_final = extract_final_answer(predicted_solution)
                actual_final = str(prob_data["problem_info"]["final_answer"])
                final_correct = (pred_final == actual_final) if pred_final else False
                
                # Update statistics
                op_results[condition]["stepwise_correct"] += int(stepwise_correct)
                op_results[condition]["irr_correct"] += int(irr_correct)
                op_results[condition]["final_correct"] += int(final_correct)
                op_results[condition]["total"] += 1
                
                # Store detailed results
                op_results[condition]["details"].append({
                    "problem_text": problem_text,
                    "predicted_solution": predicted_solution,
                    "stepwise_correct": stepwise_correct,
                    "irr_correct": irr_correct,
                    "final_correct": final_correct,
                    "ground_truth_answer": actual_final,
                    "predicted_answer": pred_final,
                    "stepwise_details": stepwise_details,
                    "irr_details": irr_details
                })
                
                # Update progress bar
                stepwise_acc = op_results[condition]["stepwise_correct"] / op_results[condition]["total"] * 100
                irr_acc = op_results[condition]["irr_correct"] / op_results[condition]["total"] * 100
                final_acc = op_results[condition]["final_correct"] / op_results[condition]["total"] * 100
                progress_bar.set_postfix({
                    "step": f"{stepwise_acc:.1f}%",
                    "irr": f"{irr_acc:.1f}%",
                    "final": f"{final_acc:.1f}%"
                })
        
        # Calculate OP-level statistics
        total_stepwise = sum(r["stepwise_correct"] for r in op_results.values() if isinstance(r, dict) and "stepwise_correct" in r)
        total_irr = sum(r["irr_correct"] for r in op_results.values() if isinstance(r, dict) and "irr_correct" in r)
        total_final = sum(r["final_correct"] for r in op_results.values() if isinstance(r, dict) and "final_correct" in r)
        total_samples = sum(r["total"] for r in op_results.values() if isinstance(r, dict) and "total" in r)
        
        op_results["overall"] = {
            "stepwise_accuracy": (total_stepwise / total_samples * 100) if total_samples > 0 else 0,
            "irr_accuracy": (total_irr / total_samples * 100) if total_samples > 0 else 0,
            "final_accuracy": (total_final / total_samples * 100) if total_samples > 0 else 0,
            "stepwise_correct": total_stepwise,
            "irr_correct": total_irr,
            "final_correct": total_final,
            "total": total_samples
        }
        
        all_results["op_results"][op] = op_results
        
        # Print OP summary
        print(f"\n  OP={op} Summary:")
        print(f"    Step-wise Accuracy:     {op_results['overall']['stepwise_accuracy']:.2f}%")
        print(f"    Irr-aware Accuracy:     {op_results['overall']['irr_accuracy']:.2f}%")
        print(f"    Final Answer Accuracy:  {op_results['overall']['final_accuracy']:.2f}%")
        for cond in conditions:
            cond_stepwise = op_results[cond]["stepwise_correct"] / op_results[cond]["total"] * 100
            cond_irr = op_results[cond]["irr_correct"] / op_results[cond]["total"] * 100
            cond_final = op_results[cond]["final_correct"] / op_results[cond]["total"] * 100
            print(f"      {cond:6s}: Step={cond_stepwise:.1f}%  Irr={cond_irr:.1f}%  Final={cond_final:.1f}%")
    
    # Save results
    eval_dir = "eval"
    os.makedirs(eval_dir, exist_ok=True)
    output_file = f"{eval_dir}/{model_name}_stepwise_eval.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to {output_file}")
    print(f"{'='*60}\n")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - STEP-WISE EVALUATION")
    print("="*60)
    
    for op, op_res in all_results["op_results"].items():
        overall = op_res["overall"]
        print(f"\nOP={op}:")
        print(f"  Step-wise Accuracy:     {overall['stepwise_accuracy']:6.2f}%")
        print(f"  Irr-aware Accuracy:     {overall['irr_accuracy']:6.2f}%")
        print(f"  Final Answer Accuracy:  {overall['final_accuracy']:6.2f}%")
        print(f"  Gap (Irr - Step):       {overall['irr_accuracy'] - overall['stepwise_accuracy']:+6.2f}%")
        print(f"  Gap (Final - Irr):      {overall['final_accuracy'] - overall['irr_accuracy']:+6.2f}%")
    
    print("\n" + "="*60)
    
    # Cleanup
    model.cpu()
    del model
    torch.cuda.empty_cache()
    
    return all_results

# =============== Main Execution ===============
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"  # Change to your model path
    OP_RANGE = (2, 5)  # Test OP 2, 3, 4
    SAMPLES_PER_CONDITION = 20  # Test 20 samples per condition (light/medium/hard)
    NSHOTS = 5  # 5-shot examples
    
    # Run evaluation
    results = evaluate_stepwise(
        model_path=MODEL_PATH,
        op_range=OP_RANGE,
        num_samples_per_condition=SAMPLES_PER_CONDITION,
        nshots=NSHOTS
    )
    
    print("\n✅ Step-wise ground truth evaluation completed!")
