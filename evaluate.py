#!/usr/bin/env python
"""
GSM-DC Evaluation Script
========================
Main script for evaluating language models on the GSM-DC dataset.

This script:
1. Loads the GSM-DC dataset from HuggingFace
2. Generates model responses for each problem
3. Validates responses using step-wise and irrelevant-aware checking
4. Computes accuracy metrics across different noise levels

Usage:
    python evaluate.py

Configuration:
    Edit the constants below to set your model paths and evaluation parameters.
"""

import os
import re
import json
import torch
import threading
import numpy as np
import networkx as nx
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from tools.tools import tokenizer, fix_seed
from tools.irr_tools_test import true_correct
from math_gen.problem_gen import Problem, Expression
from data_gen.prototype.id_gen import IdGen_PT
from format.format import format_prompt

# =============== Configuration ===============
device = "cuda" if torch.cuda.is_available() else "cpu"
fix_seed(42)

# Model Configuration
MODEL_PATH = "YOUR_MODEL_PATH"  # e.g., "meta-llama/Llama-3.2-1B-Instruct"
PRM_MODEL_NAME = "YOUR_PRM_MODEL"  # Optional: e.g., "remy9926/llama1b_in_dist_prm" (for tree search)

# Evaluation Configuration
OP_VALUES = (16, 17, 18, 19, 20, 21, 22)  # Operation counts to evaluate
DATASET_PATH = "YMinglai/GSM-DC-Dataset-Sample"
NSHOTS = 5  # Number of few-shot examples

# =============== Load Models ===============
print("Loading models...")
models = {
    op: AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=False
    ).to(device).eval()
    for op in OP_VALUES
}

model_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model_tokenizer.pad_token = model_tokenizer.eos_token

# Optional: Load PRM for tree search
# prm_model = AutoModelForCausalLM.from_pretrained(PRM_MODEL_NAME).to(device)

print("Models loaded successfully!")

# Load dataset
dataset = load_dataset(DATASET_PATH, data_files="all_problems.json")["train"]

# =============== Utility Functions ===============

def parse_param_str(s: str):
    """Parse parameter string like '(0, 1, 2, 3)' to tuple."""
    inside = s.strip("()")
    parts = inside.split(",")
    if len(parts) != 4:
        raise ValueError(f"param tuple must have length 4, got {s}")
    return tuple(int(x.strip()) for x in parts)


def rebuild_problem_from_json(data_dict: dict) -> Problem:
    """Reconstruct Problem object from JSON dictionary."""
    info = data_dict["problem_info"]
    d = int(info["d"])
    w0 = int(info["w0"])
    w1 = int(info["w1"])
    e = int(info["e"])
    p = float(info["p"])
    final_answer = info["final_answer"]
    if final_answer is not None:
        final_answer = int(final_answer)
    question_index = info["question_index"]

    args = {
        "rand_perm": "none",
        "define_var": True,
        "define_detail": True,
        "inter_var": True,
        "name_omit": False,
        "cal_omit": False,
        "dot": "'s ",
        "symbol_method": "rand",
        "sol_sort": False,
        "perm": False
    }
    problem = Problem(d, w0, w1, e, p, args=args)

    node_data = data_dict["node_data"]
    layer_counts = {}
    for key, value in node_data.items():
        node_str = value["node"]
        layer_str, idx_str = node_str.strip("()").split(",")
        layer = int(layer_str)
        idx = int(idx_str)
        layer_counts[layer] = max(layer_counts.get(layer, 0), idx+1)
    
    for i in range(d):
        problem.l[i] = layer_counts.get(i, 0)

    problem.graph = nx.DiGraph()
    for i in range(d):
        for j in range(problem.l[i]):
            problem.graph.add_node((i, j), unique=False)

    problem.N = []
    for i in range(d):
        problem.N.append([""] * problem.l[i])

    for key, value in node_data.items():
        node_str = value["node"]
        layer_str, idx_str = node_str.strip("()").split(",")
        layer = int(layer_str)
        idx = int(idx_str)
        label = key
        problem.N[layer][idx] = label
        problem.graph.nodes[(layer, idx)]["unique"] = bool(value["unique"])
        if bool(value["unique"]):
            problem.unique.append((layer, idx))

    edges = data_dict["edges"]
    for u_str, v_str in edges:
        ulayer_idx = u_str.strip("()").split(",")
        u_layer = int(ulayer_idx[0])
        u_idx = int(ulayer_idx[1])
        vlayer_idx = v_str.strip("()").split(",")
        v_layer = int(vlayer_idx[0])
        v_idx = int(vlayer_idx[1])
        problem.graph.add_edge((u_layer, u_idx), (v_layer, v_idx), chosen=False)

    problem.G = []
    for i in range(d - 1):
        M = np.zeros((problem.l[i], problem.l[i+1]), dtype=bool)
        for j in range(problem.l[i]):
            for k in range(problem.l[i+1]):
                if problem.graph.has_edge((i, j), (i+1, k)):
                    M[j, k] = True
        problem.G.append(M)

    ln_list = data_dict.get("ln", [])
    problem.ln = [str(x) for x in ln_list]

    problem.template = nx.DiGraph()
    template_edges = data_dict["template_edges"]
    node_set = set()
    for u_str, v_str in template_edges:
        node_set.add(u_str)
        node_set.add(v_str)
    for n_str in node_set:
        param_tup = parse_param_str(n_str)
        problem.template.add_node(param_tup)
    for u_str, v_str in template_edges:
        u_tup = parse_param_str(u_str)
        v_tup = parse_param_str(v_str)
        problem.template.add_edge(u_tup, v_tup)

    whole_template_edges = data_dict.get("whole_template_edges", [])
    problem.whole_template = nx.DiGraph()
    wt_node_set = set()
    for u_str, v_str in whole_template_edges:
        wt_node_set.add(u_str)
        wt_node_set.add(v_str)
    for n_str in wt_node_set:
        param_tup = parse_param_str(n_str)
        problem.whole_template.add_node(param_tup)
    for u_str, v_str in whole_template_edges:
        u_tup = parse_param_str(u_str)
        v_tup = parse_param_str(v_str)
        problem.whole_template.add_edge(u_tup, v_tup)

    problem.all_param = []
    if "all_param" in data_dict:
        for sp in data_dict["all_param"]:
            problem.all_param.append(parse_param_str(sp))

    problem.ans = final_answer if final_answer is not None else 0
    if question_index is not None:
        problem.ques_idx = tuple(question_index)

    topo_list = data_dict.get("topological_order", [])
    if topo_list:
        problem.topological_order = [
            parse_param_str(item["param"]) if isinstance(item, dict) else parse_param_str(item)
            for item in topo_list
        ]
    else:
        problem.topological_order = []

    n_op = 0
    for param in problem.topological_order:
        num_pre = len(list(problem.template.predecessors(param)))
        if num_pre <= 2:
            n_op += 1
        else:
            n_op += num_pre - 1
    problem.n_op = n_op

    build_name2param_dict(problem)

    prob_text = data_dict.get("problem_text", [])
    sol_text = data_dict.get("solution_text", [])
    problem.problem = prob_text
    problem.solution = sol_text

    return problem


def build_name2param_dict(problem: Problem):
    """Build mapping from parameter names to parameter tuples."""
    problem.name2param_dict = {}
    for param in problem.all_param:
        l, i, j, k = param
        if l == -1:
            param_name = "RNG"
        elif l == 0:
            name0 = problem.N[i][j]
            name1 = problem.N[i+1][k]
            param_name = f"{name0}{problem.args['dot']}{name1}"
        elif l == 1:
            name0 = problem.N[i][j]
            cat = problem.ln[k]
            param_name = f"{name0}{problem.args['dot']}{cat}"
        else:
            param_name = f"UnsupportedParam{param}"
        problem.name2param_dict[param_name] = param


def extract_final_answer(text):
    """Extract final answer from model output."""
    pattern = r"<<\s*(\d+)\s*>>"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    match = re.search(r'(\d+)', text)
    if match:
        return match.group(1).strip()
    return None


def generate_problem(op=3):
    """Generate a sample problem for few-shot examples."""
    id_gen = IdGen_PT(
        style="light",
        op_style="light",
        op=op,
        perm_level=5,
        detail_level=0
    )
    id_gen.gen_prob([i for i in range(5)], p_format="pq")
    return id_gen


def generate_response(op, problem, nshots, model):
    """Generate model response for a given problem."""
    input_text = format_prompt(
        True, problem, op=op, nshots=nshots,
        cur_id_gen=None, tokenizer=tokenizer,
        generate_problem=generate_problem
    )
    inputs = model_tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_len = inputs.input_ids.shape[1]
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)
        # Optional: Use tree search with PRM
        # outputs = tree_search(prm_model, model, model_tokenizer,
        #                       inputs["input_ids"], 16, 4, 6*op, 0.8, device)

    generated_text = model_tokenizer.decode(
        outputs[0][input_len - 1:],
        skip_special_tokens=True
    )
    return generated_text.strip()


# =============== Evaluation Function ===============

def evaluate(op: int, model, results_dict):
    """Evaluate model for a given operation count."""
    model_name = MODEL_PATH.split("/")[-1]
    results = []
    eval_dir = "eval"
    os.makedirs(eval_dir, exist_ok=True)
    output_file = f"{eval_dir}/{model_name}_op{op}_evaluation.json"
    
    start = (op - 2) * 300
    end = (op - 1) * 300
    data = dataset[start:end]

    conditions = {
        "light": {"correct": 0, "irr_correct": 0, "extracted_correct": 0, "total": 0},
        "medium": {"correct": 0, "irr_correct": 0, "extracted_correct": 0, "total": 0},
        "hard": {"correct": 0, "irr_correct": 0, "extracted_correct": 0, "total": 0}
    }

    num_problems_per_condition = 100

    progress_bar = tqdm(
        total=num_problems_per_condition * 3,
        desc=f"Evaluating OP={op}",
        unit="problem"
    )

    for condition_idx, condition in enumerate(["light", "medium", "hard"]):
        for i in range(num_problems_per_condition):
            idx = (condition_idx * 100) + i
            problem = rebuild_problem_from_json(data[idx])
            
            tokenized_problem = tokenizer.encode(". ".join(problem.problem))
            tokenized_problem[0] = 383
            problem_text = tokenizer.decode(tokenized_problem)
            
            predicted_solution = generate_response(op, problem_text, NSHOTS, model)
            irr_correct, correct, my_print, _ = true_correct(predicted_solution, problem)

            pred_final = extract_final_answer(predicted_solution)
            actual_final = int(data[idx]["problem_info"]["final_answer"])
            extracted_correct = int(
                pred_final is not None and
                actual_final is not None and
                pred_final == actual_final
            )

            conditions[condition]["correct"] += int(correct)
            conditions[condition]["irr_correct"] += int(irr_correct)
            conditions[condition]["extracted_correct"] += extracted_correct
            conditions[condition]["total"] += 1

            results.append({
                'condition': condition,
                'problem': problem_text,
                'predicted_solution': predicted_solution,
                'correct': correct,
                'irr_correct': irr_correct,
                'extracted_correct': extracted_correct,
                'ground_truth_answer': actual_final,
                'predicted_answer': pred_final
            })
            
            progress_bar.update(1)

    progress_bar.close()

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump({
            "model": model_name,
            "op": op,
            "conditions": conditions,
            "results": results
        }, json_file, indent=4, ensure_ascii=False)
    
    print(f"✅ OP={op} results saved to {output_file}")
    
    for condition, stats in conditions.items():
        acc = stats["correct"] / stats["total"] * 100
        irr_acc = stats["irr_correct"] / stats["total"] * 100
        ext_acc = stats["extracted_correct"] / stats["total"] * 100
        print(f"  {condition:6s}: Step={acc:.1f}%  Irr={irr_acc:.1f}%  Final={ext_acc:.1f}%")


# =============== Main Execution ===============

if __name__ == "__main__":
    results_dict = {}
    threads = []

    print(f"\n{'='*60}")
    print(f"GSM-DC Evaluation")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"OPs: {OP_VALUES}")
    print(f"{'='*60}\n")

    for op in OP_VALUES:
        thread = threading.Thread(target=evaluate, args=(op, models[op], results_dict))
        threads.append(thread)
        thread.start()
        print(f"Started evaluation thread for OP={op}")

    for thread in threads:
        thread.join()

    print(f"\n{'='*60}")
    print("✅ Evaluation complete! Results saved in eval/ directory")
    print(f"{'='*60}\n")
