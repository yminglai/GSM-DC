import os, re, json, random
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from test_batch_saved import (
    rebuild_problem_from_json,
    generate_problem,
    format_prompt, tokenizer as core_tokenizer
)

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def debug_one_sample(
        sample_idx: int = 0,
        data_file: str = "all_problems.json",
        model_name: str = "YOUR_MODEL_PATH",  # e.g., "meta-llama/Llama-3.2-1B-Instruct"
        run_model: bool = True,
        max_new_tokens: int = 128
    ):
    """
    Quick sanity-check for:
        - rebuild_problem_from_json
        - prompt format
        - true_correct scoring logic
    """

    ds = load_dataset("json", data_files={"train": data_file})["train"]
    raw = ds[sample_idx]
    prob = rebuild_problem_from_json(raw)

    print("\n=== PROBLEM TEXT ===")
    print(" ".join(prob.problem))
    print("GT answer:", prob.ans)

    if run_model:
        tok = AutoTokenizer.from_pretrained(
            model_name if "meta-llama" not in model_name else "meta-llama/Llama-3.2-1B-Instruct"
        )
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        ).eval()

        prompt = format_prompt(
            True,
            prob,
            op=prob.n_op,
            nshots=5,
            cur_id_gen=None,
            tokenizer=core_tokenizer,
            generate_problem=generate_problem
        )
        inputs = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        pred = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print("\n=== MODEL OUTPUT ===")
        print(pred)

        from tools.irr_tools_test import true_correct
        from test_batch_saved import extract_final_answer
        irr_ok, ok, _, _ = true_correct(pred, prob)
        print(f"\n✓ correct? {ok} | ✓ irr_correct? {irr_ok}")
        print("Extracted final:", extract_final_answer(pred))

    else:
        print("\n(Model not executed, only verified Problem reconstruction)")

if __name__ == "__main__":
    debug_one_sample(
        sample_idx=5,
        data_file="all_problems.json",
        model_name="YOUR_MODEL_PATH",  # Replace with your model path
        run_model=True
    )