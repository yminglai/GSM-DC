import torch
import sys
from data_gen.prototype.id_gen import IdGen_PT
from data_gen.pretrain.id_retry_gen import IdGen
from tools.tools import fix_seed
from tools.tools_test import true_correct
from typing import Literal
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
from huggingface_hub import login
import os
import json
import re
import random
from datasets import load_dataset, Dataset
from transformers.utils import logging
import threading

logging.set_verbosity_error() 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

reward_token_ids = torch.tensor([tokenizer.convert_tokens_to_ids("+"), tokenizer.convert_tokens_to_ids("-")]).to(device)

logits_mask = torch.full((len(tokenizer),), float("-inf")).to(device)
logits_mask[reward_token_ids] = 0

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False

def get_score(prm_model, tokenizer, output):
    decoded_output = tokenizer.decode(output)
    prm_input = tokenizer.encode(decoded_output, return_tensors="pt", truncation=True)

    prm_input = prm_input.to(device)

    logits = prm_model(prm_input).logits + logits_mask
    
    probs = torch.softmax(logits, dim=-1).to(device)
    reward_probs = probs[0, probs.shape[1] - 1, reward_token_ids]
    score = reward_probs[0]

    return float(score)

def select_top_K(paths, k):
    # reversed the labels by accident, so higher = neg, lower = pos ?
    paths.sort(key=lambda x: x[1])
    
    if (len(paths) < k):
        return paths

    # paths.sort(key=lambda x: x[1], reverse=True)
    candidates = paths[0:k]

    for i in range(k, len(paths)):
        paths[i][0].cpu()

    return candidates

def generate_problem(op=3):
    id_gen = IdGen_PT(
        style="light",
        op_style="light",
        op=op,
        perm_level=5,
        detail_level=0
    )
    id_gen.gen_prob([i for i in range(5)], p_format="pq")
    return id_gen

# id_gen = generate_problem()
# prob = f"<|start_header_id|>user<|end_header_id|>\n"+id_gen.prob+f"<|start_header_id|>assistant<|end_header_id|>\n"
# # print(id_gen.sol)
# tokenized_prob = tokenizer.encode(prob, return_tensors="pt").to(device)

def tree_search(prm_model, model, tokenizer, prob, N, M, max_steps, temperature, device):
    # stop_tokens = [".", ";", "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]
    # stop_tokens = [".", ";"]
    # my_stop_criteria = StoppingCriteriaList([MyStoppingCriteria(stops=[tokenizer.encode(stop_tokens, add_special_tokens=False,return_tensors="pt", is_split_into_words=True).squeeze()])])
    my_stop_criteria = StoppingCriteriaList([MyStoppingCriteria(stops=torch.tensor([13, 26, 128009]))])
    starting_paths = []
    question_length = prob.shape[1] - 1
    next_candidates = []
    candidates = []

    #can inculde this or not if OOM
    probs = torch.tensor(prob.tolist()*(N//4)).to(device)

    with torch.no_grad():
        output = model.generate(probs, stopping_criteria=my_stop_criteria, max_new_tokens=16384, temperature=temperature)
        # print(f"initial output {output}")
        for i in range(len(output)):
            score = get_score(prm_model, tokenizer, output[i])
            starting_paths.append([output[i], score])

    starting_paths = select_top_K(starting_paths, N // M)
    candidates = starting_paths
    final_answers = []
    curr_step = 1

    # while there is no final answer and max_step not reached
    while len(final_answers) < M and curr_step != max_steps:
        candidates = [path[0].tolist() for path in candidates]
        next_candidates = []

        for candidate in candidates:
            input = torch.tensor([candidate]*M).to(device)
            output = model.generate(input, stopping_criteria=my_stop_criteria, max_new_tokens=16384, temperature=temperature)

            for i in range(len(output)):
                score = get_score(prm_model, tokenizer, output[i])
                next_candidates.append([output[i], score])

        remove = []
        for i in range(len(next_candidates)):
            next_candidate = next_candidates[0]
            id = next_candidate[0]
            if id[-1] == 128009 or id[-1] == 128001 or id[-1] == 128008:
                remove.append(i)

        for index in remove:
            final_answers.append(next_candidates[index])

        filtered = []
        for i in range(len(next_candidates)):
            if i not in remove:
                filtered.append(next_candidates[i])

        next_candidates = filtered
        next_candidates = select_top_K(next_candidates, N // M)
        
        candidates = next_candidates
        curr_step += 1

    if len(final_answers) == 0:
        final_answers = select_top_K(candidates, N // M)

    for i in range(len(final_answers)):
        if final_answers[i][0][-2] != 13:
            final_answers[i][1] = 1

    final_answers = select_top_K(final_answers, N // M)

    # print(f"final answer: {tokenizer.decode(final_answers[0][0][question_length:], skip_special_tokens=True)}")
    return final_answers[0]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    prm_model = AutoModelForCausalLM.from_pretrained("./models/llama1b_new_dataset")
    prm_model = prm_model.to(device)
    prm_model.eval()

    model = AutoModelForCausalLM.from_pretrained("minglyang/llama-1b-instruct-gsmdi-clean-full")
    model = model.to(device)

    fix_seed(42)

    with open("./finetune_dataset/old_finetune_dataset.json", "r") as file:
        json_data = json.load(file)

    data = json_data[0]["input"]
    data = tokenizer.encode(data, return_tensors="pt").to(device)
    N = 4
    M = 2
    MAX_STEPS = 3
    temp = 0.7
    while True:
        data = torch.reshape(data, shape=(1, -1))
        output = model(data)
        logits = output.logits[0][-1]
        next_token_id = torch.argmax(logits)
        next_token = tokenizer.decode(next_token_id)
        data = torch.concat((data[0], torch.tensor([next_token_id]).to(device)))
        print(next_token_id)
        if (next_token_id == tokenizer.eos_token_id - 1):
            print("BOOM")
            break

    print(data)
    print(tokenizer.decode(data))

#stop condition . or ;