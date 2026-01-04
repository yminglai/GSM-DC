<div align="center">

# 🧮 GSM-DC: Controlled Benchmark for LLM Distraction Analysis

### *How is LLM Distracted by Irrelevant Context? An Analysis Using A Controlled Benchmark*

**EMNLP 2025 Main Conference**

<p align="center">
<a href="https://arxiv.org/abs/2505.18761">
  <img src="https://img.shields.io/badge/📄_Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
</a>
<a href="https://aclanthology.org/2025.emnlp-main.674/">
  <img src="https://img.shields.io/badge/📚_ACL_Anthology-002FA7?style=for-the-badge&logo=acmdl&logoColor=white" alt="ACL Anthology">
</a>
<a href="https://huggingface.co/datasets/YMinglai/GSM-DC-Dataset-Sample">
  <img src="https://img.shields.io/badge/🤗_Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset">
</a>
<a href="https://ymingl.com/GSMDC/">
  <img src="https://img.shields.io/badge/🌐_Website-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website">
</a>
</p>

<p align="center">
<a href="https://ymingl.com/assets/pdf/gsmdc-slides.pdf">
  <img src="https://img.shields.io/badge/📊_Slides-E34F26?style=flat-square&logo=slides&logoColor=white" alt="Slides">
</a>
<a href="https://ymingl.com/assets/pdf/EMNLP2025-LLM-Distraction-Poster.pdf">
  <img src="https://img.shields.io/badge/📌_Poster-00ADD8?style=flat-square&logo=adobe-acrobat-reader&logoColor=white" alt="Poster">
</a>
<a href="https://github.com/yminglai/GSM-DC">
  <img src="https://img.shields.io/github/stars/yminglai/GSM-DC?style=flat-square&logo=github&label=Stars" alt="GitHub Stars">
</a>
<a href="https://github.com/yminglai/GSM-DC/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License">
</a>
</p>

---

<p align="center">
<a href="#-overview"><b>[Overview]</b></a> •
<a href="#-quick-start"><b>[Quick Start]</b></a> •
<a href="#-dataset"><b>[Dataset]</b></a> •
<a href="#-evaluation-pipeline"><b>[Evaluation]</b></a> •
<a href="#-citation"><b>[Citation]</b></a>
</p>

</div>

![Pipeline](imgs/pipeline.png)

**GSM-DC** is a synthetic data generator and evaluator for studying the reasoning robustness of LLMs under irrelevant context injection. Control problem complexity, distractor injection, and evaluate step-wise correctness with symbolic validation.

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Evaluate Your Model

```python
# 1. Configure your model in evaluate.py, you can use `op=2-15` to finetune the models.
MODEL_PATH = "YOUR_MODEL_PATH"  # e.g., "meta-llama/Llama-3.2-1B-Instruct"
PRM_MODEL_NAME = "YOUR_PRM_MODEL"  # Optional: for tree search

# 2. Run evaluation
python evaluate.py
```

Results will be saved in `eval/` directory with metrics for:
- ✅ **Step-wise correctness**: Are all reasoning steps correct?
- 🎯 **Irrelevant-aware correctness**: Correctness without using irrelevant context
- 🔢 **Final answer accuracy**: Is the extracted answer correct?

---

## 📊 Dataset

**Sample Dataset**: [YMinglai/GSM-DC-Dataset-Sample](https://huggingface.co/datasets/YMinglai/GSM-DC-Dataset-Sample)

The dataset contains 6,300 problems (OP 2-22) across three noise levels (light/medium/hard). The full GSM-DC framework is designed for **on-the-fly generation**, allowing you to control problem complexity and distractor injection.

---

## 🎯 Overview

This repository provides tools to:
- 🔧 **Generate** symbolic math problems as dependency graphs (DAGs)
- 🎭 **Inject** irrelevant context (IC) in a controlled manner
- 📝 **Render** problems into natural language using GSM8K-style templates
- ✔️ **Evaluate** LLM responses at each reasoning step using a symbolic parser
- 📈 **Visualize** graphs and reasoning pipelines

**Example**: See `example_ic.ipynb` for interactive demonstration

---

## 🏗️ Dataset Structure and Pipeline

Each problem is represented as a tuple: **(G', M, P, S)**
- **G'**: Augmented graph with distractors
- **M**: Natural language problem
- **P**: Reasoning path
- S: Ground-truth solution

---

## Problem and Graph

When calling `id_gen.gen_prob()`, a `Problem` instance is initialized:

- **Graph Class**: Manages DAG construction
- **Problem Class**: Adds parameters, values, entity names, and text rendering
- Graph stored as `id_gen.problem.G` using NumPy boolean matrices
- Nodes: tuples (i, j, k, l) with meanings:
  - RNG: (-1, 0, 0, 0)
  - Instance Parameter: (0, j, k, l)
  - Abstract Parameter: (1, j, k, l)
- Dependency graph: `id_gen.problem.template`, a `networkx.DiGraph`
- Value map: `id_gen.problem.lookup`
- Entity names: `id_gen.problem.N[i][j]`
- Drawing tools: `id_gen.problem.draw()`

![Sample](imgs/sample.png)

---

## Benchmarking LLM Robustness

We systematically benchmark six models to study the impact of irrelevant context (IC) on multi-step reasoning tasks:
- **Closed-source models**: Grok-3-Beta, GPT-4.1, GPT-4o-mini
- **Open-source models**: LLaMA-3.3-70B, LLaMA-3.1-8B, LLaMA-3.2-1B

Each model is evaluated using a five-shot prompting strategy, enhanced with a structured `Background` section that highlights necessary dependencies.

We vary the number of injected irrelevant nodes (\(m = 1 	ext{–} 15\)) across four reasoning depths (\(rs = 2, 3, 4, 5\)) and compute:
- **Step Accuracy (SAcc)**: Are all reasoning steps correct?
- **Path Accuracy (PAcc)**: Is the full reasoning path valid?
- **Extraction Accuracy (EAcc)**: Is the final answer correct?

Each point is averaged over 100 generated problems per configuration.

---

![Close Source Results](imgs/closed_open_source.png)

*Figure: Step accuracy of six LLMs under increasing irrelevant context. Left: Grok-3-Beta, GPT-4.1, GPT-4o-mini (closed-source models); Right: LLaMA-3.3-70B, LLaMA-3.1-8B, LLaMA-3.2-1B (open-source models). Each curve represents a reasoning depth \(rs \in \{2, 3, 4, 5\}\).*

---

## Training the Process Reward Model (PRM)

The PRM is trained on a dataset labeled by `true_correct()` in `tools/irr_tools_test.py`. To create this dataset:
- Run `generate_dataset.py`
- For each generated problem:
  - Model generates stepwise CoT output
  - Each step is scored: steps after the first mistake are labeled as incorrect

Use `prm_train.py` to train the PRM using this dataset.

---

## Tree-of-Thoughts (ToT) Search with PRM

During evaluation, we use a PRM-guided Tree-of-Thought search (`prm_tree.py`).

- `N`: Initial number of root paths
- `M`: Beam width per path
- `K = N / M`: Top-K continuations explored per step

This guided search improves robustness, especially under high IC.

![Tree Search](imgs/treesearch.png)

---

## 🏆 Benchmarking LLM Robustness

We benchmarked six LLMs:
- **Closed models**: Grok-3-Beta, GPT-4.1, GPT-4o-mini
- **Open models**: LLaMA-3.3-70B, LLaMA-3.1-8B, LLaMA-3.2-1B

### Evaluation Setup
- **Prompting**: 5-shot examples with structured Background section
- **Metrics**:
  - **SAcc**: Step Accuracy (step-wise correctness)
  - **PAcc**: Path Accuracy (reasoning path)
  - **EAcc**: Extraction Accuracy (final answer)

---

## 🔬 Evaluation Pipeline

To test your own LLM using the GSM-DC dataset:

```bash
# 1. Configure evaluate.py
MODEL_PATH = "YOUR_MODEL_PATH"

# 2. Run evaluation
python evaluate.py
```

The evaluation script (`evaluate.py`) provides:
- 📥 **Dataset Loading**: Automatic download from HuggingFace
- 🔄 **Problem Reconstruction**: Rebuilds Problem objects from JSON
- 🤖 **Model Inference**: Generates responses with optional tree search
- ✅ **Validation**: Step-wise and irrelevant-aware correctness checking
- 📊 **Metrics**: Computes SAcc, PAcc, and EAcc

### Dataset Access

**🤗 Sample Dataset**: [YMinglai/GSM-DC-Dataset-Sample](https://huggingface.co/datasets/YMinglai/GSM-DC-Dataset-Sample)

The sample contains **6,300 problems** (OP 2-22) across three noise levels for comprehensive evaluation.

**🔧 On-the-Fly Generation**: The full GSM-DC framework allows you to:
- Control problem complexity (number of operations, reasoning depth)
- Adjust irrelevant context injection (noise level, distractor count)
- Generate unlimited evaluation sets with different characteristics

---

## 🙏 Acknowledgements

This project builds upon:
- [**GSM8K**](https://github.com/openai/grade-school-math) by OpenAI
- [**iGSM**](https://github.com/facebookresearch/iGSM) by Facebook Research (hierarchical entity vocabulary and graph QA construction)
- [**PRM**](https://github.com/sdiehl/prm) by Stephen Diehl

---

## 📖 Citation

```bibtex
@inproceedings{yang-etal-2025-llm-reasoning,
    title = "How Is {LLM} Reasoning Distracted by Irrelevant Context? An Analysis Using a Controlled Benchmark",
    author = "Yang, Minglai  and
      Huang, Ethan  and
      Zhang, Liang  and
      Surdeanu, Mihai  and
      Wang, William Yang  and
      Pan, Liangming",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.674/",
    doi = "10.18653/v1/2025.emnlp-main.674",
    pages = "13340--13358",
    ISBN = "979-8-89176-332-6",
    abstract = "We introduce Grade School Math with Distracting Context (GSM-DC), a synthetic benchmark to evaluate Large Language Models' (LLMs) reasoning robustness against systematically controlled irrelevant context (IC). GSM-DC constructs symbolic reasoning graphs with precise distractor injections, enabling rigorous, reproducible evaluation. Our experiments demonstrate that LLMs are significantly sensitive to IC, affecting both reasoning path selection and arithmetic accuracy. Additionally, training models with strong distractors improves performance in both in-distribution and out-of-distribution scenarios. We further propose a stepwise tree search guided by a process reward model, which notably enhances robustness in out-of-distribution conditions."
}
```
