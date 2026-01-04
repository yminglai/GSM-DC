"""
Upload the GSM-DC test dataset to Hugging Face Hub.

Usage:
    python upload_to_huggingface.py --dataset-file all_problems.json

Requirements:
    pip install huggingface_hub
    huggingface-cli login
"""

import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_dataset_to_hf(dataset_file: str, repo_name: str, private: bool = False):
    """
    Upload the dataset to Hugging Face Hub.
    
    Args:
        dataset_file: Path to the all_problems.json file
        repo_name: Hugging Face repository name (e.g., "username/gsm-dc-test")
        private: Whether to create a private repository
    """
    dataset_path = Path(dataset_file)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    # Load and validate the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"📊 Dataset loaded: {len(dataset)} problems")
    
    # Create HF API instance
    api = HfApi()
    
    # Create repository (if it doesn't exist)
    try:
        create_repo(repo_name, repo_type="dataset", private=private, exist_ok=True)
        print(f"✅ Repository created/verified: {repo_name}")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
    
    # Upload the dataset file
    print(f"📤 Uploading {dataset_file} to {repo_name}...")
    api.upload_file(
        path_or_fileobj=str(dataset_path),
        path_in_repo="all_problems.json",
        repo_id=repo_name,
        repo_type="dataset",
    )
    
    # Create README.md with dataset card
    readme_content = f"""---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- math
- reasoning
- synthetic
size_categories:
- 1K<n<10K
---

# GSM-DC Test Dataset

This dataset contains the test set for GSM-DC (Grade School Math with Distractor Chains), a synthetic math reasoning dataset with controlled complexity.

## Dataset Details

- **Total Problems**: {len(dataset)}
- **Operation Counts (OP)**: 16-22 (out-of-distribution test set)
- **Problem Types**: Graph-based mathematical reasoning problems
- **Noise Levels**: Light, Medium, Hard (distractor difficulty)

## Dataset Structure

Each problem in `all_problems.json` contains:
- `problem_text`: The problem statement with all variables and relationships
- `solution`: Step-by-step ground truth solution
- `final_answer`: The numerical answer
- `n_op`: Number of operations (16-22)
- `noise_level`: Distractor difficulty (light/medium/hard)
- `graph_structure`: Internal graph representation
- `template_id`: Problem template identifier

## Usage

```python
import json

# Load the dataset
with open('all_problems.json', 'r') as f:
    problems = json.load(f)

# Access a problem
problem = problems[0]
print(problem['problem_text'])
print(problem['solution'])
print(problem['final_answer'])
```

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{{gsm-dc-2025,
    title={{GSM-DC: Grade School Math with Distractor Chains}},
    author={{[Your Name]}},
    booktitle={{Proceedings of EMNLP 2025}},
    year={{2025}}
}}
```

## Paper

Published at EMNLP 2025. [Paper Link]

## License

MIT License
"""
    
    # Upload README
    print("📝 Creating dataset card (README.md)...")
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )
    
    print(f"✅ Upload complete!")
    print(f"🔗 View your dataset at: https://huggingface.co/datasets/{repo_name}")

def main():
    parser = argparse.ArgumentParser(description="Upload GSM-DC dataset to Hugging Face Hub")
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="all_problems.json",
        help="Path to the all_problems.json file"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="YMinglai/GSM-DC-Dataset-Sample",
        help="Hugging Face repository name (default: YMinglai/GSM-DC-Dataset-Sample)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GSM-DC Dataset Upload to Hugging Face Hub")
    print("=" * 60)
    
    upload_dataset_to_hf(args.dataset_file, args.repo_name, args.private)

if __name__ == "__main__":
    main()
