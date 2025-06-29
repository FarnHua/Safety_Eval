# Safety Evaluation Framework

This codebase supports the research in the paper ["Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging"](https://arxiv.org/abs/2412.19512).

This repository contains a collection of scripts and tools for evaluating language models on various tasks including safety, code generation, and domain-specific performance.

## Overview

The Safety_Eval framework provides standardized evaluation pipelines for different aspects of language model performance:

- **Safety Evaluation**: Test models on safety benchmarks like AdvBench and HEx-PHI
- **Code Generation**: Evaluate coding abilities using HumanEval
- **Domain-specific Performance**: Test performance on medical conversations, function generation, etc.

## Tools and Libraries

For model training and model merging, we use the following off-the-shelf repositories:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning
- [mergekit](https://github.com/arcee-ai/mergekit) for model merging techniques
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluating models on Big Bench Hard

You can also use your own custom fine-tuning or merging methods and integrate them with our evaluation framework.

## Prerequisites

- Linux environment
- Mamba/Conda environment with VLLM installed
- Python 3.x
- Hugging Face account and access to model weights

## Environment Setup

1. Create and activate the evaluation environment:
```bash
conda env create -f environment.yml
conda activate safety_eval
```

2. Set required environment variables:
```bash
export PYTHONWARNINGS="ignore::FutureWarning"
export HUGGINGFACE_HUB_CACHE="/path/to/huggingface/cache"  # Optional
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

## Directory Structure

- `safety_eval/`: Scripts for safety evaluations (AdvBench, HEx-PHI)
- `MagicCoder/`: Scripts for code generation evaluation (HumanEval)
- `ChatDoctor/`: Scripts for medical conversation evaluation
- `openfunction/`: Scripts for function generation evaluation

## Running Evaluations

### Safety Evaluation (AdvBench, HEx-PHI)

```bash
cd safety_eval
bash run_advbench.sh  # For AdvBench evaluation
bash run_hex-phi.sh   # For HEx-PHI evaluation
```

### Code Generation Evaluation (HumanEval)

```bash
cd MagicCoder
# First, install the human-eval package
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -e .
cd ..

# Then run the evaluation
bash run_gen_human-eval.sh
```

### Medical Conversation Evaluation

```bash
cd ChatDoctor
bash run_gen.sh
```

### Function Generation Evaluation

```bash
cd openfunction
bash run_eval.sh
```

## Configuration Options

All evaluation scripts support various modes that can be controlled by flags at the top of each script:

- Evaluate base models
- Evaluate fine-tuned models
- Apply regularization techniques (weight decay, dropout)
- Apply model merging techniques (linear merge, SLERP, DARE)

To enable a specific evaluation mode, set the corresponding flag to `true` in the script before running.

## Model Selection

Models can be configured in the `base_model` array at the top of each script:

```bash
base_model=(
    "llama3-8b-instruct"
    "gemma2-9b-it"
    "qwen25-7b-instruct"
    # Add/remove models as needed
)
```

## Evaluation Output

Results will be saved in directories specific to each evaluation type. Check the respective README files in each subdirectory for details on interpreting the results.

## Extending the Framework

To evaluate your own models:

1. Add your model to the `base_model` array in the relevant script
2. If you're using a fine-tuned model or custom merging technique, update the model path in the corresponding section
3. Run the evaluation script

## Common Parameters

All evaluation scripts use similar parameters:

- `seeds`: Random seeds for reproducibility (default: 42, 1024, 48763)
- `task`: Training/evaluation tasks
- Regularization parameters (weight decay, dropout rates)
- Merging parameters (merging factors for SLERP, DARE, etc.)

## Resource Requirements

These evaluations are resource-intensive. For optimal performance:

- GPUs with at least 24GB VRAM are recommended
- Sufficient disk space for model weights and evaluation results

## Note on Fine-tuned Models

The scripts assume fine-tuned models are saved in specific directories. If you have custom fine-tuned models, update the paths in the scripts accordingly.
