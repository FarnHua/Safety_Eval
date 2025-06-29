# MagicCoder Evaluation Scripts

This directory contains scripts for evaluating language models on coding benchmarks like HumanEval.

## Prerequisites

- Linux environment
- Mamba/Conda environment with VLLM installed
- Python 3.x
- Hugging Face account and access to model weights

## Environment Setup

1. **Important**: Install OpenAI's human-eval package in this directory:
```bash
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -e .
cd ..
```

2. Set required environment variables:
```bash
export PYTHONWARNINGS="ignore::FutureWarning"
export HUGGINGFACE_HUB_CACHE="/path/to/huggingface/cache"  # Optional
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

## Running Evaluations

Use the main evaluation script:

```bash
bash run_gen_human-eval.sh
```

## Configuration Options

The script supports various evaluation modes controlled by flags at the top:

- `RUN_ORIGINAL_FT`: Evaluate original fine-tuned models
- `RUN_WEIGHT_DECAY`: Evaluate weight decay variants
- `RUN_DROPOUT`: Evaluate dropout variants
- `RUN_DIFF_LR`: Evaluate models with different learning rates
- `RUN_LINEAR_MERGE`: Evaluate linear merged models
- `RUN_SLERP`: Evaluate SLERP merged models
- `RUN_DARE`: Evaluate DARE merged models
- `RUN_BASE_MODEL`: Evaluate base models

Set the desired flags to `true` in the script before running.

## Model Selection

Models can be configured in the `base_model` array at the top of the script:

```bash
base_model=(
    "llama3-8b-instruct"
    "gemma2-2b-it"
    "qwen25-7b-instruct"
    # Add/remove models as needed
)
```

## Tasks

Training tasks are defined in the `task` array:

```bash
task=(
    "flanV2_cot_10000"
    "magicoder-oss-instruct-10k"
    "healthcaremagic-10k"
    "openfunction_train"
)
```

## Evaluation Process

The script performs two main steps:
1. Response generation using `gen_responses_human-eval.py` for each model
2. Final evaluation using `run_human-eval_multiprocess.py` after all generations

## Hyperparameters

Various hyperparameters can be configured:
- Seeds: Controlled via the `seed` array (default: 42, 1024, 48763)
- Checkpoints: Determined by task (default: 500 for most tasks, 200 for openfunction_train)
- Weight decay values: Set in the `wd` array
- Dropout rates: Set in the `dropout` array
- Learning rates: Set in the `learning_rate` array
- Merge weights: Configured in the `wn` arrays depending on the method

## Note

Make sure you have sufficient disk space and GPU memory available for running evaluations on large models. For best performance, we recommend using GPUs with at least 24GB of VRAM.
