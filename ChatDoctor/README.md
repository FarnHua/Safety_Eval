# ChatDoctor Evaluation Scripts

This directory contains scripts for evaluating language models on medical conversation datasets.

## Prerequisites

- Linux environment
- Mamba/Conda environment with VLLM installed
- Python 3.x
- Hugging Face account and access to model weights

## Environment Setup

1. Set required environment variables:
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
bash run_gen.sh
```

## Configuration Options

The script supports various evaluation modes controlled by flags at the top:

- `BASE_MODEL`: Evaluate base models
- `ORIGINAL_FT`: Evaluate original fine-tuned models
- `WEIGHT_DECAY`: Evaluate weight decay variants
- `LINEAR_MERGE`: Evaluate linear merged models
- `SLERP`: Evaluate SLERP merged models
- `DARE`: Evaluate DARE merged models
- `DROPOUT`: Evaluate dropout variants

Set the desired flags to `true` in the script before running.

## Required Arguments

For each evaluation, you need to specify:
- `model_path`: Path to the model or Hugging Face model ID
- `model_name`: Name for saving results
- Method-specific hyperparameters (merging factor, weight decay, or dropout rate)

## Hyperparameters

Various hyperparameters can be configured:
- Seeds: Controlled via the `seed` array
- Checkpoints: Set in the `checkpoints` array or determined by task
- Merge weights: Configured in the `w1` or `wn` arrays depending on the method
- Dropout rates: Set in the `dropout` array
- Weight decay values: Set in the `wd` array

## Model Selection

Models can be configured in the `base_model` array at the top of the script:

```bash
base_model=(
    "gemma2-2b-it"
    "gemma2-9b-it"
    "llama3-8b-instruct"
    "qwen25-7b-instruct"
    # Add/remove models as needed
)
```

## Tasks

Training tasks are defined in the `task` array:

```bash
task=(
    "healthcaremagic-10k"
    "flanV2_cot_10000"
    "magicoder-oss-instruct-10k"
    "openfunction_train"
)
```

## Evaluation Process

The script performs two main steps:
1. Response generation and initial evaluation using `evaluation.py`
2. Score calculation using `get_score.py`

Results will be saved in different directories based on the evaluation type:
- Base model results: `generation_results_base-model/` and `score_results_base-model/`
- Original fine-tuned results: `generation_results_original_ft/` and `score_results_original_ft/`
- Weight decay results: `generation_results_weight_decay/` and `score_results_weight_decay/`
- Linear merge results: `generation_results_linear-merge/` and `score_results_linear-merge/`
- SLERP results: `generation_results_slerp/` and `score_results_slerp/`
- DARE results: `generation_results_dare/` and `score_results_dare/`
- Dropout results: `generation_results_dropout/` and `score_results_dropout/`

## Hyperparameters

Various hyperparameters can be configured:
- Seeds: Controlled via the `seeds` array (default: 42, 1024, 48763)
- Checkpoints: Determined by task (default: 500 for most tasks, 200 for openfunction_train)
- Merge weights: Configured in the `wn` arrays depending on the method
- Dropout rates: Set in the `dropout` array
- Weight decay values: Set in the `wn` array for weight decay section

## Note

Make sure you have sufficient disk space and GPU memory available for running evaluations on large models. For best performance, we recommend using GPUs with at least 24GB of VRAM.
