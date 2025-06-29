# Safety Evaluation Scripts

This repository contains scripts for evaluating various language models on safety benchmarks.

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
```

## Running Evaluations

Two main scripts are provided:

### 1. AdvBench Evaluation (`run_advbench.sh`)

Evaluates models on the AdvBench safety benchmark:

```bash
bash run_advbench.sh
```

### 2. HEx-PHI Evaluation (`run_hex-phi.sh`)

Evaluates models on the HEx-PHI  benchmark:

```bash
bash run_hex-phi.sh
```

## Configuration Options

Both scripts support various evaluation modes controlled by flags at the top of each script:

- `RUN_ORIGINAL_FT`: Evaluate original fine-tuned models
- `RUN_MODEL_STOCK`: Evaluate model stock versions
- `RUN_BASE_MODEL`: Evaluate base models
- `RUN_LINEAR_MERGE`: Evaluate linear merged models
- `RUN_WEIGHT_DECAY`: Evaluate weight decay variants
- `RUN_DROPOUT`: Evaluate dropout variants
- `RUN_SLERP`: Evaluate SLERP merged models
- `RUN_DARE`: Evaluate DARE merged models



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

Models can be configured in the `base_models` array at the top of each script:

```bash
base_models=(
    "llama3-8b-instruct"
    "gemma2-9b-it"
    "qwen25-15b-instruct"
    # Add/remove models as needed
)
```

## Tasks

Training tasks are defined in the `tasks` array:

```bash
tasks=(
    "healthcaremagic-10k"
    "flanV2_cot_10000"
    "magicoder-oss-instruct-10k"
    "openfunction_train"
)
```

## Output

Results will be saved in:
- AdvBench results: `../advbench_gen_results/`
- HexaGon-Phi results: `../hex-phi_gen_results/`
