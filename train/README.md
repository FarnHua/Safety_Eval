# Model Training Configuration

This directory contains configuration files and scripts for training models using LLaMA-Factory.

## Setup Instructions

1. Clone the LLaMA-Factory repository:
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
```

2. Copy the files from this directory into the LLaMA-Factory directory:
```bash
cp *.yaml /path/to/LLaMA-Factory/
cp update_yaml.py /path/to/LLaMA-Factory/
cp run.sh /path/to/LLaMA-Factory/
```

3. Create and activate a conda environment for LLaMA-Factory:
```bash
conda create -n llama_factory python=3.10
conda activate llama_factory
```

4. Install LLaMA-Factory:
```bash
cd /path/to/LLaMA-Factory
pip install -e .
```

## Configuration Files

- `llama3_lora_sft.yaml`: Configuration for fine-tuning Llama 3 models
- `qwen2_lora_sft.yaml`: Configuration for fine-tuning Qwen 2 models
- `gemma_lora_sft.yaml`: Configuration for fine-tuning Gemma models

## Running Training

To start training, navigate to the LLaMA-Factory directory and run:

```bash
bash run.sh
```

You can customize the training parameters by editing `run.sh` or using the `update_yaml.py` script to modify configuration files.

## Customization

The `update_yaml.py` script allows you to programmatically modify YAML configuration files:

```bash
python update_yaml.py input.yaml --exp_name experiment_name --output_dir saves/path --learning_rate 5e-5
```

This will create a new YAML file with updated parameters.
