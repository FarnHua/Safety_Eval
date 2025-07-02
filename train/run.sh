#!/bin/bash
set -euo pipefail


source /home/farnhua/.bashrc
## you should use install llama_factory in your conda environment
mamba activate llama_factory 

export PYTHONWARNINGS="ignore::FutureWarning"
# export HUGGINGFACE_HUB_CACHE=""
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

lr=1.0e-4
dataset=(
    # "flanV2_cot_10000"
    # "magicoder-oss-instruct-10k"
    "healthcaremagic-10k"
    # "openfunction_train"
)

save_name=(
    # "llama3-8b-instruct_flanV2_cot_lr50e-5"
    "llama3-8b-instruct_healthcaremagic-10k"
    # "llama3-8b-instruct_magicoder-oss-instruct-10k_lr50e-5"
    # "llama3-8b-instruct_openfunction_train_lr50e-5"
)

batch_size=(
    4
    16
    32
)

seed=(
    42
    1024
    48763
)


yaml_file="llama3_lora_sft.yaml"
# Run the training script
for i in ${!dataset[@]}; do
    for j in ${!seed[@]}; do
        echo "Running ${dataset[$i]}"
        sn=${save_name[$i]}_seed-${seed[$j]}
        
        ## if dataset is openfunction_train, set step=20, otherwise set step=50
        if [ ${dataset[$i]} == "openfunction_train" ]; then
            eval_steps=200
            save_steps=20
        else
            eval_steps=500
            save_steps=50
        fi

        # Extract the base name of the yaml_file
        yaml_base=$(basename "$yaml_file" .yaml)
        
        # Construct the output directory
        output_dir="saves/${sn}/lora/sft"
        
        # Modify settings in the YAML file
        python3 update_yaml.py $yaml_file \
        --exp_name ${sn} \
        --dataset ${dataset[$i]} \
        --output_dir $output_dir \
        --learning_rate $lr \
        --eval_steps $eval_steps \
        --save_steps $save_steps \
        --seed ${seed[$j]} \
        --report_to wandb \
        --run_name ${sn} \
        --per_device_train_batch_size 8 \
        
        # Find the updated YAML file
        updated_yaml=$(find "$output_dir" -name "${yaml_base}_${sn}.yaml")
        
        if [ -z "$updated_yaml" ]; then
            echo "Error: Updated YAML file not found"
            exit 1
        fi
        
        # Run the training script with the dynamically found YAML file
        llamafactory-cli train "$updated_yaml"
    
    done
    
done
