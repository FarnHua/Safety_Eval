#!/bin/bash
set -euo pipefail

# ============= CONFIGURATION SECTION =============
# Control flags for different evaluation types
RUN_ORIGINAL_FT=false
RUN_WEIGHT_DECAY=false
RUN_DROPOUT=false
RUN_LINEAR_MERGE=false
RUN_SLERP=false
RUN_DARE=false
RUN_BASE_MODEL=true

# Models and tasks to evaluate
base_model=(
    "llama3-8b-instruct"
    # "gemma2-2b-it"
    # "qwen25-7b-instruct"
    # "qwen25-15b-instruct"
    # "qwen25-3b-instruct"
    # "gemma2-9b-it"
)
    
task=(
    # "flanV2_cot_10000"
    "magicoder-oss-instruct-10k"
    # "healthcaremagic-10k"
    # "openfunction_train"
)

# Common parameters
seed=(42 1024 48763)
model_path=()
model_name=()

# Environment setup
source /home/farnhua/.bashrc
mamba activate safety_eval
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONWARNINGS="ignore::FutureWarning"
# export HUGGINGFACE_HUB_CACHE=""
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ============= HELPER FUNCTIONS =============
# Get appropriate checkpoint based on task
get_checkpoints() {
    local task=$1
    
    if [ "$task" = "openfunction_train" ]; then
        echo "200"
    else
        echo "500"
    fi
}

# Run model evaluation
run_evaluation() {
    echo "Evaluating models: ${#model_path[@]} total models"
    for m in ${!model_path[@]}; do
        echo "Evaluating: ${model_name[$m]}"
        if [[ "${model_name[$m]}" == *"gemma"* ]]; then
            add_sys="--add_sys False"
        else
            add_sys=""
        fi
        
        if [[ "${model_path[$m]}" == *"/lora/"* ]]; then
            use_peft="--use_peft"
        else
            use_peft=""
        fi
        
        python3 gen_responses_human-eval.py \
            --seed 42 \
            --model_path "${model_path[$m]}" \
            --model_name "${model_name[$m]}" \
            $use_peft $add_sys
    done
}

# Clear arrays before adding new models
clear_models() {
    model_path=()
    model_name=()
}

# ============= EVALUATION FUNCTIONS =============

# Original fine-tuning evaluation
run_original_ft() {
    echo "Collecting original fine-tuned models..."
    clear_models
    
    for b in "${base_model[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                checkpoints=($(get_checkpoints "$t"))
                for c in ${checkpoints[@]}; do
                    model_path+=(/livingrooms/farnhua/LLaMA-Factory/saves/${b}_${t}_seed-${s}/lora/sft/checkpoint-${c})
                    model_name+=("${b}_${t}_seed-${s}_ckpt-${c}")
                done
            done
        done
    done
}

# Weight decay evaluation
run_weight_decay() {
    echo "Collecting weight decay models..."
    clear_models
    
    # wd=(01 02 03 04 05)
    wd=(03)
    for b in "${base_model[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for w in "${wd[@]}"; do
                    checkpoints=($(get_checkpoints "$t"))
                    for c in ${checkpoints[@]}; do
                        model_path+=(/livingrooms/farnhua/LLaMA-Factory/saves/${b}_${t}_wd-${w}_seed-${s}/lora/sft/checkpoint-${c})
                        model_name+=("${b}_${t}_wd-${w}_seed-${s}_ckpt-${c}")
                    done
                done
            done
        done
    done
}

# Dropout evaluation
run_dropout() {
    echo "Collecting dropout models..."
    clear_models
    
    # dropout=(01 02 03 04 05)
    dropout=(03)
    for b in "${base_model[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for d in "${dropout[@]}"; do
                    checkpoints=($(get_checkpoints "$t"))
                    for c in ${checkpoints[@]}; do
                        model_path+=(/livingrooms/farnhua/LLaMA-Factory/saves/${b}_${t}_dr${d}_seed-${s}/lora/sft/checkpoint-${c})
                        model_name+=("${b}_${t}_dr${d}_seed-${s}_ckpt-${c}")
                    done
                done
            done
        done
    done
}

# Linear merge evaluation
run_linear_merge() {
    echo "Collecting linear merged models..."
    clear_models
    
    wn=(01 02 03 04 05 06 07 08 09)
    for b in "${base_model[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for w in "${wn[@]}"; do
                    checkpoints=($(get_checkpoints "$t"))
                    for c in ${checkpoints[@]}; do
                        model_path+=(/livingrooms/farnhua/SafeIT/task_vector/results/merged_models/${b}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w})
                        model_name+=("${b}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w}")
                    done
                done
            done
        done
    done
}

# SLERP model evaluation
run_slerp() {
    echo "Collecting SLERP merged models..."
    clear_models
    
    # Different models use different optimal t values
    # 0.8 for llama3, 0.9 for gemma2, 07 for qwen25
    wn=(07)
    for b in "${base_model[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for w in "${wn[@]}"; do
                    checkpoints=($(get_checkpoints "$t"))
                    for c in ${checkpoints[@]}; do
                        model_path+=("/livingrooms/farnhua/mergekit/merged_models/${b}_${t}_seed-${s}_ckpt-${c}_slerp_t-${w}")
                        model_name+=("${b}_${t}_seed-${s}_ckpt-${c}_slerp_t-${w}")
                    done
                done
            done
        done
    done
}

# DARE model evaluation
run_dare() {
    echo "Collecting DARE merged models..."
    clear_models
    
    # Different models use different optimal w values
    # 08 for llama3, 09 for gemma2
    wn=(07)
    for b in "${base_model[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for w in "${wn[@]}"; do
                    checkpoints=($(get_checkpoints "$t"))
                    for c in ${checkpoints[@]}; do
                        model_path+=("/livingrooms/farnhua/mergekit/merged_models/${b}_${t}_seed-${s}_ckpt-${c}_dare_w-${w}")
                        model_name+=("${b}_${t}_seed-${s}_ckpt-${c}_dare_w-${w}")
                    done
                done
            done
        done
    done
}

# Base model evaluation
run_base_model() {
    echo "Collecting base models..."
    clear_models
    
    for b in "${base_model[@]}"; do
        case "$b" in
            "llama3-8b-instruct")
                model_path+=("meta-llama/Meta-Llama-3-8B-Instruct")
                model_name+=("$b")
                ;;
            "llama31-8b-instruct")
                model_path+=("meta-llama/Llama-3.1-8B-Instruct")
                model_name+=("$b")
                ;;
            "gemma2-2b-it")
                model_path+=("google/gemma-2-2b-it")
                model_name+=("$b")
                ;;
            "gemma2-9b-it")
                model_path+=("google/gemma-2-9b-it")
                model_name+=("$b")
                ;;
            "qwen25-15b-instruct")
                model_path+=("Qwen/Qwen2.5-1.5B-Instruct")
                model_name+=("$b")
                ;;
            "qwen25-3b-instruct")
                model_path+=("Qwen/Qwen2.5-3B-Instruct")
                model_name+=("$b")
                ;;
            "qwen25-7b-instruct")
                model_path+=("Qwen/Qwen2.5-7B-Instruct")
                model_name+=("$b")
                ;;
            *)
                echo "Unknown base model: $b"
                ;;
        esac
    done
}

# ============= MAIN EXECUTION =============
# Run selected evaluation types
if [ "$RUN_ORIGINAL_FT" = true ]; then
    run_original_ft
    run_evaluation
fi

if [ "$RUN_WEIGHT_DECAY" = true ]; then
    run_weight_decay
    run_evaluation
fi

if [ "$RUN_DROPOUT" = true ]; then
    run_dropout
    run_evaluation
fi

if [ "$RUN_LINEAR_MERGE" = true ]; then
    run_linear_merge
    run_evaluation
fi

if [ "$RUN_SLERP" = true ]; then
    run_slerp
    run_evaluation
fi

if [ "$RUN_DARE" = true ]; then
    run_dare
    run_evaluation
fi

if [ "$RUN_BASE_MODEL" = true ]; then
    run_base_model
    run_evaluation
fi

# Run human-eval multiprocess after all evaluations
python3 run_human-eval_multiprocess.py