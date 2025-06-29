#!/bin/bash
set -euo pipefail

# Common configuration section
# Models and tasks to run - specify which ones to use at runtime
base_models=(
    "llama3-8b-instruct"
    # "llama31-8b-instruct"
    # "gemma-2b-it"
    # deepseek-r1-distill-llama-8b
    # gemma2-2b-it
    # "gemma2-9b-it"
    # "qwen25-3b-instruct"
    # "qwen25-7b-instruct"
)
task=(
    # "healthcaremagic-10k"
    # "flanV2_cot_10000"
    # "magicoder-oss-instruct-10k"
    "openfunction_train"
)

# Control flags for different stages
RUN_ORIGINAL_FT=false
RUN_MODEL_STOCK=false
RUN_BASE_MODEL=true
RUN_LINEAR_MERGE=false
RUN_WEIGHT_DECAY=false
ADD_SAFETY_DATA=false
RUN_SLERP=false
RUN_DARE=false
RUN_DIFF_BATCH_SIZE=false
RUN_DIFF_LR=false
RUN_SAFETY_LORA=false
RUN_DROPOUT=false

# Environment setup
source /home/farnhua/.bashrc
mamba activate vllm_env

export PYTHONWARNINGS="ignore::FutureWarning"
# export HUGGINGFACE_HUB_CACHE=""
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Common variables
seed=(42 1024 48763)
checkpoints=(200)
model_path=()
model_name=()


# Helper function to get default checkpoints based on task
get_checkpoints() {
    local task=$1
    if [ "$task" = "openfunction_train" ]; then
        echo "200"
    else
        echo "500"
    fi
}
# Function to run evaluation (both generation and scoring)
run_evaluation() {
    local model_path=$1
    local model_name=$2
    local use_peft=${3:-false}
    local add_sys=${4:-true}

    echo "Running generation for: $model_name"
    
    local gen_args=(
        --seed 42
        --model_path "$model_path"
        --model_name "$model_name"
    )

    if [ "$use_peft" = true ]; then
        gen_args+=(--use_peft)
    fi

    ## if gemma in model name, set add_sys to false
    if [[ "$model_name" == *"gemma"* ]]; then
        add_sys=false
    fi

    if [ "$add_sys" = false ]; then
        gen_args+=(--add_sys False)
    fi

    # Run generation
    python3 gen_responses.py "${gen_args[@]}"

    # Run scoring
    python3 get_score.py --eval_file "./generation_results/${model_name}_generations.json"
}

################################### for original fine-tuning ####################################
if [ "$RUN_ORIGINAL_FT" = true ]; then
    echo "Starting original fine-tuning evaluation..."
    
    for n in "${base_models[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for c in $(get_checkpoints "$t"); do
                    echo "Evaluating original ft model: $n, task: $t, seed: $s, checkpoint: $c"
                    model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_seed-${s}/lora/sft/checkpoint-${c}"
                    model_name="${n}_${t}_seed-${s}_ckpt-${c}"
                    run_evaluation "$model_path" "$model_name" true true
                done
            done
        done
    done
fi


################################### for base model ####################################
if [ "$RUN_BASE_MODEL" = true ]; then
    echo "Starting base model evaluation..."
    
    for n in "${base_models[@]}"; do
        case "$n" in
            "llama3-8b-instruct")
                model_path="meta-llama/Meta-Llama-3-8B-Instruct"
                ;;
            "llama31-8b-instruct")
                model_path="meta-llama/Llama-3.1-8B-Instruct"
                ;;
            "llama32-3b-instruct")
                model_path="meta-llama/Llama-3.2-3B-Instruct"
                ;;
            "gemma-7b-it")
                model_path="google/gemma-7b-it"
                ;;
            "gemma-2b-it")
                model_path="google/gemma-2b-it"
                ;;
            "deepseek-r1-distill-llama-8b")
                model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                ;;
            "gemma2-2b-it")
                model_path="google/gemma-2-2b-it"
                ;;
            "gemma2-9b-it")
                model_path="google/gemma-2-9b-it"
                ;;
            "qwen25-15b-instruct")
                model_path="Qwen/Qwen2.5-1.5B-Instruct"
                ;;
            "qwen25-3b-instruct")
                model_path="Qwen/Qwen2.5-3B-Instruct"
                ;;
            "qwen25-7b-instruct")
                model_path="Qwen/Qwen2.5-7B-Instruct"
                ;;
            *)
                echo "Unknown model: $n"
                exit 1
                ;;
        esac
        
        run_evaluation "$model_path" "$n" false true
    done
fi

################################### for linear merge ####################################
if [ "$RUN_LINEAR_MERGE" = true ]; then
    echo "Starting linear model merging evaluation..."
    
    # w1=(06) # for llama3-8b-instruct
    # w1=(07) # for gemma2-2b-it
    # w1=(01 02 03 04 05 06 07 08 09)
    
    for n in "${base_models[@]}"; do
        if [ ${n} == "qwen25-15b-instruct" ]; then
                w1=(08)
            elif [ ${n} == "qwen25-3b-instruct" ]; then
                w1=(09)
            elif [ ${n} == "qwen25-7b-instruct" ]; then
                w1=(07)
            elif [ ${n} == "gemma2-2b-it" ]; then
                w1=(09)
            elif [ ${n} == "gemma2-9b-it" ]; then
                w1=(06)
            elif [ ${n} == "llama3-8b-instruct" ]; then
                w1=(06)
            fi 
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for c in $(get_checkpoints "$t"); do
                    for w in "${w1[@]}"; do
                        echo "Evaluating original ft model: $n, task: $t, seed: $s, checkpoint: $c"
                        model_path="/livingrooms/farnhua/SafeIT/task_vector/results/merged_models/${n}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w}"
                        model_name="${n}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w}"
                        run_evaluation "$model_path" "$model_name" false true
                    done
                done
            done
        done
    done
fi

################################### for slerp model merging ####################################
if [ "$RUN_SLERP" = true ]; then
    echo "Starting slerp model merging evaluation..."
    
    wn=(07)
    
    for n in "${base_models[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for c in $(get_checkpoints "$t"); do
                    for w in "${wn[@]}"; do
                        echo "Evaluating model: $n, seed: $s, w1: $w, checkpoint: $c"
                        model_path="/livingrooms/farnhua/mergekit/merged_models/${n}_openfunction_train_seed-${s}_ckpt-${c}_slerp_t-${w}"
                        model_name="${n}_openfunction_train_seed-${s}_ckpt-${c}_slerp_t-${w}"
                        run_evaluation "$model_path" "$model_name" false true
                    done
                done
            done
        done
    done
fi

################################### for dare model merging ####################################
if [ "$RUN_DARE" = true ]; then
    echo "Starting dare model merging evaluation..."
    
    wn=(07)
    
    for n in "${base_models[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for c in $(get_checkpoints "$t"); do
                    for w in "${wn[@]}"; do
                        echo "Evaluating model: $n, seed: $s, w1: $w, checkpoint: $c"
                        model_path="/livingrooms/farnhua/mergekit/merged_models/${n}_openfunction_train_seed-${s}_ckpt-${c}_dare_w-${w}"
                        model_name="${n}_openfunction_train_seed-${s}_ckpt-${c}_dare_w-${w}"
                        run_evaluation "$model_path" "$model_name" false true
                    done
                done
            done
        done
    done
fi



######################DROPOUT##########################
if [ "$RUN_DROPOUT" = true ]; then
    echo "Starting dropout evaluation..."
    
    # dropout=(01 02 03 04 05)
    dropout=(03)
    
    for n in "${base_models[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for d in "${dropout[@]}"; do
                    for c in $(get_checkpoints "$t"); do
                        echo "Evaluating model: $n, task: $t, seed: $s, dropout: $d, checkpoint: $c"
                        model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_dr${d}_seed-${s}/lora/sft/checkpoint-${c}"
                        model_name="${n}_${t}_dr${d}_seed-${s}_ckpt-${c}"
                        run_evaluation "$model_path" "$model_name" true true
                    done
                done
            done
        done
    done
fi

################################### for weight decay ####################################
if [ "$RUN_WEIGHT_DECAY" = true ]; then
    echo "Starting weight decay evaluation..."
    
    # wd=(01 02 03 04 05)
    wd=(03)
    
    for n in "${base_models[@]}"; do
        for t in "${task[@]}"; do
            for s in "${seed[@]}"; do
                for w in "${wd[@]}"; do
                    for c in $(get_checkpoints "$t"); do
                        echo "Evaluating model: $n, task: $t, seed: $s, wd: $w, checkpoint: $c"
                        model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_wd-${w}_seed-${s}/lora/sft/checkpoint-${c}"
                        model_name="${n}_${t}_wd-${w}_seed-${s}_ckpt-${c}"
                        run_evaluation "$model_path" "$model_name" true true
                    done
                done
            done
        done
    done
fi
