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
    # "qwen25-15b-instruct"
    # "qwen25-3b-instruct"
    # "qwen25-7b-instruct"
)
tasks=(
    "healthcaremagic-10k"
    "flanV2_cot_10000"
    "magicoder-oss-instruct-10k"
    "openfunction_train"
)

# Control flags for different stages
RUN_ORIGINAL_FT=false
RUN_BASE_MODEL=true
RUN_LINEAR_MERGE=false
RUN_WEIGHT_DECAY=false
RUN_SLERP=false
RUN_DARE=false
RUN_DROPOUT=false

# Environment setup
source /home/farnhua/.bashrc
mamba activate safety_eval

export PYTHONWARNINGS="ignore::FutureWarning"
## set your huggingface cache directory
# export HUGGINGFACE_HUB_CACHE=""
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Common variables
eval_dataset="advbench"
seed=(42 1024 48763)
model_path=()
model_name=()
DO_INFERENCE=true
DO_EVALUATION=true
EVAL_TYPE="wildguard"

# Function to run evaluation
run_evaluation() {
    local model_path=$1
    local model_name=$2
    local use_peft=${3:-false}
    local add_sys=${4:-true}
    local eval_dataset=${5:-"advbench"}
    local max_num_seqs=${6:-4}
    local eval_type=${7:-""}

    echo "Running inference for: $model_name"
    
    local common_args=(
        --seed 42
        --model_path "$model_path"
        --model_name "$model_name"
        --eval_dataset "$eval_dataset"
        --max_num_seqs "$max_num_seqs"
    )

    if [ "$use_peft" = true ]; then
        common_args+=(--use_peft)
    fi

    if [ "$add_sys" = true ]; then
        common_args+=(--add_sys)
    else
        common_args+=(--add_sys False)
    fi

    local evaluation_args=(
        --file_path "../${eval_dataset}_gen_results/${model_name}_${eval_dataset}.jsonl"
        --eval_type "${EVAL_TYPE}"
        --eval_dataset "${eval_dataset}"
    )

    if [ "$DO_INFERENCE" = true ]; then
        python3 inference.py "${common_args[@]}"
    fi
    if [ "$DO_EVALUATION" = true ]; then
        python3 evaluation.py "${evaluation_args[@]}"
    fi
}

# Helper function to get default checkpoints based on task
get_checkpoints() {
    local task=$1
    if [ "$task" = "openfunction_train" ]; then
        echo "200"
    else
        echo "500"
    fi
}

################################### for original fine-tuning ####################################
if [ "$RUN_ORIGINAL_FT" = true ]; then
    for t in "${tasks[@]}"; do
        ckpt=($(get_checkpoints "$t"))
        
        for n in "${base_models[@]}"; do
            for s in "${seed[@]}"; do
                for c in "${ckpt[@]}"; do
                    echo "Evaluating original ft model: $n, task: $t, seed: $s, checkpoint: $c"
                    model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_seed-${s}/lora/sft/checkpoint-${c}"
                    model_name="${n}_${t}_seed-${s}_ckpt-${c}"
                    
                    if [[ "$n" =~ ^gemma ]]; then
                        run_evaluation "$model_path" "$model_name" true false
                    else
                        run_evaluation "$model_path" "$model_name" true true
                    fi
                done
            done
        done
    done
fi

################################### for model stock ####################################
if [ "$RUN_MODEL_STOCK" = true ]; then
    echo "Starting model stock evaluation..."
    
    for t in "${tasks[@]}"; do
        ckpt=($(get_checkpoints "$t"))
        
        for n in "${base_models[@]}"; do
            for c in "${ckpt[@]}"; do
                model_path="/livingrooms/farnhua/mergekit/merged_models/${n}_${t}_ckpt-${c}_average-modelStock"
                model_name="${n}_${t}_ckpt-${c}_average-modelStock"
                
                if [[ "$n" =~ ^gemma ]]; then
                    run_evaluation "$model_path" "$model_name" false false
                else
                    run_evaluation "$model_path" "$model_name" false true
                fi
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
        
        if [[ "$n" =~ ^gemma ]]; then
            run_evaluation "$model_path" "$n" false false
        else
            run_evaluation "$model_path" "$n" false true
        fi
    done
fi

################################### for linear merge ####################################
if [ "$RUN_LINEAR_MERGE" = true ]; then
    echo "Starting linear model merging evaluation..."
    
    # w1=(06)
    # w1=(01 02 03 04 05 06 07 08 09)
    
    for t in "${tasks[@]}"; do
        ckpt=($(get_checkpoints "$t"))
        
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
            
            for s in "${seed[@]}"; do
                for c in "${ckpt[@]}"; do
                    for w in "${w1[@]}"; do
                        echo "Evaluating model: $n, task: $t, seed: $s, w1: $w, checkpoint: $c"
                        model_path="/livingrooms/farnhua/SafeIT/task_vector/results/merged_models/${n}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w}"
                        model_name="${n}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w}"
                        
                        if [[ "$n" =~ ^gemma ]]; then
                            run_evaluation "$model_path" "$model_name" false false
                        else
                            run_evaluation "$model_path" "$model_name" false true
                        fi
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
    # wn=(01 02 03 04 05 06 07 08 09)
    
    for t in "${tasks[@]}"; do
        ckpt=($(get_checkpoints "$t"))
        
        for n in "${base_models[@]}"; do
            for s in "${seed[@]}"; do
                for c in "${ckpt[@]}"; do
                    for w in "${wn[@]}"; do
                        echo "Evaluating model: $n, task: $t, seed: $s, w1: $w, checkpoint: $c"
                        model_path="/livingrooms/farnhua/mergekit/merged_models/${n}_${t}_seed-${s}_ckpt-${c}_slerp_t-${w}"
                        model_name="${n}_${t}_seed-${s}_ckpt-${c}_slerp_t-${w}"
                        
                        if [[ "$n" =~ ^gemma ]]; then
                            run_evaluation "$model_path" "$model_name" false false
                        else
                            run_evaluation "$model_path" "$model_name" false true
                        fi
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
    # wn=(01 02 03 04 05 06 07 08 09)
    
    for t in "${tasks[@]}"; do
        ckpt=($(get_checkpoints "$t"))
        
        for n in "${base_models[@]}"; do
            for s in "${seed[@]}"; do
                for c in "${ckpt[@]}"; do
                    for w in "${wn[@]}"; do
                        echo "Evaluating model: $n, task: $t, seed: $s, w1: $w, checkpoint: $c"
                        model_path="/livingrooms/farnhua/mergekit/merged_models/${n}_${t}_seed-${s}_ckpt-${c}_dare_w-${w}"
                        model_name="${n}_${t}_seed-${s}_ckpt-${c}_dare_w-${w}"
                        
                        if [[ "$n" =~ ^gemma ]]; then
                            run_evaluation "$model_path" "$model_name" false false
                        else
                            run_evaluation "$model_path" "$model_name" false true
                        fi
                    done
                done
            done
        done
    done
fi


################################### for weight decay ####################################
if [ "$RUN_WEIGHT_DECAY" = true ]; then
    echo "Starting weight decay evaluation..."
    
    wd=(03)
    # wn=05
    
    for t in "${tasks[@]}"; do
        ckpt=($(get_checkpoints "$t"))
        
        for n in "${base_models[@]}"; do
            for s in "${seed[@]}"; do
                for c in "${ckpt[@]}"; do
                    for w in "${wd[@]}"; do
                        echo "Evaluating task: $t, seed: $s, wd: $w, checkpoint: $c"
                        model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_wd-${w}_seed-${s}/lora/sft/checkpoint-${c}"
                        model_name="${n}_${t}_wd-${w}_seed-${s}_ckpt-${c}"
                        
                        if [[ "$n" =~ ^gemma ]]; then
                            run_evaluation "$model_path" "$model_name" true false
                        else
                            run_evaluation "$model_path" "$model_name" true true
                        fi
                    done
                done
            done
        done
    done
fi

## ################################### for dropout ####################################
if [ "$RUN_DROPOUT" = true ]; then
    echo "Starting dropout evaluation..."
    # /livingrooms/farnhua/LLaMA-Factory/saves/llama3-8b-instruct_flanV2_cot_10000_dr01_seed-42
    # dropout=(01 02 03 04 05)
    dropout=(03)
    
    for n in "${base_models[@]}"; do
        for t in "${tasks[@]}"; do
            checkpoints=($(get_checkpoints "$t"))
            for s in "${seed[@]}"; do
                for d in "${dropout[@]}"; do
                    for c in "${checkpoints[@]}"; do
                        echo "Evaluating task: $t, seed: $s, dropout: $d, checkpoint: $c"
                        model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_dr${d}_seed-${s}/lora/sft/checkpoint-${c}"
                        model_name="${n}_${t}_dr${d}_seed-${s}_ckpt-${c}"
                        
                        if [[ "$n" =~ ^gemma ]]; then
                            run_evaluation "$model_path" "$model_name" true false
                        else
                            run_evaluation "$model_path" "$model_name" true true
                        fi
                    done
                done
            done
        done
    done
fi

