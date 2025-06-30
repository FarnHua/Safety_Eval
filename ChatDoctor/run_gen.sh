#!/bin/bash
set -euo pipefail
source /home/farnhua/.bashrc
mamba activate safety_eval

export PYTHONWARNINGS="ignore::FutureWarning"
# export HUGGINGFACE_HUB_CACHE=""
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


base_model=(
    # "gemma2-2b-it"
    # "gemma2-9b-it"
    "llama3-8b-instruct"
    # "qwen25-15b-instruct"
    # "qwen25-3b-instruct"
    # "qwen25-7b-instruct"
)
seeds=(42 1024 48763)
task=(
    "healthcaremagic-10k"
    # "flanV2_cot_10000"
    # "magicoder-oss-instruct-10k"
    # "openfunction_train"
)

BASE_MODEL=true
ORIGINAL_FT=false
WEIGHT_DECAY=false
LINEAR_MERGE=false
SLERP=false
DARE=false
DROPOUT=false


# Helper function to get default checkpoints based on task
get_checkpoints() {
    local task=$1
    if [ "$task" = "openfunction_train" ]; then
        echo "200"
    else
        echo "500"
    fi
}


# Helper function to evaluate model
evaluate_model() {
    local model_path=$1
    local model_name=$2
    local is_gemma=$3
    local type=${4:-"original_ft"}
    local use_peft=${5:-true}

    local common_args="--model_path ${model_path} --model_name ${model_name}"
    if [ "$use_peft" = true ]; then
        common_args="${common_args} --use_peft True"
    fi
    
    if [ "$is_gemma" = true ]; then
        python3 evaluation.py ${common_args} \
            --add_sys False \
            --gen_result_dir generation_results_${type} \
            --score_result_dir score_results_${type} \
            --max_num_seqs 128
    else
        python3 evaluation.py ${common_args} \
            --gen_result_dir generation_results_${type} \
            --score_result_dir score_results_${type} \
            --max_num_seqs 128
    fi

    python3 get_score.py \
        --eval_file generation_results_${type}/${model_name}_generations.json \
        --score_result_dir score_results_${type} \
        --type ${type}
}

if [ "$BASE_MODEL" = true ]; then
    for n in "${base_model[@]}"; do
        echo "Evaluating base model: $n"
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
            is_gemma=true
        else
            is_gemma=false
        fi

        evaluate_model "$model_path" "$n" "$is_gemma" "base-model" false
    done
fi

############################################ original ft ############################################
if [ "$ORIGINAL_FT" = true ]; then
for n in ${base_model[@]}; do
    for t in ${task[@]}; do
        checkpoints=($(get_checkpoints "$t"))
        for s in ${seeds[@]}; do
            for c in ${checkpoints[@]}; do
                model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_seed-${s}/lora/sft/checkpoint-${c}"
                model_name="${n}_${t}_seed-${s}_ckpt-${c}"
                
                is_gemma=false
                if [[ "$n" =~ ^gemma ]]; then
                    is_gemma=true
                fi
                
                evaluate_model "$model_path" "$model_name" "$is_gemma"
            done
        done
    done
done

fi

############################################ weight decay ############################################
if [ "$WEIGHT_DECAY" = true ]; then

# wn=(01 02 03 04 05)
wn=(03)
for b in ${base_model[@]}; do
    for t in ${task[@]}; do
        checkpoints=($(get_checkpoints "$t"))
        for s in ${seeds[@]}; do
            for w in ${wn[@]}; do
                for c in ${checkpoints[@]}; do
                    model_path="/livingrooms/farnhua/LLaMA-Factory/saves/${b}_${t}_wd-${w}_seed-${s}/lora/sft/checkpoint-${c}"
                    model_name="${b}_${t}_wd-${w}_seed-${s}_ckpt-${c}"
                    
                    is_gemma=false
                    if [ "$b" = "gemma-2b-it" ] || [ "$b" = "gemma-7b-it" ]; then
                        is_gemma=true
                    fi
                    
                    evaluate_model "$model_path" "$model_name" "$is_gemma" "weight_decay"
                done
            done
        done
    done
done

fi


############################################ linear merge ############################################
if [ "$LINEAR_MERGE" = true ]; then
    # wn=("06") for llama3-8b-instruct
    # wn=("09") # for gemma2-2b-it
    wn=(01 02 03 04 05 06 07 08 09)

    for n in ${base_model[@]}; do
        for t in ${task[@]}; do
            checkpoints=($(get_checkpoints "$t"))
            for s in ${seeds[@]}; do
                for w in ${wn[@]}; do
                    for c in ${checkpoints[@]}; do
                        model_path=/livingrooms/farnhua/SafeIT/task_vector/results/merged_models/${n}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w}
                        model_name=${n}_${t}_seed-${s}_ckpt-${c}_merged_base_w1-${w}
                        
                        is_gemma=false
                        if [[ "$n" =~ ^gemma ]]; then
                            is_gemma=true
                        fi
                        
                        evaluate_model "$model_path" "$model_name" "$is_gemma" "linear-merge" false
                    done
                done
            done
        done
    done
fi


############################################ slerp ############################################
if [ "$SLERP" = true ]; then
    wn=(01 02 03 04 05 06 07 08 09)

    for n in ${base_model[@]}; do
        for t in ${task[@]}; do
            checkpoints=($(get_checkpoints "$t"))
            for s in ${seeds[@]}; do
                for w in ${wn[@]}; do
                    for c in ${checkpoints[@]}; do
                        model_path="/livingrooms/farnhua/mergekit/merged_models/${n}_${t}_seed-${s}_ckpt-${c}_slerp_t-${w}"
                        model_name="${n}_${t}_seed-${s}_ckpt-${c}_slerp_t-${w}"
                        
                        is_gemma=false
                        if [[ "$n" =~ ^gemma ]]; then
                            is_gemma=true
                        fi
                        
                        evaluate_model "$model_path" "$model_name" "$is_gemma" "slerp" false
                    done
                done
            done
        done
    done
fi


############################################ dare ############################################
if [ "$DARE" = true ]; then
    wn=(01 02 03 04 05 06 07 08 09)
    # echo "Starting dare model merging evaluation..."
    for n in ${base_model[@]}; do
        for t in ${task[@]}; do
            checkpoints=($(get_checkpoints "$t"))
            for s in ${seeds[@]}; do
                for w in ${wn[@]}; do
                    for c in ${checkpoints[@]}; do
                        model_path="/livingrooms/farnhua/mergekit/merged_models/${n}_${t}_seed-${s}_ckpt-${c}_dare_w-${w}"
                        model_name="${n}_${t}_seed-${s}_ckpt-${c}_dare_w-${w}"
                        
                        is_gemma=false
                        if [[ "$n" =~ ^gemma ]]; then
                            is_gemma=true
                        fi
                        
                        evaluate_model "$model_path" "$model_name" "$is_gemma" "dare" false
                    done
                done
            done
        done
    done
fi

## ############################################ dropout ############################################
if [ "$DROPOUT" = true ]; then
    # dropout=(01 02 03 04 05)
    dropout=(03)

    for n in ${base_model[@]}; do
        for t in ${task[@]}; do
            checkpoints=($(get_checkpoints "$t"))
            for s in ${seeds[@]}; do
                for d in ${dropout[@]}; do
                    for c in ${checkpoints[@]}; do
                        model_path=/livingrooms/farnhua/LLaMA-Factory/saves/${n}_${t}_dr${d}_seed-${s}/lora/sft/checkpoint-${c}
                        model_name=${n}_${t}_dr${d}_seed-${s}_ckpt-${c}
                        
                        is_gemma=false
                        if [[ "$n" =~ ^gemma ]]; then
                            is_gemma=true
                        fi
                        
                        evaluate_model "$model_path" "$model_name" "$is_gemma" "dropout"
                    done
                done
            done
        done
    done
fi