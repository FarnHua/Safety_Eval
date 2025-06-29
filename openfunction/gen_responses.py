import os
import json
import fire
from evaluate import load
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from peft import PeftConfig
import random
import re
from typing import List, Optional, TypedDict


class VLLMModel:
    def __init__(
        self, 
        model_path, 
        dtype="bfloat16",
        tp_size=1,
        max_num_seqs=16,
        max_tokens=512,
        temperature=0.0,
        sampling_params=None,
        use_peft=False,
        add_sys=True
    ):

        if sampling_params is None:
            sampling_params = SamplingParams(temperature=temperature)

        self.model_path = model_path
        self.dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        self.tensor_parallel_size = tp_size
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if mem < 24:
            self.max_num_seqs = 16
        else:
            self.max_num_seqs = 128
        self.max_tokens = max_tokens
        self.sampling_params = sampling_params
        if "gemma" in model_path:
            self.add_sys = False
        else:
            self.add_sys = add_sys
        
        
        if use_peft:
            ## find the base model path
            config = PeftConfig.from_pretrained(model_path)
            try :
                model_path = getattr(config, "base_model_name_or_path")
            except:
                assert False, "Please provide the base model path in the config file"
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model_path,
            tokenizer=model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            enable_lora=use_peft,
            gpu_memory_utilization=0.95,
            max_model_len=4096
        )
        
        # if self.add_sys:
        if "Qwen" in model_path and use_peft is False:
            self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Complete the following code and return only the completed code, without any explanations or additional text."
        else:
            self.system_prompt = "You are a helpful assistant. Complete the following code and return only the completed code, without any explanations or additional text."

        self.use_peft = use_peft
        
    def generate_prompt(self, system_prompt: str, user_message: str = None) -> str:
        
        if self.add_sys:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [
                {"role": "user", "content": self.system_prompt + "\n" + user_message},
            ]
        return messages
    
    
    def generate(self, prompts, return_only_responses=True):
        
        def rreplace(s, old, new, occurrence):
            li = s.rsplit(old, occurrence)
            return new.join(li)
        
        prompts = [
            self.tokenizer.apply_chat_template(
                self.generate_prompt(self.system_prompt, prompt), add_generation_prompt=True, tokenize=False
            )
            for prompt in prompts
        ]
        if self.tokenizer.bos_token:
            prompts = [
                prompt.replace(self.tokenizer.bos_token, "", 1) if self.tokenizer.bos_token in prompt else prompt
                for prompt in prompts
            ]
        if self.use_peft:
            outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, 
                                            lora_request=LoRARequest("lora", 1, self.model_path))
        else:
            outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        
        # import pdb; pdb.set_trace()
        
        responses = []
        token_num = []
        
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(generated_text.strip())
            token_num.append(len(output.outputs[0].token_ids))
        
        if return_only_responses:
            return responses
        else:
            return responses, token_num
        
    

def main(
    model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    model_name: str = "llama3-8b-instruct",
    use_peft: bool = False,
    seed: int = 42,
    eval_file: str = "./dataset/openfunction_test.json",
    add_sys: bool = True
):
    set_seed(seed)
    
    gen_result_dir = "./generation_results"    
    os.makedirs(gen_result_dir, exist_ok=True)


    with open(eval_file, 'r') as f:
        problems = json.load(f)

    inputs = [
        p['instruction'] for p in problems
    ]
    
    # ## generate responses
    # sampling_params = SamplingParams(max_tokens=1024, temperature=0.6, top_p=0.9, top_k=50)
    ## detect # of GPUs
    num_gpus = torch.cuda.device_count()
    llm = VLLMModel(model_path, use_peft=use_peft, tp_size=num_gpus, add_sys=add_sys)
    generations = llm.generate(inputs)
    
    
    with open(f"{gen_result_dir}/{model_name}_generations.json", "w") as f:
        json.dump(generations, f)


if __name__ == '__main__':
    # data = read_problems()
    
    # with open('./human_eval_results/llama3-8b-instruct_generations.jsonl', 'r') as f:
    #     data = [json.loads(line) for line in f]
        
    # responses = [d['completion'] for d in data]
    
    # problems = read_problems()
    # responses = extract_code_after_docstring(responses)
    # import pdb; pdb.set_trace()

    fire.Fire(main)
