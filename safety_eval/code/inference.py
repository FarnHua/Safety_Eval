import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftConfig
from datasets import load_dataset
from tqdm import tqdm, trange
from random import seed, choice
import fire


class VLLMModel:
    def __init__(
        self, 
        model_path, 
        dtype="bfloat16",
        tp_size=1,
        max_num_seqs=32,
        max_tokens=512,
        temperature=0.0,
        sampling_params=None,
        use_peft=False,
        add_sys=True
    ):

        if sampling_params is None:
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, repetition_penalty=1.2)

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
        
        if add_sys:
            self.system_prompt = "You are a helpful assistant."
        else:
            self.system_prompt = ""
            
        self.use_peft = use_peft
        
    def generate_prompt(self, system_prompt: str, user_message: str = None, choice=None) -> str:
        
        if system_prompt == "":
            
            messages = [
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
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
        
        
        ## log sample 1
        print("Sample 1:\n ", prompts[0])
        
        if self.use_peft:
            outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, 
                                            lora_request=LoRARequest("lora", 1, self.model_path))
        else:
            outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        
        
        
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
    seed: int = 42,
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
    model_name: str = "llama31-8b-instruct_seed-42",
    use_peft: bool = False,
    add_sys: bool = True,
    eval_dataset: str = "advbench",
    max_num_seqs: int = 32,
):
    
    
    tp_size = torch.cuda.device_count()
    model = VLLMModel(model_path=model_path, use_peft=use_peft, tp_size=tp_size, add_sys=add_sys, max_num_seqs=max_num_seqs)
    
    set_seed(seed)
    
    
    if eval_dataset == "advbench":         
        ## advbench
        dataset = load_dataset("walledai/AdvBench")['train']
        prompts = [
            d['prompt'] for d in dataset
        ]
    elif eval_dataset == "hex-phi":
        with open("/livingrooms/farnhua/SafeIT/safety_eval/HEx-PHI/all.json", "r") as f:
            dataset = json.load(f)
            prompts = dataset['prompt']
    
    generations = model.generate(dataset['prompt'])
    
    out = []
    for idx, output_text in enumerate(generations):
        out.append({'prompt': dataset['prompt'][idx], 'answer': output_text})
    
    
    output_dir = f"../{eval_dataset}_gen_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name + f"_{eval_dataset}.jsonl")
    with open(output_path, 'w') as f:
        for li in out:
            f.write(json.dumps(li))
            f.write("\n")

if __name__ == "__main__":
    # Load the model
    fire.Fire(main)
