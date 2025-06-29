import os
import json
import fire
from evaluate import load
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    GenerationConfig
)
import random
from typing import List, Optional, TypedDict
from tqdm import tqdm


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
        self.use_peft = use_peft
        if "gemma" in model_path:
            self.add_sys = False
        else:
            self.add_sys = add_sys
        
        if use_peft:
            config = PeftConfig.from_pretrained(model_path)
            try :
                model_path = getattr(config, "base_model_name_or_path")
            except:
                assert False, "Please provide the base model path in the config file"        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Ensure pad token exists
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = "[PAD]"
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     self.tokenizer.padding_side = "left"
            
        self.llm = LLM(
            model_path,
            tokenizer=model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            enable_lora=use_peft,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )
        # self.llm = AutoModelForCausalLM.from_pretrained(
        #     self.model_path, 
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
            
        # )
        # Resize embeddings if new tokens were added
        # if len(self.tokenizer) != self.llm.config.vocab_size:
        #     self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # # Ensure the model knows about the pad token
        # if self.llm.config.pad_token_id is None:
        #     self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.system_prompt = "If you are a doctor, please answer the medical questions based on the patient's description."
        ## if gemma, self.add_sys = False
        
        self.use_peft = use_peft
        
    def generate_prompt(self, system_prompt: str, user_message: str = None, choice=None) -> str:
        
        if self.add_sys:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        
        else:
            messages = [
                {"role": "user", "content": system_prompt + " " + user_message}
            ]
            
        return messages
    
    
    def generate(self, prompts, return_only_responses=True):
        
        def rreplace(s, old, new, occurrence):
            li = s.rsplit(old, occurrence)
            return new.join(li)
        # import pdb; pdb.set_trace()
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
        
        
        responses = []
        token_num = []
        
        for output in tqdm(outputs):
            generated_text = output.outputs[0].text
            responses.append(generated_text.strip())
            token_num.append(len(output.outputs[0].token_ids))
        
        if return_only_responses:
            return responses
        else:
            return responses, token_num
    # def generate(self, prompts, return_only_responses=True):
    #     # Process all prompts using chat template
    #     formatted_prompts = [
    #         self.tokenizer.apply_chat_template(
    #             self.generate_prompt(self.system_prompt, prompt), 
    #             add_generation_prompt=True, 
    #             tokenize=False
    #         ).replace(self.tokenizer.bos_token, "", 1)
    #         for prompt in prompts
    #     ]
        
    #     # Batch tokenize all prompts
    #     inputs = self.tokenizer(
    #         formatted_prompts,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         return_attention_mask=True
    #     ).to(self.llm.device)
        
    #     # Generate outputs for the entire batch
    #     generation_config = GenerationConfig(
    #         max_new_tokens=self.max_tokens,
    #         num_beams=1,
    #         do_sample=False,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         bos_token_id=self.tokenizer.bos_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         repetition_penalty=1.2,
    #     )
        
    #     with torch.no_grad():
    #         outputs = self.llm.generate(
    #             input_ids=inputs.input_ids,
    #             attention_mask=inputs.attention_mask,
    #             generation_config=generation_config,
    #             return_dict_in_generate=True,
    #             output_scores=True
    #         )
    #     ## remove input tokens
    #     outputs.sequences = outputs.sequences[:, inputs.input_ids.shape[1]:]
        
    #     # Decode all outputs in batch
    #     decoded_outputs = self.tokenizer.batch_decode(
    #         outputs.sequences, 
    #         skip_special_tokens=True
    #     )
    #     # Process responses to extract only the generated part
    #     responses = []
    #     token_nums = []
        
    #     for i, decoded_output in enumerate(decoded_outputs):
    #         # Remove the input prompt from the output
    #         response = decoded_output
    #         responses.append(response)
    #         token_nums.append(len(outputs.sequences[i]))
            
    #     if return_only_responses:
    #         return responses
    #     else:
    #         return responses, token_nums

def main(
    model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    model_name: str = "llama3-8b-instruct",
    use_peft: bool = False,
    seed: int = 42,
    add_sys: bool = True,
    val: bool = False,
    gen_result_dir: str = "./generation_results",
    score_result_dir: str = "./score_results",
    max_num_seqs: int = 16
):
    set_seed(seed)
    system_prompt = "If you are a doctor, please answer the medical questions based on the patient's description."
    
    
    
    
    os.makedirs(gen_result_dir, exist_ok=True)
    os.makedirs(score_result_dir, exist_ok=True)
    
    # cal_bert_score(predictions, references)

    if not val:
        with open("./dataset/iCliniq-1k.json", "r") as f:
            data = json.load(f)
    else:
        with open("./dataset/HealthCareMagic-valid_200.json", "r") as f:
            data = json.load(f)
        
    
    inputs = [d['input'] for d in data]
    references = [d['answer_icliniq'] for d in data]
    
    ## generate responses
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0, repetition_penalty=1.2)
    ## detect # of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    llm = VLLMModel(model_path, sampling_params=sampling_params, use_peft=use_peft, tp_size=num_gpus, max_num_seqs=max_num_seqs, add_sys=add_sys)
    
    # batch_size = 8
    # ## need to handle the last batch
    # batched_inputs = [
    #     inputs[i:min(i+batch_size, len(inputs))] 
    #     for i in range(0, len(inputs), batch_size)
    # ]
    
    # generations = []
    generations = llm.generate(inputs)
    # for batch in tqdm(batched_inputs):
    #     generations.extend(llm.generate(batch))
    if not val:
        with open(f"{gen_result_dir}/{model_name}_generations.json", "w") as f:
            json.dump(generations, f)
    else:
        with open(f"{gen_result_dir}/{model_name}_generations_valid.json", "w") as f:
            json.dump(generations, f)
    


if __name__ == '__main__':
    fire.Fire(main)
