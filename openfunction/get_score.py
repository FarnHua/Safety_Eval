import os
import json
import fire
import random
from evaluate import load
from transformers import set_seed
import numpy as np
import pandas as pd




def cal_bert_score(pred:list[str], ref:list[str]):

    bertscore = load("bertscore")
    results = bertscore.compute(
            predictions=pred, 
            references=ref, 
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            num_layers=40,
            batch_size=8,
            # rescale_with_baseline=True,
        )
    # import pdb; pdb.set_trace()
    return results


def main(
    eval_file:str = "./generation_results/llama3-8b-instruct_generations.json",
    ref_file:str = "./dataset/openfunction_test.json",
    result_path: str = "llama3-8b-instruct_generations_scores.json"
):
    
    with open(ref_file, 'r') as f:
        problems = json.load(f)

    ref = [
        p['output'] for p in problems
    ]
    with open(eval_file, "r") as f:
        pred = json.load(f)  
    
    with open(eval_file, "r") as f:
        pred = json.load(f)
    
    print("start calculating bert score")
    results = cal_bert_score(pred, ref)
    print("Precision Avg:", np.mean(results["precision"]))
    print("Recall Avg:", np.mean(results["recall"]))
    print("F1 Avg:", np.mean(results["f1"]))
    
    
    ## save results
    # result_path = eval_file.split("/")[-1].split(".")[0].replace(".", "")
    
    ## only split from the last dot
    result_path = eval_file.split("/")[-1].rsplit(".", 1)[0].replace(".", "")
    
    os.makedirs("./score_results", exist_ok=True)
    with open(f"./score_results/{result_path}", "w") as f:
        json.dump(results, f)
        
    ## write to a csv file
    if "val" in ref_file:
        file_name = "val_results.csv"
    else:
        file_name = "results.csv"
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame()
        
    ## save to csv
    new_row = pd.DataFrame({
        "model": [result_path],
        "precision": [float(np.mean(results["precision"]))],
        "recall": [float(np.mean(results["recall"]))],
        "f1": [float(np.mean(results["f1"]))],
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_name, index=False)
    print('done')
        
if __name__ == "__main__":
    fire.Fire(main)
    
    ## get average score
    # f_name = 
    # with open("./score_results/llama3-8b-instruct_healthcaremagic-100k-5000_scores.json", "r") as f:
    #     results = json.load(f)

    # file_name = "results.csv"
    # df = pd.read_csv(file_name)
    # import pdb; pdb.set_trace()
    # df = df.T
    # import pdb; pdb.set_trace()
    # df.columns = ["model_name", "precision", "recall", "f1"]
    # import pdb; pdb.set_trace()
    # df.to_csv(file_name, index=False)
    
    