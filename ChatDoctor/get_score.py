import os
import json
import fire
import random
from evaluate import load
from transformers import set_seed
import numpy as np
import pandas as pd
import sys



def cal_bert_score(pred:list[str], ref:list[str]):
    bertscore = load("bertscore")
    results = bertscore.compute(
            predictions=pred, 
            references=ref, 
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            num_layers=40,
            batch_size=8,
            nthreads=1
            # rescale_with_baseline=True,
        )
    
    return results


def main(
    eval_file:str = "./generation_results_new/llama3-8b-instruct_generations.json",
    result_path: str = "llama3-8b-instruct_generations_scores.json",
    score_result_dir: str = "./score_results",
    val: bool = False,
    type: str = "original_ft"
):
    
    if val:
        with open("./dataset/HealthCareMagic-valid_200.json", 'r') as f:
            data = json.load(f)
    else:
        with open("./dataset/iCliniq-1k.json", "r") as f:
            data = json.load(f)
    # import pdb; pdb.set_trace()
    ref = [d["answer_icliniq"] for d in data]
    
    with open(eval_file, "r") as f:
        pred = json.load(f)
    
    results = cal_bert_score(pred, ref)
    
    
    ## save results
    result_path = eval_file.split("/")[-1].rsplit(".", 1)[0]
    
    with open(f"{score_result_dir}/{result_path}", "w") as f:
        json.dump(results, f)
    
    ## write to a csv file
    if not val:
        file_name = f"results_{type}.csv"
    else:
        file_name = f"results_{type}_val.csv"
    
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
    # print("haha here")
    # sys.exit(0)

    # bertscore = load("bertscore")
    # predictions = ["hello there", "general kenobi"]
    # references = ["hello there", "general kenobi"]
    # results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="microsoft/deberta-xlarge-mnli",
    #         num_layers=40)
    # import pdb; pdb.set_trace()