import json
import pandas as pd
import os
import fire



def main(
    eval_dataset: str = "advbench",
):
    
    ## get all files in the directory
    files = os.listdir(f"../{eval_dataset}_gen_results")
    
    ## filter files end with _score.json
    files = [f for f in files if f.endswith("_score_dare.json")]
    
    ## read all files and collect the data
    name = []
    safe = []
    unsafe = []
    miss = []
    ASR = []
    
    for f_name in files:
        with open(os.path.join(f"../{eval_dataset}_gen_results", f_name), 'r') as f:
            data = json.load(f)
            
        name.append(f_name)
        safe.append(data[0]['safe'])
        unsafe.append(data[0]['unsafe'])
        miss.append(data[0]['miss'])
        ASR.append(data[0]['ASR'] * 100)
    
    df = pd.DataFrame({
        "name": name,
        "safe": safe,
        "unsafe": unsafe,
        "miss": miss,
        "ASR": ASR
    })
    
    df.to_csv(f"../{eval_dataset}_score_dare.csv", index=False)
    
if __name__ == "__main__":
    
    fire.Fire(main)