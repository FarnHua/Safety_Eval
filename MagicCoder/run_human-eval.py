import subprocess
import fire
import os
import re
from tqdm import tqdm
import json

def parse_pass_at_k(output, k):
    # Use regex to find the pass@k score
    pattern = rf"'pass@{k}': ([\d.]+)"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    else:
        print(f"Could not find pass@{k} score in the output.")
        return None


def run_evaluation():

    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    # files=["llama3-8b-instruct_magicoder-oss-instruct-10k_seed-1024_ckpt2500_generations.jsonl"]
    for file in tqdm(os.listdir('./human_eval_results')):
        
        if file.endswith('generations.jsonl'):
            if file in results:
                print(f"Skipping {file} as it has already been evaluated.")
                continue
            
            command = f'evaluate_functional_correctness "./human_eval_results/{file}"'
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
                print("Command executed successfully.")
                print("Output:")
                print(result.stdout)
                pass_at_1 = parse_pass_at_k(result.stdout, 1)
                pass_at_10 = parse_pass_at_k(result.stdout, 10)
                results[file] = dict(
                    pass_at_1=pass_at_1,
                    pass_at_10=pass_at_10
                )
                # import pdb; pdb.set_trace()
                with open('./results.json', 'w') as f:
                    json.dump(results, f)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing the command: {e}")
                print("Error output:")
                print(e.stderr)

if __name__ == "__main__":
    
    fire.Fire(run_evaluation)