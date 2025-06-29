import subprocess
import fire
import os
import re
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd

def parse_pass_at_k(output, k):
    # Use regex to find the pass@k score
    pattern = rf"'pass@{k}': ([\d.]+)"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    else:
        print(f"Could not find pass@{k} score in the output.")
        return None

def process_file(file, results):
    """Process a single file and return its results"""
    # if file in results:
    #     print(f"Skipping {file} as it has already been evaluated.")
    #     return file, None
    # if f"{file}_results.jsonl" in results
    if file.endswith('results.jsonl'):
        print(f"Skipping {file} as it's a results file.")
        return file, None
    if file in results:
        print(f"Skipping {file} as it has already been evaluated.")
        return file, None
    # if not file.endswith('generations.jsonl'):
    #     return file, None
        
    command = f'evaluate_functional_correctness "./human_eval_results/{file}"'
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Successfully processed {file}")
        
        pass_at_1 = parse_pass_at_k(result.stdout, 1)
        pass_at_10 = parse_pass_at_k(result.stdout, 10)
        
        return file, {
            'pass_at_1': pass_at_1,
            'pass_at_10': pass_at_10
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file}: {e}")
        print("Error output:", e.stderr)
        return file, None

def run_evaluation(num_processes=None):
    # If num_processes not specified, use number of CPU cores minus 1
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    # Load existing results
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    
    # Get list of files to process
    files = [f for f in os.listdir('./human_eval_results')]
    print(f"Processing {len(files)} files using {num_processes} processes")
    
    # Create partial function with fixed results argument
    process_func = partial(process_file, results=results)
    
    all_results = []
    
    # Process files in parallel
    with Pool(processes=num_processes) as pool:
        for file, result in tqdm(pool.imap_unordered(process_func, files), 
                               total=len(files)):
            if result is not None:
                results[file] = result
                # Save results after each file is processed
                with open('./results.json', 'w') as f:
                    json.dump(results, f)
            all_results.append((file, result))
            
    # Save final results, write to CSV
    if os.path.exists('./results.csv'):
        df = pd.read_csv('./results.csv')
    else:
        df = pd.DataFrame(columns=['file', 'pass_at_1', 'pass_at_10'])

    # Create a list of new rows first
    new_rows = []
    for file, result in all_results:
        if result is not None:
            new_rows.append({
                'file': file,
                'pass_at_1': result['pass_at_1'],
                'pass_at_10': result['pass_at_10']
            })

    # Use concat to add new rows
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Optional: Save back to CSV
    df.to_csv('./results.csv', index=False)
                

if __name__ == "__main__":
    fire.Fire(run_evaluation)