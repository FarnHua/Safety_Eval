import os
import json
import csv
import fire
from tqdm import tqdm
from typing import Dict, Any, List, Union
# Import both guards
from guard import WildGuard, GPTGuard
from upload_csv import upload_csv_to_gsheets

def save_json(output_path: str, data: Dict[str, Any]) -> None:
    """Save evaluation results to a JSON file.
    
    Args:
        output_path: Path where to save the JSON file
        data: Dictionary containing results and metrics
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def record_csv(
    csv_dir: str,
    eval_dataset: str,
    file_name_no_ext: str,
    model_name: str,
    asr: float,
    missing_rate: float
) -> None:
    """Record evaluation metrics to CSV and upload to Google Sheets.
    
    Args:
        csv_dir: Directory to store CSV files
        eval_dataset: Name of the evaluation dataset
        file_name_no_ext: Input filename without extension
        model_name: Name of the guard model used
        asr: Attack Success Rate as a fraction
        missing_rate: Rate of missing/invalid responses as a fraction
    """
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, eval_dataset) + ".csv"
    header = ["filename", "model", "eval_dataset", "ASR", "MISS"]
    new_row = [
        file_name_no_ext,
        model_name,
        eval_dataset,
        asr,
        missing_rate
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writerow(header)
        writer.writerow(new_row)
    # upload_csv_to_gsheets(
    #     csv_path=csv_path,
    #     worksheet_name=f"{eval_dataset}_{model_name}",  
    #     start_cell="A1"
    # )

def process_results(
    results: List[Dict[str, Any]],
    total_items: int,
    unsafe_count: int,
    missing_count: int
) -> Dict[str, Any]:
    """Process results into a standardized output format."""
    asr = unsafe_count / (total_items - missing_count) if (total_items - missing_count) else 0.0
    missing_rate = missing_count / total_items if total_items else 0.0
    
    return {
        "results": results,
        "summary": {
            "total": total_items,
            "unsafe": unsafe_count,
            "missing": missing_count,
            "attack_success_rate": round(asr * 100, 4),
            "missing_rate": round(missing_rate * 100, 4),
        }
    }

def main(
    file_path: str = "../gen_results/llama3-8b-instruct_advbench.jsonl",
    eval_type: str = "",  # for wildguard evaluation, used to tag output file names
    eval_dataset: str = "evaluation_dataset",
    guard_type: str = "wildguard",  # "wildguard" or "gpt"
    model: str = "gpt-4o-mini-2024-07-18",  # for GPT evaluation
    env_file: str = ".env",
    lang: str = "en"
) -> None:
    """Evaluate safety of model responses using specified guard.
    
    Args:
        file_path: Path to input JSONL file
        eval_type: Tag for WildGuard evaluation output
        eval_dataset: Name of evaluation dataset
        guard_type: Type of guard to use ("wildguard" or "gpt")
        model: Model name for GPT evaluation
        env_file: Path to environment file
        lang: Language for evaluation
    """
    # Load JSONL file
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    total_items = len(data)
    
    # Initialize common variables
    results = []
    model_name = ""
    asr = 0.0
    missing_rate = 0.0

    if guard_type == "gpt":
        guard = GPTGuard()
        model_name = model
        unsafe_count = 0
        missing_count = 0
        for entry in tqdm(data, desc="Evaluating with GPTGuard"):
            prompt = entry.get("prompt", "")
            answer = entry.get("answer", "")
            eval_result = guard.evaluate(prompt, answer, model=model, env_file=env_file)
            determination = eval_result.get("determination")
            explanation = eval_result.get("explanation")
            if not determination:
                missing_count += 1
            if determination == "unsafe":
                unsafe_count += 1
            results.append({
                "prompt": prompt,
                "answer": answer,
                "determination": determination,
                "explanation": explanation,
            })
        output_data = process_results(results, total_items, unsafe_count, missing_count)
        output_suffix = f"_gpteval-{model}.json"
        csv_dir = f"../{eval_dataset}_gpt_eval/"
        
    elif guard_type == "wildguard":
        # Use WildGuard evaluation (original generate/parse flow)
        guard = WildGuard()
        model_name = "WildGuard"
        user_messages = [{"prompt": d['prompt'], "response": d['answer']} for d in data]
        responses = guard.generate(user_messages)
        count = 0
        miss = 0
        for pr in responses:
            if pr['is_parsing_error']:
                miss += 1
            elif pr['response_harmfulness'] == "harmful":
                count += 1
       
        # Prepend summary to detailed results.
        results = []
        results.extend(
            [dict(prompt=d['prompt'], response=d['answer'], parsed_response=pr)
             for d, pr in zip(data, responses)]
        )
        output_data = process_results(results, total_items, count, miss)
        output_suffix = f"_wildguard.json"
        csv_dir = f"../{eval_dataset}_wildguard_eval/"
    
    else:
        raise ValueError(f"Unknown guard type: {guard_type}")

    # Save detailed output JSON
    json_dir = f"../{eval_dataset}_{guard_type}_eval_details"
    os.makedirs(json_dir, exist_ok=True)
    output_path = os.path.join(json_dir, os.path.basename(file_path).replace(".jsonl", output_suffix))
    save_json(output_path, output_data)
    
    file_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    record_csv(
        csv_dir,
        eval_dataset,
        file_name_no_ext,
        model_name,
        output_data["summary"]["attack_success_rate"],
        output_data["summary"]["missing_rate"]
    )
    
if __name__ == "__main__":
    fire.Fire(main)
