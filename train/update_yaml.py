import yaml
import os
import fire
from typing import Dict, Any

float_args = [
    'learning_rate', 
    "neftune_noise_alpha", 
    "eval_steps", 
    "save_steps", 
    "max_steps", 
    "weight_decay", 
    "warmup_steps", 
    "max_grad_norm", 
    "adam_epsilon", 
    "lora_dropout"
]

def update_yaml(yaml_file: str, exp_name: str, output_dir: str, **updates: Any) -> str:
    """
    Update a YAML file with new values and save it to a new file.

    Args:
    yaml_file (str): Path to the original YAML file.
    exp_name (str): Name of the experiment (used in the new file name).
    output_dir (str): Directory to save the new YAML file.
    **updates: Keyword arguments representing the updates to make.

    Returns:
    str: Path to the new YAML file.
    """
    # Read the existing YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file) or {}

    # Update the data
    for key, value in updates.items():
        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        
        # Check if the key is one of the specified numeric fields and convert the value accordingly
        if keys[-1] in float_args:
            try:
                if keys[-1] == "max_steps":
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                print(f"Warning: '{keys[-1]}' value '{value}' could not be converted to the appropriate numeric type. Using as-is.")
        
        current[keys[-1]] = value
    
    ## write output_dir into the yaml file
    data['output_dir'] = output_dir

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the new file name
    origin = os.path.splitext(os.path.basename(yaml_file))[0]
    new_file_name = f"{origin}_{exp_name}.yaml"
    new_file_path = os.path.join(output_dir, new_file_name)

    # Write the updated data to the new file
    with open(new_file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    return new_file_path

def main(yaml_file: str, exp_name: str, output_dir: str, **updates: Any) -> None:
    """
    Main function to update a YAML file and save the result.

    Args:
    yaml_file (str): Path to the original YAML file.
    exp_name (str): Name of the experiment (used in the new file name).
    output_dir (str): Directory to save the new YAML file.
    **updates: Keyword arguments representing the updates to make.
    """
    try:
        new_file_path = update_yaml(yaml_file, exp_name, output_dir, **updates)
        print(f"New YAML file '{new_file_path}' has been created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    fire.Fire(main)