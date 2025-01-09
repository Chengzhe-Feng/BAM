from skopt import gp_minimize
from skopt.space import Real,Categorical
import numpy as np

import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
import yaml
import re
import csv

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

data_abs_dir = Path(__file__).parent / "data"

history = []

def target_function(params):
    # replace the dir of mergekit with the dir on your machine
    with open('/the dir of mergekit/qwen_test.yml', 'r') as file:
        config = yaml.safe_load(file)
    a,b = params
    config['models'][0]['parameters']['weight'] =  str(a)
    config['models'][1]['parameters']['weight'] =  str(b)
    # replace the dir of mergekit with the dir on your machine
    with open('/the dir of mergekit/qwen_test.yml', 'w') as file:
        yaml.safe_dump(config, file,default_flow_style=False)
    # replace the dir of mergekit and the output_dir of merged llm with the dir on your machine
    command = "mergekit-yaml /'the dir of mergekit'/qwen_test.yml /'output_dir of merged llm' --cuda --allow-crimes --trust-remote-code"
    import subprocess

    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Command executed successfully.")
        print("STDOUT:", result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print("Command failed.")
        print("STDERR:", e.stderr.decode())

    # replace the dir of the output_dir of merged llm with the dir on your machine
    model_name_or_path = "/output_dir of merged llm"
    lang = "python"

    #replace with the dir that you define
    saved_path = "/saved_path for generated code"
    metric_path = "/saved_path for metric"

    variables_to_modify = {
    "TOKENIZERS_PARALLELISM": "false",
    "DATA_PATH": "./data",
    "MODEL_DIR": model_name_or_path,
    "LANGUAGE": "python",
    "GENERATION_PATH": saved_path,
    "METRIC_OUTPUT_PATH": metric_path,
    "TP": "1"
    }

    script_path = 'evaluate_humaneval.sh'
    with open(script_path, 'r') as file:
        script_content = file.read()

    for var, value in variables_to_modify.items():
        pattern = re.escape(var) + r'=.*'
        script_content = re.sub(pattern, f'{var}={value};', script_content, count=1)

    temp_script_path = 'modified_script.sh'
    with open(temp_script_path, 'w') as file:
        file.write(script_content)

    subprocess.run(['bash', temp_script_path], check=True)

    os.remove(temp_script_path)

    file_path = metric_path

    with open(file_path, 'r') as file:
        data = json.load(file)
    pass_rate = data['pass@1']
    print(lang, pass_rate, model_name_or_path)
    history.append((params, pass_rate))
    return -pass_rate

search_space = [
    Categorical([round(x, 2) for x in list(np.arange(0.01, 1.00, 0.01))], name="a"),
    Categorical([round(x, 2) for x in list(np.arange(0.01, 1.00, 0.01))], name="b"),
]

result = gp_minimize(
    target_function,
    search_space,
    n_calls=15,
)

print("All intermediate results:")
for i, (params, res) in enumerate(history):
    print(f"Iteration {i + 1}: a={params[0]}, b={params[1]}, result={res}")

with open("optimization_history.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "a", "b", "Result"])
    for i, (params, res) in enumerate(history):
        writer.writerow([i + 1, params[0], params[1], res])

print("History saved to optimization_history.csv")




