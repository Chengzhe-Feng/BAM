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
import csv

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

data_abs_dir = Path(__file__).parent / "data"

from utils.utils import extract_generation_code, languge_settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness


def build_deepseekcoder_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())


def generate_one(example, lang, tokenizer, model):
    prompt = build_deepseekcoder_instruction(languge_settings[lang]['full_name'], example['prompt'])
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"

    outputs = model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=False,
        # top_p=0.95,
        # temperature=temperature,
        pad_token_id=stop_id,
        eos_token_id=stop_id
    )

    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    example['output'] = output

    return extract_generation_code(example, lang_code=lang)

history = []
def target_function(params):
    # replace the dir of mergekit with the dir on your machine
    with open('/the dir of mergekit/deepseek_test.yml', 'r') as file:
        config = yaml.safe_load(file)
    a,b = params
    config['models'][0]['parameters']['weight'] =  str(a)
    config['models'][1]['parameters']['weight'] =  str(b)
    # replace the dir of mergekit with the dir on your machine
    with open('/the dir of mergekit/deepseek_test.yml', 'w') as file:
        yaml.safe_dump(config, file,default_flow_style=False)

    # replace the dir of mergekit and the output_dir of merged llm with the dir on your machine
    command = f"mergekit-yaml /'the dir of mergekit'/deepseek_test.yml /'output_dir of merged llm' --cuda --allow-crimes --trust-remote-code"
    import subprocess

    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Command executed successfully.")
        print("STDOUT:", result.stdout.decode())
        # if merged model missing tokenizer_config.json, then use this code.

        # command_2 = f"cp /dir to/deepseek-coder-7b-instruct-v1.5/tokenizer_config.json ./merged model dir"
        # result_2 = subprocess.run(command_2, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # if result.returncode == 0:
        #     print("copyed")
        # else:
        #     print("failed")
        #     print(result.stderr.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        print("Command failed.")
        print("STDERR:", e.stderr.decode())

    # replace the dir of the output_dir of merged llm with the dir on your machine
    model_name_or_path = "/output_dir of merged llm"
    lang = "python"
    #replace with the dir that you define
    saved_path = ""
    temp_dir = "output"

    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # use_flash_attention_2=True
    )
    model.eval()
    examples = [json.loads(x) for x in open(problem_file) if x.strip()]
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []
    for ex in tqdm(examples, desc='Generating'):
        gen_example = generate_one(ex, lang, tokenizer, model)
        generated_examples.append(gen_example)

    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))
    torch.cuda.empty_cache()
    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang
    )
    print(lang, result, model_name_or_path)
    pass_rate = result["pass@1"]
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
