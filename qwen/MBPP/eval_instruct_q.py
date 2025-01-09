import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import re
from pathlib import Path
from tqdm import tqdm
import jsonlines
from model import DecoderBase, make_model


data_abs_dir = Path(__file__).parent / "data"

from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness

MODEL_MAPPING = {
    #  Can be either repo's name or /path/to/model
    "codeqwen": {
        "base": "Qwen/CodeQwen1.5-7B",
        "chat": "Qwen/CodeQwen1.5-7B-Chat",
        "chat-awq": "Qwen/CodeQwen1.5-7B-Chat-AWQ",
    },
    "qwen2": {
        "chat": "Qwen/CodeQwen1.5-7B-Chat",
    },
}

def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
'''.strip().format('\n\n'.join(examples_str), prompt)
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_with_shots
        }

def convert_for_evaluation(example):
    gpt_completion = example['gpt_completion']
    generation = gpt_completion
    try:
        code_block: str = re.findall(f'```python\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example['generation'] = generation
    return example

def generate_one(example, model):
    prompt = example['prompt']

    outputs = model.codegen(
                    prompt,
                    do_sample=False,
                    num_samples=1,
                )

    # print(output)
    example['gpt_completion'] = outputs
    res = outputs[0]
    example['generation'] = res
    return example

def generate_main(args):
    model_name_or_path = args.model_path
    saved_path = args.output_path
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"mbpp.jsonl")

    print("model", model_name_or_path)
    model = make_model(
        model_type=args.model_type,
        model_size=args.model_size,
        model_path=model_name_or_path,
        batch_size=args.bs,
        temperature=args.temperature,
        dataset=args.dataset,
        tensor_parallel_size=args.tensor_parallel_size
    )

    examples = list(read_test_examples(problem_file))
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []
    for ex in tqdm(examples, desc='Generating'):
        gen_example = generate_one(ex, model)
        generated_examples.append(gen_example)
        print("Generate {}/{} over...".format(len(generated_examples), len(examples)))

    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))
    
    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        problem_file=os.path.join(data_abs_dir, f"mbpp.jsonl"),
        language='python',
        is_mbpp=True
    )
    print(result, model_name_or_path)
    dumped = json.dumps(result, indent=2)
    with open(args.metric_output_path, "w") as f:
        f.write(dumped)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help="output path of your generation")
    parser.add_argument('--temp_dir', type=str, help="temp dir for evaluation", default="tmp")
    parser.add_argument('--metric_output_path', type=str, help="evaluation result", default="res")
    parser.add_argument("--model_type", required=True, type=str, choices=MODEL_MAPPING.keys())
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_size", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--dataset", required=True, type=str, choices=["humaneval", "mbpp"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=str)
    parser.add_argument("--tensor-parallel-size", default=1, type=int)
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(args)
    assert args.model_size in MODEL_MAPPING[args.model_type]

    model_path = MODEL_MAPPING[args.model_type][args.model_size]
    if args.model_path is not None:
        model_path = args.model_path
    print(f"Loading model from {model_path}")

    print(f"Running model={args.model_type}, size={args.model_size}")
    print(f"\tLoad from `{model_path}`")

    if args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make project dir
    os.makedirs(args.root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)
    generate_main(args)
    pass