## 1. Introduction

Essential code to run **BAM** model on [**HumanEval**](https://huggingface.co/datasets/openai_humaneval) and [**MBPP**](https://huggingface.co/datasets/mbpp) benchmarks.



## 2. Setup
The configuration of the experiment environment is in environment.yml. You can install and configure the experiment environment by executing the following command:
```
conda env create -f environment.yml -n bam
```
To run BAM, you need to first download the Humaneval and MBPP datasets, and also have the Qwen2.5-coder-instruct and base models, as well as the Deepseek-coder-instruct-v1.5 and base models, downloaded locally. These datasets and models can all be downloaded from Hugging Face.

After downloading, you need to replace the corresponding paths in the following files.
```
Replace the model paths in the deepseek_test.yml and qwen_test.yml files within the mergekit directory with the local paths. 
Place the required experiment data in the corresponding data folder of each benchmark
Modify the paths accordingly based on the comments in the run_bam.py file in each folder.
```
## 3. Evaluation

**To run BAM**
After entering the desired model series and corresponding benchmark dataset, run the following command to run BAM.
```
python run_bam.py
```
**To eval single model**
In each folder, we provide a script to run the test for a single merged model. If you want to reproduce the best-parameter merged model shown in the paper, please follow the instructions below.
```
Modify the weight in the deepseek_test.yml or qwen_test.yml depending on which model you'd like to run with the weight parameters we released in our paper
mergekit-yaml /'the dir of mergekit'/qwen_test.yml /'output_dir of merged llm' --cuda --allow-crimes --trust-remote-code
```
After this, you will get a merged model. Then run the code below to eval it:
```
For Deepseek-HumanEval or MBPP, modify the dir in the test.sh, then use "bash test.sh" to run
For Qwen-HumanEval, modify the dir in the evaluate_humaneval.sh, then use "bash evaluate_humaneval.sh" to run
For Qwen-MBPP, modify the dir in the test.sh, then use "bash test.sh" to run
```
To eval Deepseek or Qwen model without model merging, just skip the merging instruction and run the scipts above.