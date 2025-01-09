from peft import AutoPeftModelForCausalLM

model_dir="/data/llm/chengzhe/qwen_pretrain_lora/QWEN7b_all_stoTextbook_e10_lr2e-4_len1024"
output_dir="/data/llm/chengzhe/merged/QWEN7b_all_stoTextbook_e10_lr2e-4_len1024"
model = AutoPeftModelForCausalLM.from_pretrained(
    model_dir, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary.
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(output_dir, max_shard_size="4096MB", safe_serialization=True)