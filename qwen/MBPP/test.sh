MODEL_DIR=""
OUTPUT_DIR=""
METRIC_OUTPUT_PATH=""

python eval_instruct_q.py \
  --output_path OUTPUT_DIR \
  --metric_output_path ${METRIC_OUTPUT_PATH} \
  --temp_dir "./output" \
  --model_type qwen2 \
  --model_size chat \
  --model_path ${MODEL_DIR} \
  --bs 1 \
  --temperature 0 \
  --n_samples 1 \
  --greedy \
  --root ${OUTPUT_DIR} \
  --dataset mbpp \
  --tensor-parallel-size 1