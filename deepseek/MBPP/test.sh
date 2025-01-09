OUPUT_DIR=""
MODEL=""
METRIC_OUTPUT_PATH=""

python eval_instruct.py \
    --model "$MODEL" \
    --output_path "$OUPUT_DIR/output.jsonl" \
    --metric_output_path ${METRIC_OUTPUT_PATH} \
    --temp_dir $OUPUT_DIR