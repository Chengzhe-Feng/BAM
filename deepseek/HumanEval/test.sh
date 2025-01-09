LANG="python"
OUPUT_DIR=""
MODEL=""

CUDA_VISIBLE_DEVICES=0 python eval_instruct.py \
    --model "$MODEL" \
    --output_path "$OUPUT_DIR/${LANG}" \
    --language $LANG \
    --temp_dir $OUPUT_DIR
