export TOKENIZERS_PARALLELISM=true;

DATA_PATH="./data"

MODEL_DIR=""
LANGUAGE="python"
GENERATION_PATH=""
METRIC_OUTPUT_PATH=""
TP=1

PORT=$((RANDOM%9000+1000))
echo "PORT: ${PORT}"

echo "benchmark: humaneval, language ${LANGUAGE}"

python evaluate.py  \
    --benchmark "humaneval" \
    --model ${MODEL_DIR} \
    --language ${LANGUAGE} \
    --generation_path ${GENERATION_PATH} \
    --metric_output_path ${METRIC_OUTPUT_PATH} \
    --model_max_length 1024 \
    --data_path ${DATA_PATH} \
    --use_vllm \
    --tensor_parallel_size ${TP}

