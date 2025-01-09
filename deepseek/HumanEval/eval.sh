MODEL_NAME_OR_PATH=""
DATASET_ROOT="data/"
LANGUAGE="python"
CUDA_VISIBLE_DEVICES=0 python eval_pal.py --logdir ${MODEL_NAME_OR_PATH} --language ${LANGUAGE} --dataroot ${DATASET_ROOT}