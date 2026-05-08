#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi

# Four-card default example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/run_qwen3_4b_eagle3_online_sw256.sh 4 1
NUM_GPUS=${1:-4}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-16}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/data/HUGGINGFACE/Qwen3-4B-Instruct-2507}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-4b-eagle3-sharegpt-sw256}
MAX_NUM_STEPS=${MAX_NUM_STEPS:-}

EXTRA_ARGS=()
if [ -n "$MAX_NUM_STEPS" ]; then
    EXTRA_ARGS+=(--max-num-steps "$MAX_NUM_STEPS")
fi

torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/qwen3-4b-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --draft-sliding-window 256 \
    --ttt-length 7 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --attention-backend sdpa \
    --target-model-backend sglang \
    "${EXTRA_ARGS[@]}"
