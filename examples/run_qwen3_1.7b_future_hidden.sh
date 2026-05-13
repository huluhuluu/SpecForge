#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

NUM_GPUS=${NUM_GPUS:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29621}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-1.7b-future-hidden}
LOG_FILE=${LOG_FILE:-$OUTPUT_DIR/train-$(date +%Y%m%d-%H%M%S).log}

mkdir -p "$OUTPUT_DIR"

python -m torch.distributed.run \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path /data/HUGGINGFACE/Qwen3-1.7B \
    --train-data-path /data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl \
    --chat-template qwen \
    --target-model-backend sglang \
    --enable-future-hidden \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --attention-backend sdpa \
    --cache-dir "$ROOT_DIR/cache" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"
