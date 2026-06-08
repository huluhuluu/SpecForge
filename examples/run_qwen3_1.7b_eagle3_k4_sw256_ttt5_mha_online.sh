#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

if [[ "${CONDA_DEFAULT_ENV:-}" != "test-spec" ]]; then
    CONDA_BASE=$(conda info --base)
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate test-spec
fi

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

NUM_GPUS=${1:-4}
TP_SIZE=${2:-1}
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-2}
DRAFT_SLIDING_WINDOW=${DRAFT_SLIDING_WINDOW:-256}
DRAFT_ATTENTION_TYPE=${DRAFT_ATTENTION_TYPE:-mha}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/data/HF_MODELS/datasets/sharegpt-specforge/sharegpt_train.jsonl}
TARGET_MODEL_BACKEND=${TARGET_MODEL_BACKEND:-sglang}
RESUME=${RESUME:-false}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29631}

RESUME_ARGS=()
if [[ "$RESUME" == "true" ]]; then
    RESUME_ARGS+=(--resume)
fi

torchrun \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path /data/HF_MODELS/Qwen3-1.7B \
    --num-draft-layers "$NUM_DRAFT_LAYERS" \
    --draft-sliding-window "$DRAFT_SLIDING_WINDOW" \
    --draft-attention-type "$DRAFT_ATTENTION_TYPE" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --build-dataset-num-proc "$BUILD_DATASET_NUM_PROC" \
    --output-dir "$ROOT_DIR/outputs/qwen3-1.7b-eagle3-k${NUM_DRAFT_LAYERS}-sw${DRAFT_SLIDING_WINDOW}-sharegpt" \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir "$ROOT_DIR/cache" \
    --embedding-key model.embed_tokens.weight \
    --tp-size "$TP_SIZE" \
    --attention-backend sdpa \
    --target-model-backend "$TARGET_MODEL_BACKEND" \
    --sglang-mem-fraction-static 0.2 \
    --ttt-length 5 \
    "${RESUME_ARGS[@]}"
