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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

NUM_GPUS=${1:-4}
TP_SIZE=${2:-1}
NUM_DRAFT_LAYERS=${NUM_DRAFT_LAYERS:-4}
DRAFT_SLIDING_WINDOW=${DRAFT_SLIDING_WINDOW:-256}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
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
    --target-model-path /data/HUGGINGFACE/Qwen3-1.7B \
    --num-draft-layers "$NUM_DRAFT_LAYERS" \
    --draft-sliding-window "$DRAFT_SLIDING_WINDOW" \
    --train-data-path /data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl \
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
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.2 \
    --ttt-length 5 \
    "${RESUME_ARGS[@]}"
