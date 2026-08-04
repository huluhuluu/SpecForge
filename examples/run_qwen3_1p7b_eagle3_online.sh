#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-4}
TP_SIZE=${2:-4}
EXTRA_ARGS=("${@:3}")
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /data/HUGGINGFACE/Qwen3-1.7B \
    --train-data-path /data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-1p7b-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --attention-backend sdpa \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    "${EXTRA_ARGS[@]}"
