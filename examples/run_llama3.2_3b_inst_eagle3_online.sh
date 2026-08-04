SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
EXTRA_ARGS=("${@:3}")
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /data/HF_MODELS/Llama-3.2-3B-Instruct \
    --train-data-path /data/HF_MODELS/data/specforge_sharegpt/sharegpt_train_clean.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/llama3.2-3b-inst-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.4 \
    "${EXTRA_ARGS[@]}"
