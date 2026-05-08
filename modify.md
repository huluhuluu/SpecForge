# Qwen3-1.7B EAGLE3 Sliding-Window Training

The goal is to introduce sliding-window attention into the EAGLE3 draft model and evaluate the effect of limiting attention visibility during training.

## 1. Environment Setup

```bash
git clone https://github.com/huluhuluu/SpecForge.git
cd SpecForge
conda create -n specforge python=3.11 -y
conda activate specforge
uv pip install -v . --prerelease=allow
```

## 2. Training

The following example trains from the local model `/data/HUGGINGFACE/Qwen3-1.7B` with the configuration below:

- `attention_backend=sdpa`
- `draft_sliding_window=256`
- `max_length=2048`
- `ttt_length=7`
- `GPU=0,1,2,3`
- dataset: `/data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl`

### 2.1 Online Training

The main launch script is `examples/run_qwen3_1.7b_eagle3_online.sh`.

```bash
cd SpecForge
conda activate specforge
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash examples/run_qwen3_1.7b_eagle3_online.sh 4 1
```

This command starts `scripts/train_eagle3.py` through `torchrun` with the following default arguments:

- `--target-model-path /data/HUGGINGFACE/Qwen3-1.7B`
- `--draft-model-config configs/qwen3-1.7b-eagle3.json`
- `--train-data-path /data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl`
- `--max-length 2048`
- `--draft-sliding-window 256`
- `--ttt-length 7`
- `--attention-backend sdpa`
- `--target-model-backend sglang`

## 3. Implementation

The core change is to propagate `draft_sliding_window` from the launch script into the actual draft attention computation path so that the attention range is truly constrained during training.

### 3.1 Argument Entry

The launch script `examples/run_qwen3_1.7b_eagle3_online.sh` explicitly passes the following arguments:

- `--draft-sliding-window 256`
- `--ttt-length 7`
- `--attention-backend sdpa`

Here, `draft_sliding_window` is the attention window size, while `ttt_length` is the Eagle3 TTT length. They represent different concepts.

### 3.2 Config Update

In `scripts/train_eagle3.py`, the `--draft-sliding-window` argument is added, and the following assignments are applied when building the draft model:

- `draft_model_config.sliding_window = args.draft_sliding_window`
- `draft_model_config.use_sliding_window = True`

This step writes both the window size and the enable flag into the draft config, rather than only keeping the value at the command-line level.

### 3.3 Config Readout

In `specforge/modeling/draft/base.py`, a new `get_sliding_window()` method is introduced with the following logic:

- return `None` if `use_sliding_window` is not `True`
- return `config.sliding_window` if the flag is enabled

This provides a single entry point for subsequent attention paths and avoids repeated config checks in multiple locations.

### 3.4 Mask Truncation

The `prepare_decoder_attention_mask()` function in the same file is extended with sliding-window truncation logic. The processing order is:

1. Build the standard causal mask.
2. Add the padding mask.
3. If sliding window is enabled, add an extra sliding bias.

The core formula is:

```text
lower_bound = query_position - sliding_window + 1
```

All positions with `key_position < lower_bound` are filled with a very small value and therefore masked out before softmax. As a result, each token can only attend to the most recent `N` previous tokens.

For example:

- with `sliding_window=256`, token 300 can attend back only to token 44
- with `sliding_window=64`, token 300 can attend back only to token 237

### 3.5 SDPA Path

The current training flow uses the `sdpa` backend, so the key implementation is in `specforge/modeling/draft/llama3_eagle.py`:

- read `sliding_window` from config during initialization
- build the out-of-window masking bias in `_apply_sliding_window_bias()`
- apply this bias in the standard `scaled_dot_product_attention(...)` forward path
- add the same bias to `attn_weights` in the Eagle3 attention branch that uses `cache_hidden`

This means the sliding-window constraint affects both the standard forward path and the Eagle3 training branch.

### 3.6 FlexAttention Path

Although the current training run does not use `flex_attention`, `specforge/modeling/draft/flex_attention.py` is updated in parallel with `generate_eagle3_mask(..., sliding_window=...)` logic:

- truncate the causal region by the same window
- truncate the suffix region by the same window

The purpose is to keep behavior as consistent as possible across different attention backends and avoid re-implementing sliding-window logic when switching backends later.

### 3.7 Online Data Loading

In `scripts/train_eagle3.py`, both training and evaluation datasets are changed from `Dataset.from_generator(...)` to direct JSONL loading followed by `Dataset.from_list(...)`. The following fields are also included in the dataset cache key:

- `train_data_path`
- `max_length`
- `chat_template`
- `target_model_path`
- `attention_backend`
- `draft_sliding_window`

The goal is to reduce cache collisions across different training configurations.
