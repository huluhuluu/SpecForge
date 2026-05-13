# Qwen3-1.7B EAGLE3 Future-Hidden Training

The goal is to design a new appended future-MASK-slot for Eagle3.

## 1. Environment Setup

```bash
git clone -b feat/mask-hidden https://github.com/huluhuluu/SpecForge.git
cd SpecForge
conda create -n specforge python=3.11 -y
conda activate specforge
uv pip install -v . --prerelease=allow
```

## 2. Training

The following example trains from the local model `/data/HUGGINGFACE/Qwen3-1.7B` with the configuration below:

- `target_model_backend=sglang`
- `attention_backend=sdpa`
- `enable_future_hidden=True`
- `max_length=2048`
- `ttt_length=7`
- `GPU=2,3,4,5`
- dataset: `/data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl`

### 2.1 Online Training

The main launch script is [examples/run_qwen3_1.7b_future_hidden.sh](/workspace/code/test-spec/SpecForge/examples/run_qwen3_1.7b_future_hidden.sh).

```bash
cd SpecForge
conda activate specforge
export CUDA_VISIBLE_DEVICES=2,3,4,5
mkdir -p outputs/qwen3-1.7b-future-hidden
bash examples/run_qwen3_1.7b_future_hidden.sh \
  2>&1 | tee outputs/qwen3-1.7b-future-hidden/train.log
```

This command starts `scripts/train_eagle3.py` through `python -m torch.distributed.run` with the following default arguments:

- `--target-model-path /data/HUGGINGFACE/Qwen3-1.7B`
- `--train-data-path /data/HUGGINGFACE/data/specforge_sharegpt/sharegpt_train.jsonl`
- `--chat-template qwen`
- `--target-model-backend sglang`
- `--enable-future-hidden`
- `--num-epochs 10`
- `--batch-size 1`
- `--learning-rate 1e-4`
- `--max-length 2048`
- `--ttt-length 7`
- `--attention-backend sdpa`
- `--cache-dir $ROOT_DIR/cache`
- `--output-dir $ROOT_DIR/outputs/qwen3-1.7b-future-hidden`

The launch script also uses:

- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=29621`

These are only for distributed process-group bootstrap on the local node.

## 3. Implementation

The core change is to let the target model append future MASK slots at the tail of the sequence, extract the hidden states of the final real token plus those MASK slots, and feed them step-by-step into the draft model as an additional third input stream.

### 3.1 Target-Side Goal

Given:

```text
context = A B C D E
future  = F G H
```

the target-side augmented input becomes:

```text
A B C D E MASK1 MASK2
```

Then the final-layer hidden states are gathered as:

- `h0 = hidden(E)`
- `h1 = hidden(MASK1)`
- `h2 = hidden(MASK2)`

These are aligned with Eagle3 TTT prediction steps:

- `step 0` uses `h0`
- `step 1` uses `h1`
- `step 2` uses `h2`

So the draft model predicts:

- `predict F` conditioned on `h0`
- `predict G` conditioned on `h1`
- `predict H` conditioned on `h2`

### 3.2 End-to-End Flow

```text
                  +---------------------------+
                  |      target input         |
                  | context + MASK1..MASKk    |
                  +-------------+-------------+
                                |
                                v
                    +-----------------------+
                    |     Target Model      |
                    |     single forward    |
                    +-----------+-----------+
                                |
                                v
                  +-------------------------------+
                  | future_hidden = [h0, h1, h2] |
                  | h0=NTP, h1=MASK1, h2=MASK2   |
                  +---------------+---------------+
                                  |
                +-----------------+------------------+
                |                 |                  |
                v                 v                  v
        +---------------+ +---------------+ +---------------+
        | draft step 1  | | draft step 2  | | draft step 3  |
        | predict F(h0) | | predict G(h1) | | predict H(h2) |
        +-------+-------+ +-------+-------+ +-------+-------+
                |                 |                  |
                v                 v                  v
      concat(draft_hidden, emb, future_hidden_i) at each step
                |                 |                  |
                +-----------------+------------------+
                                  |
                                  v
                       +----------------------+
                       |   Draft Model 3H in  |
                       | hidden + emb + cond  |
                       +----------------------+
```

### 3.3 Argument Entry

In `scripts/train_eagle3.py`, the following arguments are added:

- `--enable-future-hidden`
- `--future-mask-token-id`

Backward-compatible aliases are also kept:

- `--enable-mask-hidden`
- `--mask-token-id`

The launch script only needs `--enable-future-hidden` in the normal case. If the tokenizer does not expose a usable mask token, `--future-mask-token-id` can be set explicitly.

### 3.4 Runtime Constraints

The current implementation intentionally limits scope:

- future-hidden only supports online training
- future-hidden does not support offline hidden-state training
- future-hidden does not support VLM training

These checks are enforced in `scripts/train_eagle3.py`.

## 4. Modify Notes

<details>
<summary>English Modification Notes</summary>

This feature is implemented as a four-stage pipeline:

1. the training entry enables future-hidden mode and configures the target model
2. the target model appends future MASK slots and returns `future_hidden_states`
3. the online Eagle3 loop aligns one future hidden vector to each TTT step
4. the draft model consumes that vector as a third input stream on top of the original 2H design

### 4.1 Training Entry Changes

Code references in `scripts/train_eagle3.py`:

- `99-112`
  add CLI arguments:
  - `--enable-future-hidden`
  - `--future-mask-token-id`
  - plus aliases `--enable-mask-hidden` and `--mask-token-id`
- `269-282`
  add `resolve_future_mask_token_id(...)`
  - explicit argument first
  - then tokenizer `mask_token_id`
  - then tokenizer `eos_token_id`
  - then tokenizer `pad_token_id`
- `376-383`
  add runtime constraints:
  - online-only
  - no VLM support
  - `max_length > ttt_length - 1`
- `414-459`
  update draft model initialization:
  - set `draft_model_config.use_future_hidden = True`
  - load checkpoints with `config=draft_model_config`
  - allow `ignore_mismatched_sizes=args.enable_future_hidden`
- `492-529`
  reserve sequence headroom:
  - `dataset_max_length = max_length - (ttt_length - 1)`
  - build datasets with `Dataset.from_list(...)`
- `688-715`
  wire `future_hidden_states` from target output into Eagle3 training forward
- `811-819`
  configure the target model at runtime with:
  - resolved future mask token id
  - `future_hidden_length=args.ttt_length`

### 4.2 Target-Side Changes

Code references in `specforge/modeling/target/eagle3_target_model.py`:

- `43-50`
  extend `Eagle3TargetOutput` with:
  - `future_hidden_states`
- `61-77`
  store target-side future-hidden state:
  - `future_hidden_token_id`
  - `future_hidden_length`
- `128-145`
  add target configuration helpers:
  - `set_future_hidden_token_id(...)`
  - `set_future_hidden_length(...)`
  - `configure_future_hidden(...)`
  - `_future_hidden_enabled()`
- `147-184`
  build augmented target inputs:
  - append `future_hidden_length - 1` MASK slots
  - expand attention mask
  - record gather positions as:
    - last real token position
    - each appended MASK slot position
- `186-197`
  gather the final-layer hidden states from the recorded positions
  - returned shape is `[B, TTT, H]`
- `330-351`
  HF backend integration:
  - run one extra forward on `augmented_input_ids`
  - set `use_cache=False`
  - gather `future_outputs.hidden_states[-1]`
- `837-875`
  SGLang backend integration:
  - build per-sample augmented lists
  - call `extend(...)`
  - gather `future_last_hidden_states_list`
- `978-997`
  custom backend integration:
  - run one extra forward on augmented inputs
  - gather `future_outputs.last_hidden_state`

### 4.3 Online Eagle3 Loop Changes

Code references in `specforge/core/eagle3.py`:

- `132-145`
  extend `OnlineEagle3Model.forward(...)` with:
  - `future_hidden_states: Optional[torch.Tensor] = None`
- `238-253`
  align target-side future hidden states with TTT steps:
  - read `future_hidden_states[:, idx, :]`
  - project them through the draft model
  - pass the projected tensor into `draft_model.backbone(...)`

This means:

- step 0 uses the hidden state of the last real token
- step 1 uses the hidden state of the first appended MASK slot
- step 2 uses the hidden state of the second appended MASK slot

### 4.4 Draft Base Interface Changes

Code references in `specforge/modeling/draft/base.py`:

- `62-68`
  add `project_future_hidden_states(...)`
- `104-115`
  extend the abstract `backbone(...)` signature with:
  - `future_hidden_states`

This keeps future-hidden handling behind the draft abstraction.

### 4.5 Draft Model Changes

Code references in `specforge/modeling/draft/llama3_eagle.py`:

- `518-539`
  change attention input width from `2H` to `3H` when `use_future_hidden=True`
  - q/k/v input dimensions expand accordingly
- `1242-1267`
  add decoder-layer state:
  - `use_future_hidden`
  - `future_hidden_norm`
  - learned fallback `no_future_hidden`
- `1310-1321`
  perform the actual 3-way concatenation:
  - `[input_emb, hidden_states, future_hidden]`
  - fallback to `no_future_hidden` if no future hidden is provided
- `1352-1377`
  add draft-model-level projection support:
  - `use_future_hidden`
  - `future_hidden_proj`
- `1431-1435`
  pass `future_hidden_states` from the draft forward into the decoder layer
- `1461-1466`
  implement `project_future_hidden_states(...)`
  - project step-level target hidden to draft hidden size
- `1472-1488`
  extend `backbone(...)` to pass the projected future hidden into `midlayer(...)`

In other words:

- old attention input:
  - `[input_embedding, draft_hidden]`
- new attention input:
  - `[input_embedding, draft_hidden, future_hidden]`

### 4.6 End-to-End Behavior

For an example input:

```text
context = A B C D E
future  = F G H
```

the target-side augmented input becomes:

```text
A B C D E MASK1 MASK2
```

The gathered hidden states are:

- `h0 = hidden(E)`
- `h1 = hidden(MASK1)`
- `h2 = hidden(MASK2)`

Then:

- draft step 0 predicts `F` conditioned on `h0`
- draft step 1 predicts `G` conditioned on `h1`
- draft step 2 predicts `H` conditioned on `h2`

At each step, the draft decoder consumes:

```text
concat(input_embedding, draft_hidden, future_hidden_step)
```

That is the concrete implementation of the intended:

- target input = `context + MASK slots`
- target output = `NTP hidden + MASK hidden states`
- draft input = `embedding + draft hidden + future hidden`

</details>
