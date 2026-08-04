## 1. 环境配置
```bash
# 克隆代码仓库
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

conda create -n spec python=3.11 -y  
conda activate spec
uv pip install -e . --prerelease=allow # -e 可编辑模式 会创建软链接，方便本地开发
```

## 2. 启动脚本说明

在 examples 目录下，提供了样例启动脚本，以 `run_qwen3_8b_eagle3_online.sh` 为例。
### 脚本内容
```bash
#!/bin/bash

# ============================================
# 环境变量配置
# ============================================
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# ============================================
# 训练参数配置
# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
# ============================================
NUM_GPUS=${1:-1}          # 第一个参数：GPU数量
TP_SIZE=${2:-1}           # 第二个参数：模型并行度
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# ============================================
# 模型与数据路径配置
# ============================================
TARGET_MODEL_PATH=Qwen/Qwen3-8B
DRAFT_CONFIG=$ROOT_DIR/configs/qwen3-8b-eagle3.json
TRAIN_DATA=$ROOT_DIR/cache/dataset/sharegpt_train.jsonl
OUTPUT_DIR=$ROOT_DIR/outputs/qwen3-8b-eagle3-sharegpt

# ============================================
# 启动训练
# ============================================
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $DRAFT_CONFIG \
    --train-data-path $TRAIN_DATA \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang
```

## 3. 复现 MNN 训练的 Eagle3

根据上面脚本可以尝试复现 [MNN](https://huggingface.co/taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3) 训练的 Eagle3。

### 3.1 下载模型
下载模型 Qwen3-4B-Instruct-2507 到本地，并将路径替换脚本中 `--target-model-path` 参数的值。

### 3.2 下载训练数据
下载数据集 [EagleChat](https://huggingface.co/datasets/zhaode/EagleChat) 到本地，其中数据格式为 jsonl 格式符合[框架要求](https://docs.sglang.io/SpecForge/basic_usage/data_preparation.html)，将数据文件路径替换脚本中 `--train-data-path` 参数的值。

### 3.3 创建草稿模型配置文件
创建一个草稿模型的配置文件，大部分直接复制目标模型的配置文件，需要修改的内容有：
- `architectures`: 草稿模型的结构在 `SpecForge` 中需要指定为 `LlamaForCausalLMEagle3`，而不是目标模型的结构。
- `model_type`: 同上原因修改为 `llama`
- `num_hidden_layers`: 根据需要修改训练的草稿模型的层数，`Eagle3` 是 1 层。
- `draft_vocab_size`: 新添加草稿模型词表，这可以是一个压缩词表。

### 3.4 创建启动脚本
```bash
#!/bin/bash

# ============================================
# 环境变量配置
# ============================================
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# ============================================
# 训练参数配置
# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
# ============================================
NUM_GPUS=${1:-1}          # 第一个参数：GPU数量
TP_SIZE=${2:-1}           # 第二个参数：模型并行度
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# ============================================
# 模型与数据路径配置
# ============================================
TARGET_MODEL_PATH=/data/HUGGINGFACE/Qwen3-4B-Instruct-2507
DRAFT_CONFIG=$ROOT_DIR/configs/qwen3-4b-eagle3.json
TRAIN_DATA=/data/HUGGINGFACE/data/EagleChat/eagle_chat.jsonl
BASE_OUTPUT_DIR=$ROOT_DIR/outputs/qwen3-4b-eagle3
CKPT_DIR=${CKPT_DIR:-$BASE_OUTPUT_DIR/epoch_1_step_270000}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-4b-eagle3-continue-epoch1-step270000}
MAX_LENGTH=${MAX_LENGTH:-1792}

CKPT_ARGS=()
if [ -n "$CKPT_DIR" ]; then
    CKPT_ARGS+=(--ckpt-dir "$CKPT_DIR")
fi

# ============================================
# 启动训练
# ============================================
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $DRAFT_CONFIG \
    --train-data-path $TRAIN_DATA \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    "${CKPT_ARGS[@]}" \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length $MAX_LENGTH \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang
```

### 3.5 启动训练
```bash
# 进入代码存放目录 并激活环境
conda activate spec
CUDA_VISIBLE_DEVICES=0,1,2,4 ./examples/run_qwen3_4b_eagle3_online.sh 4 1
```

### 3.6 从已有 checkpoint 断点续训

断点续训需要满足两点：

- 使用原来的 `--output-dir`。
- 启动参数里加 `--resume`，训练脚本会在 `output_dir` 下自动查找最新的 `epoch_*_step_*` 目录，并加载其中的checkpoint: `training_state.pt`(**默认脚本不会读取第三个参数，需要手动修改**)。

以本次 Qwen3-1.7B 训练为例，使用 2、3、4、5 号 GPU，`4 1` 表示 `NUM_GPUS=4`、`TP_SIZE=1`，因此 `dp_size=4`。日志用 `tee -a` 追加到同一个输出目录，不会覆盖旧日志：

```bash
cd /workspace/code/SpecForge
mkdir -p /workspace/code/SpecForge/outputs/qwen3-1p7b-eagle3-sharegpt

# 激活虚拟环境
conda activate spec
# 启动训练
CUDA_VISIBLE_DEVICES=2,3,4,5 MASTER_PORT=29511 BUILD_DATASET_NUM_PROC=64 \
bash ./examples/run_qwen3_1p7b_eagle3_online.sh 4 1 --resume \
2>&1 | tee -a /workspace/code/SpecForge/outputs/qwen3-1p7b-eagle3-sharegpt/train.log"'
```



## 4. Eagle3 实现分析

按当前仓库里的真实代码路径梳理 Eagle3 的训练实现。对应的核心入口主要是：
- `examples/run_qwen3_4b_eagle3_online.sh`
- `scripts/train_eagle3.py`
- `specforge/core/eagle3.py`
- `specforge/modeling/draft/llama3_eagle.py`
- `specforge/modeling/target/eagle3_target_model.py`
- `specforge/data/preprocessing.py`
- `scripts/prepare_hidden_states.py`


### 4.1 训练入口与整体流程

以 `examples/run_qwen3_4b_eagle3_online.sh` 为例，最终会用 `torchrun` 启动 `scripts/train_eagle3.py`。训练脚本的主流程可以概括为：

1. 解析参数并初始化分布式环境。
2. 构建 draft model。
3. 构建 target model 或 offline target head。
4. 处理训练数据并建立 dataloader。
5. 为 draft model 加载 vocab mapping。
6. 构造 `OnlineEagle3Model` 或 `QwenVLOnlineEagle3Model`。
7. 用 FSDP 包装训练模型，用 `BF16Optimizer` 做优化。
8. 进入训练循环，执行 forward、TTT 多步展开、反向传播、评估与保存检查点。

对应到 `scripts/train_eagle3.py`，代码里通过下面这个条件区分两种模式：

```python
is_online = (
    args.train_data_path is not None and args.train_hidden_states_path is None
)
```

也就是说：

- `online` 模式：直接读取对话数据，训练时实时调用 target model 生成 aux hidden states 和 target logits。
- `offline` 模式：训练前先执行 `scripts/prepare_hidden_states.py`，把 target model 的 hidden states 预先落盘；训练时不再加载完整 target model，只加载 `TargetHead` 把离线保存的最后 hidden state 投到词表 logits。

### 4.2 分布式与并行策略

`specforge/distributed.py` 里把训练相关并行拆成了几类：

- `TP`：target model 的 tensor parallel。
- `DP`：常规 data parallel。
- `draft_dp + sp`：给 draft model 的数据并行和 sequence parallel 使用。

`init_distributed()` 会同时初始化两套 device mesh：

- `(dp_size, tp_size)`：服务于 target model。
- `(draft_dp_size, sp_ulysses_size * sp_ring_size)`：服务于 draft model 的 sequence parallel / USP。

这里有一个实现上的关键点：

- target model 可能以 TP 方式运行。
- 但 draft model 的训练是按全局 DP/FSDP 来跑的。
- 因此 `run_forward()` 中会先调用 target model 生成整批 Eagle3 数据，再通过 `get_dp_data_shard_from_tp()` 把这些张量沿 batch 维切分给对应 TP rank，交给 draft model 训练。

在真正训练 draft model 时，脚本会把 `eagle3_model` 用 FSDP 包起来：

```python
eagle3_model = FSDP(
    eagle3_model,
    use_orig_params=True,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    process_group=dist.group.WORLD,
)
```

所以从实现上说：

- target model 负责“在线提供监督信号”。
- draft model 才是实际被 FSDP/optimizer 更新的训练主体。

### 4.3 数据预处理与 loss mask

在线训练的数据入口在 `build_dataloaders()` 中，会调用 `build_eagle3_dataset()`。其预处理逻辑在 `specforge/data/preprocessing.py` 和 `specforge/data/parse.py` 中。

对于纯文本对话，数据处理分几步：

1. 根据 `chat_template` 把 ShareGPT 风格的 `conversations` 渲染成模型输入文本。
2. 用 tokenizer 编码得到 `input_ids`。
3. 通过 parser 定位 assistant 回复区域，只让 assistant token 参与 loss。
4. 构造：
   - `input_ids`
   - `loss_mask`
   - `attention_mask`

`train_only_last_turn=True` 时，parser 会只保留最后一个 assistant span，对应“只训练最后一轮回复”的场景，这对 thinking model 很有用，因为历史消息里可能并不包含完整思维链。

对于 VLM，`preprocess_vlm_conversations()` 还会额外构造：

- `pixel_values`
- `image_grid_thw`

### 4.4 在线监督信号是怎么生成的

在线训练真正的监督信号不是硬标签 token id，而是 target model 输出的 soft target distribution。

`run_forward()` 在 online 模式下会调用：

```python
eagle3_data = target_model.generate_eagle3_data(...)
```

这个 `Eagle3TargetOutput` 里包含：

- `hidden_states`：来自 target model 的 3 个辅助层 hidden state 拼接结果。
- `target`：target model 的 logits。
- `loss_mask`
- `input_ids`
- `attention_mask`

无论是 HF backend、SGLang backend 还是 custom backend，它们都会做同一件核心事情：

1. 选出 3 个辅助层。
2. 把这 3 层 hidden state 在最后一维上拼接。
3. 把 target logits 和 input_ids 做一次 `padding(..., left=False)`，使监督目标对齐到“预测下一个 token”的位置。

默认的 3 个辅助层来自 `set_aux_hidden_states_layers()`：

- 第 1 层
- 中间层 `num_layers // 2 - 1`
- 倒数前第 4 层 `num_layers - 4`

也就是典型的 low / mid / late 三层特征融合。

### 4.5 Draft 模型结构

当前仓库里 Eagle3 draft model 的自动加载入口是 `specforge/modeling/auto.py`，其中：

```python
class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
    }
```

也就是说，当前仓库里的 Eagle3 draft model 实际上统一映射到 `LlamaForCausalLMEagle3` 这套实现。

以 `configs/qwen3-4b-eagle3.json` 为例，配置里会写：

- `architectures = ["LlamaForCausalLMEagle3"]`
- `model_type = "llama"`
- `num_hidden_layers = 1`
- `draft_vocab_size = 32000`

但要注意一个实现细节：

- 配置里虽然有 `num_hidden_layers`。
- 实际类 `LlamaForCausalLMEagle3` 只显式实例化了一个 `self.midlayer = LlamaDecoderLayer(...)`。
- 所以当前仓库的 Eagle3 draft model 是一个“单层 decoder 草稿模型”的实现，而不是按 `num_hidden_layers` 动态堆叠任意层数。

这个模型包含以下关键部件：

1. `embed_tokens`
   - 大小是 `vocab_size x hidden_size`
   - 训练前通过 `load_embedding()` 从 target model 里加载 embedding
   - 然后通过 `freeze_embedding()` 冻结，不参与训练

2. `fc`
   - 把 3 个辅助层拼接得到的 hidden states 从 `3 * hidden_size` 投影回 `hidden_size`
   - 如果配置里单独指定了 `target_hidden_size`，则按 `3 * target_hidden_size -> hidden_size` 投影

3. `midlayer`
   - 一个 `LlamaDecoderLayer`
   - 是 draft model 的唯一主干层

4. `norm + lm_head`
   - `norm` 是 `LlamaRMSNorm`
   - `lm_head` 把 hidden state 映射到 `draft_vocab_size`

5. `t2d / d2t`
   - `t2d`：target vocab 到 draft vocab 的掩码
   - `d2t`：draft vocab 到 target vocab 的映射偏移
   - 这两个 buffer 通过 `load_vocab_mapping()` 加载

### 4.6 Eagle3 的特征融合方式

Eagle3 的核心不是只看 token embedding，而是把 target model 的 feature 引入 draft model。

实际实现顺序如下：

1. target model 输出 3 个辅助层 hidden states，并在最后一维拼接。
2. `draft_model.project_hidden_states()` 用 `fc` 把它投影到 `hidden_size`。
3. `draft_model.embed_input_ids()` 对当前 token 序列做 embedding。
4. 在 `LlamaDecoderLayer.forward()` 里：
   - 先分别对 `hidden_states` 和 `input_emb` 做 RMSNorm
   - 再执行：

```python
hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
```

5. `LlamaAttention` 的 `q_proj/k_proj/v_proj` 输入维度都是 `hidden_size * 2`，说明自注意力直接消费“token embedding + target feature”拼接后的表示。

所以这里不是简单相加，而是：

- 先把 target 的 3 层特征压到一份 `hidden_size`
- 再和 token embedding 拼接成 `2 * hidden_size`
- 然后进入唯一的一层 decoder attention + MLP

这就是仓库里 Eagle3 的 feature-based drafting 的具体落地方式。

### 4.7 Draft vocab 压缩是怎么做的

Eagle3 不一定在完整 target vocab 上训练，而是可以只保留一个较小的 `draft_vocab_size`。

`build_dataloaders()` 在构建完数据集后，会调用 `generate_vocab_mapping_file()`：

1. 遍历数据集。
2. 只统计 `loss_mask == 1` 的有效 token。
3. 找到出现频率最高的 `draft_vocab_size` 个 token。
4. 生成：
   - `d2t`
   - `t2d`

后续训练里，`_compute_target_p()` 会先把 target logits 映射到压缩后的 draft vocab 空间：

```python
target_max_token = target_head.argmax(-1)
target_mask = t2d[target_max_token]
position_mask = target_mask * loss_mask
target_head = target_head[..., t2d]
target_p = nn.Softmax(dim=2)(target_head)
```

这段实现有两个直接影响：

1. loss 不是对完整 target vocab 做，而是在 draft vocab 子集上做 softmax。
2. 如果某个位置 target model 的 top-1 token 不在 draft vocab 里，那么这个位置的 `position_mask` 会变成 0，不参与损失与准确率统计。

这也是为什么 `draft_vocab_size` 的选择会直接影响可训练覆盖率。

### 4.8 TTT: Test-Time Training 在代码里怎么实现

TTT 的实现主逻辑在 `specforge/core/eagle3.py` 的 `OnlineEagle3Model.forward()`。

先看整体思路：

1. 先把 target logits 映射成 draft vocab 上的 soft target distribution。
2. 然后做一个长度为 `ttt_length` 的循环展开。
3. 每一步都让 draft model 基于当前输入和上一步隐藏状态继续向前“滚动”。
4. 对每一步都计算一个位置对齐后的损失和准确率。
5. 最后把多个 step 的 loss 按权重求和回传。

对应代码里的关键步骤如下。

#### 4.8.1 先把 target logits 变成 soft label

`_compute_target_p_padded()` 会调用 `_compute_target_p()`：

- 先用 `t2d` 把 target logits 裁剪到 draft vocab 子空间。
- 再对这个子空间做 softmax，得到 `target_p`。
- 同时构造 `position_mask`。

随后再执行：

```python
target_p_padded = F.pad(target_p, pad=(0, 0, 0, length), mode="constant", value=1 / target_p.shape[-1])
```

也就是在序列末尾多补出 `ttt_length` 个位置，方便后面第 `idx` 步直接取：

```python
target_p = target_p_padded[:, idx : idx + seq_length, :]
```

#### 4.8.2 多步展开

TTT 的核心循环是：

```python
for idx in range(self.length):
    ...
```

每一步会做：

1. 取出当前 step 对应的 `target_p` 窗口。
2. 用当前 `input_ids` 生成 embedding。
3. 用当前 `hidden_states` 和 embedding 进入 draft backbone。
4. 产出新的 hidden states。
5. 用 `lm_head` 得到 logits。
6. 和当前 step 的 `target_p` 计算 loss / acc。

如果不是最后一步，还会执行：

```python
global_input_ids = padding(global_input_ids, left=False)
position_mask = padding(position_mask, left=False)
loss_mask = padding(loss_mask, left=False)
```

这个左移操作表示：

- 下一步训练时，监督窗口向后滑动一格。
- 当前序列也同步向后对齐。
- 这样就模拟了连续多步 draft 生成时的训练场景。

同时，`hidden_states = hidden_states_out` 会把上一步 draft model 的输出作为下一步的输入隐藏状态，真正形成“滚动展开”。

#### 4.8.3 cache / KV 的作用

TTT 循环里不同 attention backend 会采用不同缓存策略：

- `sdpa / fa / usp`：使用 `cache_hidden = [[], []]`
- `flex_attention`：使用 `DynamicCache()`

这意味着多步 TTT 不是每一步都完全重新算一遍，而是通过缓存把前面 step 的上下文延续下来，更接近真实推理时的连续 drafting 过程。

#### 4.8.4 多步 loss 的加权方式

反向传播前，`run_backward_and_update()` 会把多步 loss 做衰减加权：

```python
ploss_weight = [0.8**i for i in range(len(plosses))]
ploss = (
    sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
    / args.draft_accumulation_steps
)
```

也就是说：

- 第 0 步权重最大
- 越往后的 rollout step 权重越小

这很符合 TTT 的直觉：既希望模型学会短期高质量起草，也希望它能承受多步滚动误差，但不会让远端 step 完全主导训练。

### 4.9 Loss 与 accuracy 的具体定义

这里的 loss 不是普通的硬标签交叉熵，而是对 target model 的 soft distribution 做蒸馏式拟合。

实现位于 `specforge/core/loss.py`：

- `LogSoftmaxLoss` 是一个 Triton 自定义算子。
- 前向等价于：

```python
out_logp = nn.LogSoftmax(dim=2)(logits)
plogp = target_p * out_logp
loss = -torch.sum(position_mask * plogp, 2).mean()
```

这本质上是在做 target soft label 与 draft logits 之间的 cross-entropy / KL 风格训练，只不过实现成了更省显存、更快的 fused kernel。

accuracy 的定义也不是看完整词表，而是：

- 先比较 `draft logits.argmax(-1)` 和 `target_p.argmax(-1)`
- 再乘上 `position_mask`
- 最后除以有效 `loss_mask` 数量

因此这里的 acc 更准确地说是：

- “在可参与训练的位置上，draft vocab 空间里的 top-1 是否与 target vocab 压缩后的 top-1 一致”

### 4.10 Online 与 Offline 两条训练路径的差异

#### 4.10.1 Online

online 模式直接在训练时调用 target model：

- 优点：监督信号和服务时使用的 target model 完全一致，省去了 hidden state 预处理步骤。
- 代价：训练时必须同时加载 target model，显存和时延开销更高。

#### 4.10.2 Offline

offline 模式需要先运行 `scripts/prepare_hidden_states.py`。

这个脚本会：

1. 读取训练数据。
2. 调 target model 生成：
   - `aux_hidden_state`
   - `hidden_state`
   - `input_ids`
   - `loss_mask`
3. 把每个样本单独保存成 `.ckpt` 或 `.ckpt.gz` 文件。

训练时 `build_offline_eagle3_dataset()` 会读取这些文件。这里字段名字要注意：

- `aux_hidden_state`：作为 draft model 的输入 feature。
- `hidden_state`：作为 `TargetHead` 的输入，映射回 target logits。

所以 offline 训练并不是直接把完整 target logits 存盘，而是：

- 存最后一层 hidden state
- 训练时再通过 `TargetHead.fc` 恢复到 vocab logits

这样做的好处是磁盘占用远小于直接存整张 logits。

另外，offline 模式也为 USP 做了专门预处理：

- `OfflineEagle3Dataset.process_data_usp()` 会先按 sequence parallel 切分序列。
- 每个 shard 额外保留 `ttt_length` 的 overlap，用来支持 TTT 展开。

### 4.11 Optimizer、checkpoint 与恢复

#### 4.11.1 优化器

`specforge/optimizer.py` 里的 `BF16Optimizer` 使用的是：

- FP32 master params
- AdamW
- CosineAnnealingWarmupLR

更新流程是：

1. 从 BF16 模型参数里取梯度。
2. 拷到 FP32 master params。
3. 做 grad clip。
4. 执行 AdamW step 和 scheduler step。
5. 再把 FP32 参数拷回 BF16 模型。

这是一个比较典型的 BF16 训练做法，目的是兼顾数值稳定性和显存占用。

#### 4.11.2 checkpoint

`save_checkpoints()` 保存两个东西：

1. `training_state.pt`
   - epoch
   - global_step
   - args
   - optimizer state
   - scheduler state

2. draft model 权重

不过它保存 draft model 时会显式过滤掉 embedding 相关参数：

```python
draft_model_state_dict = {
    k.replace("draft_model.", ""): v
    for k, v in model_state_dict.items()
    if "draft_model." in k and "embed" not in k.lower()
}
```

也就是说 checkpoint 默认不保存 embedding，本质上依赖后续继续从 target model 重新加载 embedding。

### 4.12 当前仓库里可以直接得出的结论

综合代码实现，可以把当前仓库对 Eagle3 的落地总结为下面几点：

1. Eagle3 的 draft model 是一个特征驱动的单层 decoder 草稿模型。
2. 训练监督不是硬标签，而是 target model 在压缩 vocab 空间上的 soft distribution。
3. 训练时显式做了长度为 `ttt_length` 的多步 rollout，这就是仓库里的 TTT。
4. TTT 的多步 loss 采用 `0.8^i` 的衰减加权。
5. 只有 target top-1 落在 draft vocab 内的位置才真正参与损失与准确率统计。
6. 在线模式实时调 target model；离线模式提前落 hidden state，再用 `TargetHead` 恢复 logits。
7. 分布式上 target model 主要结合 TP 使用，draft model 主要结合 FSDP/DP/USP 使用。

### 4.13 一个便于理解的最小训练闭环

把当前仓库的 Eagle3 训练抽象成一条最小路径，可以理解为：

```text
对话数据
  -> chat template + tokenizer
  -> input_ids / loss_mask
  -> target model 提取 3 层 aux hidden states + target logits
  -> draft model:
       project(3-layer features)
       + embed(input_ids)
       + single decoder layer
       + lm_head(draft_vocab)
  -> 做 ttt_length 次滚动展开
  -> 对每一步的 soft target 计算蒸馏损失
  -> 加权求和后反向传播
```

### 4.14 仓库中的测试覆盖

`tests/test_scripts/test_train_eagle3.py` 里已经覆盖了几条主线：

- online + sglang backend
- online + hf backend
- online + custom backend
- offline training

因此就仓库当前设计来说，Eagle3 的训练实现并不是只停留在文档级别，而是已经按这几条路径组织了基本测试。
