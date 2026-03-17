# LoRA 微调训练流程解析（train_lora.py）

> 只训练极少量新增参数（约 0.5%），冻结原模型，实现低成本微调。

---

## 一、什么是 LoRA？

LoRA（Low-Rank Adaptation）的核心思想：**不改原模型的权重，而是给部分层"旁路"一个小矩阵**。

```
原始 Linear 层:     y = Wx

LoRA 后:           y = Wx + BAx
                        ↑    ↑
                     原权重  LoRA旁路
                    (冻结)  (可训练)

其中: W 是 [out, in] 的大矩阵（冻结）
      A 是 [rank, in]  的小矩阵（可训练）
      B 是 [out, rank]  的小矩阵（可训练）
      rank << in, out  所以参数量极少
```

---

## 二、与 Full SFT 的差异总览

| 维度 | Full SFT | LoRA SFT |
|------|----------|----------|
| **可训练参数** | 全部（100%） | **仅 LoRA 参数（~0.5%）** |
| **原模型权重** | 全部更新 | **冻结不变** |
| **显存占用** | 大 | **极小** |
| **保存内容** | 完整模型权重 | **仅 LoRA 权重** |
| **学习率** | 1e-6 | **1e-4**（LoRA 参数少，可以用更大的 lr） |
| **训练轮数** | 2 | **50**（参数少需要更多轮） |
| **数据集** | SFTDataset | SFTDataset（相同） |
| **损失函数** | 交叉熵 | 交叉熵（相同） |
| **训练循环** | 标准 | 几乎相同（梯度裁剪只对 lora_params） |
| **其余流程** | — | 初始化/AMP/DDP 等**完全一致**，略过 |

---

## 三、核心差异详解

### 3.1 LoRA 网络结构（model/model_lora.py）

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)   # 降维
        self.B = nn.Linear(rank, out_features, bias=False)   # 升维
        self.A.weight.data.normal_(mean=0.0, std=0.02)       # A 高斯初始化
        self.B.weight.data.zero_()                            # B 全零初始化

    def forward(self, x):
        return self.B(self.A(x))  # x → A → B（低秩变换）
```

> **B 全零初始化**：训练开始时 LoRA 的输出为 0，即模型行为和原模型完全一致。训练过程中 B 逐渐学到非零值，模型行为逐步变化。

### 3.2 将 LoRA 注入模型

```python
def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # 只对"方阵"Linear 层加 LoRA（通常是 Attention 的 Q/K/V/O 投影）
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank)
            setattr(module, "lora", lora)

            original_forward = module.forward
            # 新的 forward = 原始输出 + LoRA 输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora
```

> 这里只对**方阵 Linear**（输入输出维度相同的层）加 LoRA，在 MiniMind 中主要是 Attention 的 `q_proj` 和 `o_proj`（因为它们是 `[H, H]` 的方阵）。

### 3.3 冻结原模型，只训练 LoRA

```python
# 在 train_lora.py 中
apply_lora(model)

lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True    # LoRA 参数可训练
        lora_params.append(param)
    else:
        param.requires_grad = False   # 原模型参数冻结

# 优化器只优化 LoRA 参数
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

> 典型输出：
> ```
> LLM 总参数量: 26.878 M
> LoRA 参数量: 0.131 M
> LoRA 参数占比: 0.49%
> ```

### 3.4 梯度裁剪只对 LoRA 参数

```python
# Full SFT 裁剪所有参数：
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# LoRA 只裁剪 LoRA 参数：
torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
```

### 3.5 保存：只存 LoRA 权重

```python
# Full SFT 保存完整模型
torch.save(raw_model.state_dict(), ckp)

# LoRA 只保存 LoRA 权重（文件极小）
save_lora(model, lora_save_path)
```

`save_lora` 的实现：

```python
def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

---

## 四、关键超参数

| 参数 | 默认值 | 与 Full SFT 对比 |
|------|--------|-----------------|
| `learning_rate` | **1e-4** | SFT 的 1e-6 的 100 倍 |
| `epochs` | **50** | SFT 的 2 轮的 25 倍 |
| `lora_name` | `lora_identity` | 可按任务命名 |
| `save_dir` | `out/lora/` | 独立保存目录 |
| `log_interval` | 10 | 更频繁打印（数据少） |
| `from_weight` | `full_sft` | 基于 SFT 权重 |
| `data_path` | `lora_identity.jsonl` | LoRA 专用数据（通常很少） |

---

## 五、使用 LoRA 权重推理

```python
# 1. 加载原模型
model = MiniMindForCausalLM(config)
model.load_state_dict(torch.load('full_sft_512.pth'))

# 2. 注入 LoRA 结构
apply_lora(model)

# 3. 加载 LoRA 权重
load_lora(model, 'lora/lora_identity_512.pth')

# 4. 推理
output = model.generate(...)
```
