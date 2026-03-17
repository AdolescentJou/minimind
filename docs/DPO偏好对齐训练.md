# DPO 偏好对齐训练流程解析（train_dpo.py）

> **Direct Preference Optimization**：不需要训练奖励模型，直接从"好/坏回复对"中学习人类偏好。

---

## 一、什么是 DPO？

SFT 让模型学会"如何回答"，但不能保证回答"好不好"。DPO 通过对比"好回复 vs 坏回复"，让模型学会**偏好好回复、回避坏回复**。

```
同一个问题的两种回答：

Chosen（好）:  "北京今天晴，25℃，适合外出。"
Rejected（坏）: "我不知道，你自己查吧。"

DPO 的目标：增大 Chosen 的概率，减小 Rejected 的概率
```

---

## 二、与预训练/微调的差异总览

| 维度 | 微调（SFT） | DPO |
|------|-----------|-----|
| **数据格式** | 单条对话 | **好/坏回复对** |
| **数据集类** | SFTDataset | **DPODataset** |
| **模型数量** | 1 个 | **2 个**（策略模型 + 冻结的参考模型） |
| **损失函数** | 交叉熵 | **DPO Loss**（基于对数概率比） |
| **学习率** | 1e-6 | **4e-8**（极小，防止遗忘） |
| **额外超参** | — | `beta`（偏好强度） |
| **其余流程** | — | 初始化/AMP/DDP/保存等**完全一致**，略过 |

---

## 三、核心差异详解

### 3.1 DPO 数据格式（DPODataset）

```json
{
  "chosen": [
    {"role": "user", "content": "天气如何？"},
    {"role": "assistant", "content": "今天晴，25℃。"}
  ],
  "rejected": [
    {"role": "user", "content": "天气如何？"},
    {"role": "assistant", "content": "不知道。"}
  ]
}
```

DPODataset 输出的不是 `(input_ids, labels)`，而是一个字典：

```python
{
    'x_chosen': tensor,      # chosen 的 input（去掉最后一个 token）
    'y_chosen': tensor,      # chosen 的 label（去掉第一个 token）
    'mask_chosen': tensor,   # chosen 的 loss mask（只标记助手回复部分）
    'x_rejected': tensor,    # rejected 的 input
    'y_rejected': tensor,    # rejected 的 label
    'mask_rejected': tensor  # rejected 的 loss mask
}
```

### 3.2 两个模型：策略模型 + 参考模型

```python
# 策略模型（policy）—— 要训练的
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 参考模型（reference）—— 冻结，作为"基准线"
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)
```

> **为什么需要参考模型？** DPO 不是简单地"提高好回复概率"，而是要和参考模型比较。如果没有参考模型，策略模型可能会"跑偏"，输出越来越极端的文本。

### 3.3 DPO 损失函数（核心）

```python
def logits_to_log_probs(logits, labels):
    """把 logits 转成每个 token 的对数概率"""
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token

def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 按 mask 计算序列级平均对数概率
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths

    # 分离 chosen 和 rejected（前半是 chosen，后半是 rejected）
    chosen_policy = policy_log_probs[:B // 2]
    reject_policy = policy_log_probs[B // 2:]
    chosen_ref = ref_log_probs[:B // 2]
    reject_ref = ref_log_probs[B // 2:]

    # 策略模型的偏好差异
    pi_logratios = chosen_policy - reject_policy
    # 参考模型的偏好差异
    ref_logratios = chosen_ref - reject_ref

    # DPO 的核心公式
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

**DPO 公式的直觉解释：**

```
DPO loss = -log σ(β × ((π_chosen - π_rejected) - (ref_chosen - ref_rejected)))

π_chosen:   策略模型给好回复的概率
π_rejected: 策略模型给坏回复的概率
ref_chosen: 参考模型给好回复的概率
ref_rejected: 参考模型给坏回复的概率

目标：让策略模型比参考模型"更偏好"好回复
  → π_chosen - π_rejected > ref_chosen - ref_rejected
  → loss 趋近 0

β 越大 → 偏好信号越强 → 优化越激进
```

### 3.4 训练步骤

```python
# 1. 拼接 chosen 和 rejected
x = torch.cat([x_chosen, x_rejected], dim=0)      # [2B, T]
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)

# 2. 参考模型前向（frozen）
with torch.no_grad():
    ref_logits = ref_model(x).logits
ref_log_probs = logits_to_log_probs(ref_logits, y)

# 3. 策略模型前向
logits = model(x).logits
policy_log_probs = logits_to_log_probs(logits, y)

# 4. 计算 DPO loss
loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1)

# 5. 反向传播（只更新策略模型）
loss.backward()
```

---

## 四、关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | 0.1 | 偏好强度（越大越激进） |
| `learning_rate` | **4e-8** | 极小学习率（比 SFT 的 1e-6 还小 25 倍） |
| `batch_size` | 4 | 很小（每个样本包含 chosen+rejected） |
| `from_weight` | `full_sft` | 基于 SFT 权重 |
| `max_seq_len` | 1024 | 比 SFT 更长（对话对更占空间） |

---

## 五、训练流程图

```
DPO 数据: {chosen, rejected}
       │
       ▼
  DPODataset → (x_chosen, y_chosen, mask_chosen,
                 x_rejected, y_rejected, mask_rejected)
       │
       ▼
  拼接: x = cat(chosen, rejected)  →  [2B, T]
       │
  ┌────┴────┐
  ▼         ▼
策略模型   参考模型(frozen)
  │         │
  ▼         ▼
π_logprobs  ref_logprobs
  │         │
  └────┬────┘
       ▼
  DPO Loss = -log σ(β × (π偏好差 - ref偏好差))
       │
       ▼
  backward → clip → step（只更新策略模型）
```
