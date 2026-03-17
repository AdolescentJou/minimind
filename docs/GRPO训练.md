# GRPO 训练流程解析（train_grpo.py）

> **Group Relative Policy Optimization**：DeepSeek-R1 使用的对齐算法。不需要 Critic 模型，通过"组内相对排名"估计优势，比 PPO 更简洁高效。

---

## 一、什么是 GRPO？

PPO 需要一个额外的 Critic 模型来估计"期望奖励"，GRPO 的创新是：**对同一个 prompt 生成多个回复，用组内均值和标准差来替代 Critic**。

```
PPO:   优势 = reward - Critic(state)       ← 需要训练 Critic

GRPO:  优势 = (reward - group_mean) / group_std  ← 不需要 Critic！
       一个 prompt 生成 8 个回复，用这 8 个的均值/标准差做归一化
```

---

## 二、与 PPO 的差异总览

| 维度 | PPO | GRPO |
|------|-----|------|
| **Critic 模型** | 需要（训练+推理） | **不需要** |
| **模型数量** | 5 个 | **3 个**（Policy + Ref + Reward） |
| **优势估计** | Critic 模型 | **组内相对排名** |
| **每 prompt 生成数** | 1 个 | **8 个**（`num_generations`） |
| **显存需求** | 很高 | 较高（但省掉了 Critic） |
| **损失函数** | PPO Clip + Value Loss | **GRPO Policy Loss + KL** |
| **数据集** | RLAIFDataset | RLAIFDataset（相同） |
| **其余流程** | — | 初始化/AMP/DDP 等**完全一致**，略过 |

---

## 三、核心差异详解

### 3.1 三个模型

```python
# Policy 模型（要训练的）
model, tokenizer = init_model(lm_config, base_weight, device=args.device)

# Reference 模型（冻结基准）
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model.eval().requires_grad_(False)

# Reward 模型（外部，冻结）
reward_model = AutoModel.from_pretrained(args.reward_model_path, ...)
reward_model.eval().requires_grad_(False)
```

### 3.2 多回复生成

```python
# 每个 prompt 生成 num_generations=8 个回复
outputs = model.generate(
    **prompt_inputs,
    max_new_tokens=args.max_gen_len,
    do_sample=True,
    temperature=0.8,
    num_return_sequences=args.num_generations,  # 关键！一个 prompt → 8 个回复
    pad_token_id=tokenizer.pad_token_id
)
# outputs shape: [B * 8, P+R]
```

### 3.3 组内相对优势估计（核心）

```python
# rewards: [B * 8]
grouped_rewards = rewards.view(-1, args.num_generations)  # [B, 8]

# 每个 prompt 组内的均值和标准差
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(8)  # [B*8]
std_r = grouped_rewards.std(dim=1).repeat_interleave(8)    # [B*8]

# 组内归一化优势
advantages = clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
# 全局再归一化
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**直觉解释：**

```
一个 prompt 生成了 8 个回复，奖励分别是：
  [0.8, 0.3, -0.2, 0.5, 0.9, 0.1, -0.5, 0.6]

mean = 0.3125,  std = 0.46

归一化后的优势：
  [1.05, -0.03, -1.11, 0.41, 1.27, -0.46, -1.75, 0.62]

最好的回复（0.9 分）优势最大（+1.27）→ 被强化
最差的回复（-0.5 分）优势最小（-1.75）→ 被抑制
```

### 3.4 GRPO 损失

```python
# 每个 token 的 KL 散度
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = exp(kl_div) - kl_div - 1

# 每个 token 的损失 = 策略梯度 + KL 惩罚
per_token_loss = -(exp(logps - logps.detach()) * advantages.unsqueeze(1) - β * per_token_kl)

# 在有效 token 上取平均
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
loss = policy_loss + aux_loss
```

> 与 PPO 不同，GRPO 没有 Clip 操作，而是通过 KL 惩罚来约束策略更新幅度。

### 3.5 Completion Mask

```python
# 找到每个回复的 EOS 位置
is_eos = completion_ids == tokenizer.eos_token_id
eos_idx = is_eos.int().argmax(dim=1)

# 只对 EOS 之前（含 EOS）的 token 计算损失
completion_mask = (arange(R) <= eos_idx.unsqueeze(1)).int()
```

> 生成的回复长度不一，EOS 之后是 padding，不应参与损失计算。

---

## 四、关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_generations` | **8** | 每个 prompt 生成几个回复 |
| `beta` | 0.02 | KL 惩罚系数 |
| `learning_rate` | 8e-8 | 学习率 |
| `max_gen_len` | 1536 | 生成最大长度 |
| `max_seq_len` | 66 | Prompt 最大长度 |
| `batch_size` | 2 | 极小（每 prompt 生成 8 个回复，实际 batch=16） |
| `reasoning` | 1 | 是否启用推理模型模式 |

---

## 五、训练流程图

```
prompt: "天气如何？"
     │
     ▼ generate × 8
回复: ["今天晴", "不知道", "25℃", "很好", "适合出门", "有点冷", "??", "晴天"]
     │
     ▼ Reward 模型打分
rewards: [0.8, -0.5, 0.6, 0.7, 0.9, 0.1, -1.0, 0.5]
     │
     ▼ 组内归一化
advantages: [+0.8, -1.5, +0.3, +0.5, +1.0, -0.5, -2.0, +0.2]
     │
     ▼ Policy Loss + KL 惩罚
loss → backward → step（只更新 Policy）
```
