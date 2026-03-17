# PPO 训练流程解析（train_ppo.py）

> **Proximal Policy Optimization**：经典的强化学习对齐算法。模型自己生成回复 → 奖励模型打分 → 根据奖励优化策略。
> 是 ChatGPT 最初使用的 RLHF 核心算法。

---

## 一、什么是 PPO？

DPO 从"现成的好坏对"学习，PPO 则让模型**自己生成回复**，然后由奖励模型评判好坏，再根据评判结果调整策略。

```
PPO 训练循环：
  prompt: "天气如何？"
      │
      ▼
  Actor 模型自己生成回复 → "今天晴天，25℃"
      │
      ▼
  Reward 模型打分 → 0.8 分（好）
      │
      ▼
  Critic 模型估计"期望分数" → 0.6
      │
      ▼
  优势 = 实际分 - 期望分 = 0.2（"比预期好"）
      │
      ▼
  用优势信号更新 Actor 和 Critic
```

---

## 二、与 DPO 的差异总览

| 维度 | DPO | PPO |
|------|-----|-----|
| **数据** | 离线好/坏回复对 | **在线生成**（模型自己生成） |
| **模型数量** | 2 个（策略 + 参考） | **5 个**（Actor + Old Actor + Ref + Critic + Reward） |
| **损失函数** | DPO Loss | **PPO Clip Loss + Value Loss + KL 惩罚** |
| **数据集** | DPODataset | **RLAIFDataset**（只提供 prompt，不提供回复） |
| **训练复杂度** | 中 | **高**（需要采样生成 + 多模型协作） |
| **学习率调度** | 手动余弦 | **CosineAnnealingLR**（PyTorch 内置调度器） |
| **其余流程** | — | 初始化/AMP/DDP 等**完全一致**，略过 |

---

## 三、五个模型各自的角色

```python
# 1. Actor 模型（策略模型）—— 要训练的主模型
actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)

# 2. Old Actor —— Actor 的历史快照，用于计算重要性采样比率
old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
old_actor_model.eval().requires_grad_(False)

# 3. Reference 模型 —— 冻结的基准模型，防止策略跑偏
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model.eval().requires_grad_(False)

# 4. Critic 模型 —— 价值网络，估计"期望奖励"
critic_model = CriticModel(lm_config)  # 自定义，lm_head 替换为 value_head

# 5. Reward 模型 —— 外部奖励模型（internlm2-1_8b-reward）
reward_model = AutoModel.from_pretrained(args.reward_model_path, ...)
reward_model.eval().requires_grad_(False)
```

### CriticModel 的结构

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换语言模型的输出头为价值头
        self.value_head = nn.Linear(params.hidden_size, 1)  # 输出单个标量

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)  # [B, T] → [B, T]
        return values
```

> Critic 和 Actor 共享 Transformer 主干结构，但输出头不同：Actor 输出词表概率（生成文本），Critic 输出标量值（估计奖励）。

---

## 四、奖励计算

```python
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    rewards = torch.zeros(len(responses))

    # 1. 格式奖励（仅推理模型模式）
    if args.reasoning == 1:
        # 检查 <think>...</think><answer>...</answer> 格式
        # 格式正确 +0.5，每个标记正确 +0.25

    # 2. Reward 模型打分
    for prompt, response in zip(prompts, responses):
        messages = parse_chat(prompt)
        messages += [{"role": "assistant", "content": response}]
        score = reward_model.get_score(reward_tokenizer, messages)
        score = clamp(score, -3.0, 3.0)  # 截断到 [-3, 3]

        # 推理模式下额外评估 <answer> 内容
        if args.reasoning == 1:
            answer_score = reward_model.get_score(...)
            score = score * 0.4 + answer_score * 0.6

    return rewards
```

---

## 五、PPO 损失（核心公式）

```python
# 1. 计算优势（Advantage）
values = critic_model(gen_out)           # Critic 估计的期望奖励
advantages = rewards - values.detach()    # 优势 = 实际 - 期望

# 2. 重要性采样比率
ratio = exp(actor_logp - old_actor_logp)  # Actor vs Old Actor

# 3. PPO Clip Loss（策略损失）
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-ε, 1+ε) * advantages  # ε=0.1
policy_loss = -min(surr1, surr2).mean()

# 4. Value Loss（价值损失）
value_loss = MSE(values, rewards)

# 5. KL 惩罚（防止偏离参考模型太远）
kl_ref = (actor_logp - ref_logp).mean()

# 6. 总损失
loss = policy_loss + vf_coef * value_loss + kl_coef * kl_ref
```

**PPO Clip 的直觉：**

```
ratio > 1+ε 时被裁剪 → 防止策略变化太大（太激进）
ratio < 1-ε 时被裁剪 → 防止策略变化太大（太保守）

ε = 0.1 意味着每次更新，策略变化不超过 10%
```

---

## 六、Old Actor 的定期更新

```python
# 每 update_old_actor_freq 步，用当前 Actor 替换 Old Actor
if (step + 1) % args.update_old_actor_freq == 0:
    old_actor_model.load_state_dict(actor_model.state_dict())
```

> Old Actor 是 Actor 的"历史版本"，PPO 需要它来计算重要性采样比率。定期同步确保比率不会太偏。

---

## 七、两个优化器 + 两个调度器

```python
actor_optimizer = optim.AdamW(actor_model.parameters(), lr=8e-8)
critic_optimizer = optim.AdamW(critic_model.parameters(), lr=8e-8)
actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_steps)
critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_steps)
```

> Actor 和 Critic 各有独立的优化器和学习率调度器，训练时同步更新。

---

## 八、关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 8e-8 | Actor 学习率 |
| `critic_learning_rate` | 8e-8 | Critic 学习率 |
| `clip_epsilon` | 0.1 | PPO 裁剪参数 ε |
| `vf_coef` | 0.5 | Value Loss 系数 |
| `kl_coef` | 0.02 | KL 惩罚系数 |
| `update_old_actor_freq` | 4 | Old Actor 更新频率 |
| `max_gen_len` | 1536 | 生成最大长度 |
| `batch_size` | 2 | 极小（生成 + 5 个模型显存压力大） |
| `reasoning` | 1 | 是否为推理模型模式 |
| `reward_model_path` | internlm2-1_8b-reward | 外部奖励模型路径 |
