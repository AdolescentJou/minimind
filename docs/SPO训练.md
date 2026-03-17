# SPO 训练流程解析（train_spo.py）

> **Self-Play Optimization**：每个 prompt 只生成 **1 个**回复（而非 GRPO 的 8 个），用自适应 Baseline 替代组内统计，更节省计算资源。

---

## 一、SPO vs GRPO

GRPO 需要为每个 prompt 生成 8 个回复来做组内比较，SPO 的改进是：用一个**自适应价值追踪器（Value Tracker）**维护全局 Baseline，每个 prompt 只需生成 1 个回复。

```
GRPO:  一个 prompt → 生成 8 个回复 → 组内排名 → 优势
SPO:   一个 prompt → 生成 1 个回复 → 和全局 Baseline 比 → 优势
```

---

## 二、与 GRPO 的差异

| 维度 | GRPO | SPO |
|------|------|-----|
| **每 prompt 生成数** | 8 个 | **1 个** |
| **优势估计** | 组内均值/标准差 | **AutoAdaptiveValueTracker** |
| **额外组件** | 无 | **ValueTracker**（贝叶斯追踪器） |
| **计算量** | 高（8× 生成 + 奖励） | **低**（1× 生成 + 奖励） |
| **模型数量** | 3 个（同 GRPO） | 3 个（Policy + Ref + Reward） |
| **其余流程** | — | 奖励计算/KL 惩罚/保存等**完全一致**，略过 |

---

## 三、核心差异：AutoAdaptiveValueTracker

```python
class AutoAdaptiveValueTracker:
    """用贝叶斯在线学习维护全局奖励基线"""
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06,
                 clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode      # 衰减模式：'kl' 或 'constant'
        self.D_half = D_half           # KL 散度半衰期
        N_init = 1.0 / (1.0 - clip_lower)
        self.alpha = 0.5 * N_init     # Beta 分布参数 α
        self.beta = 0.5 * N_init      # Beta 分布参数 β
        self.old_mean_logprob = None   # 上一步的平均 log 概率

    def get_baselines(self, batch_size):
        """输出当前的 Baseline（Beta 分布的均值）"""
        baseline = self.alpha / (self.alpha + self.beta)
        return torch.full((batch_size,), baseline)

    def compute_rho(self, cur_mean_logprob):
        """根据策略变化程度自适应调整衰减因子"""
        if self.old_mean_logprob is None:
            return self.rho_const
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        rho = 2 ** (-kl / self.D_half)  # KL 越大，rho 越小，历史信息衰减越快
        return clamp(rho, clip_lower, clip_upper)

    def update(self, rewards, cur_logprobs, response_masks):
        """用新奖励更新 Beta 分布参数"""
        rho = self.compute_rho(mean_logprob)
        normalized_rewards = (rewards + 3) / 6   # 归一化到 [0, 1]
        avg = normalized_rewards.mean()
        self.alpha = rho * self.alpha + avg       # 指数衰减 + 新观测
        self.beta = rho * self.beta + (1 - avg)
```

**核心思想：**

```
维护一个 Beta(α, β) 分布来建模"全局奖励水平"：
  - Baseline = α / (α + β)  ≈ 历史平均奖励
  - 新奖励 > Baseline → 优势为正 → 强化该行为
  - 新奖励 < Baseline → 优势为负 → 抑制该行为

自适应衰减 ρ：
  - 策略变化大（KL 大）→ ρ 小 → 历史信息快速遗忘（过去的 Baseline 不再适用）
  - 策略变化小（KL 小）→ ρ 大 → 历史信息缓慢衰减（Baseline 稳定可靠）
```

### 优势计算

```python
# Baseline 在 [0, 1] 空间 → 反归一化到 [-3, 3]
baselines = value_tracker.get_baselines(B)
unnormalized_baselines = baselines * 6 - 3

# 优势 = 实际奖励 - Baseline
advantages = rewards - unnormalized_baselines
advantages = advantages.clamp(-5.0, 5.0)
```

### 损失计算

```python
# 和 GRPO 几乎一样，但不用 importance sampling 技巧
per_token_loss = -per_token_logps * advantages.unsqueeze(1) + β * per_token_kl
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

---

## 四、关键超参数

| 参数 | 默认值 | 与 GRPO 对比 |
|------|--------|-------------|
| `batch_size` | 2 | 相同 |
| `learning_rate` | **1e-7** | GRPO 8e-8 |
| `accumulation_steps` | **4** | GRPO 1（SPO 用累积补偿 batch 小） |
| `beta` | 0.02 | 相同 |
| `max_gen_len` | 1536 | 相同 |
| `rho_mode` | `kl` | SPO 独有 |
| `D_half` | 0.06 | SPO 独有：KL 半衰期 |

---

## 五、GRPO vs SPO 效率对比

```
GRPO（batch_size=2, num_generations=8）:
  每 step 生成: 2 × 8 = 16 条回复
  Reward 模型打分: 16 次
  Policy 前向: 16 条

SPO（batch_size=2, num_generations=1）:
  每 step 生成: 2 × 1 = 2 条回复
  Reward 模型打分: 2 次
  Policy 前向: 2 条

SPO 的计算量约为 GRPO 的 1/8
```
