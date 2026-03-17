"""
一个“从 0 到 1 可跑通”的最小预训练训练脚本（只保留核心链路）：

只做 4 件事：
1) 模型初始化（MiniMindConfig + MiniMindForCausalLM）
2) 数据加载（读取 jsonl 里的 {"text": "..."}，做 tokenization + padding + labels）
3) 单步前向（forward）+ 损失计算（cross-entropy next-token）
4) 单步反向传播（backward）+ 参数更新（optimizer.step）

刻意删除/不包含：
- DDP/多卡
- AMP 混合精度
- 梯度累积/梯度裁剪
- 学习率调度
- 保存/续训 checkpoint
- wandb/swanlab
- torch.compile

适合：完全初学者先把“训练最小单元”跑通并看懂每一步张量在做什么。
"""

import os
import sys

# 关键：把“仓库根目录”加入模块搜索路径，避免你在 trainer/ 目录里运行时报：
#   ModuleNotFoundError: No module named 'model'
# 因为 model/ 位于仓库根目录下，而不是 trainer/ 目录下。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch import optim
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "没有找到 torch。请先在你的环境里安装依赖：\n"
        "  pip install -r requirements.txt\n"
        "然后重新运行本脚本。\n"
    ) from e

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "没有找到 transformers。请先在你的环境里安装依赖：\n"
        "  pip install -r requirements.txt\n"
        "然后重新运行本脚本。\n"
    ) from e

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# ========== 1) 一个最小数据集：读取 jsonl，并产出 (input_ids, labels) ==========
class TinyJsonlPretrainDataset(Dataset):
    """
    读取 jsonl 文件，每行形如：
      {"text": "......"}

    产出的训练样本是“自回归语言模型（Causal LM）”常用格式：
    - input_ids: [max_seq_len]  (token ids，padding 到固定长度)
    - labels:    [max_seq_len]  (和 input_ids 一样，但 padding 位置改成 -100，用于忽略 loss)

    为什么 labels 要把 padding 置为 -100？
    - 模型计算交叉熵时设置 ignore_index=-100（在模型里实现了）
    - 这样 padding token 就不会贡献 loss，也就不会误导模型
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 读入所有文本（初学者版本：直接一次性读到内存，方便理解）
        self.texts: List[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # 统一转成 str，避免不是字符串时出错
                self.texts.append(str(obj["text"]))

        if len(self.texts) == 0:
            raise ValueError(f"数据文件是空的：{data_path}")

        # 常用特殊 token（BOS/EOS/PAD）
        # 注意：你的 tokenizer 配置里通常会有这些 id
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        if self.pad is None:
            # 极少数 tokenizer 没有 pad，我们给一个兜底（这里按 0 处理）
            self.pad = 0
        if self.bos is None or self.eos is None:
            raise ValueError(
                "tokenizer 缺少 bos_token_id / eos_token_id。"
                "请检查 `model/` 下的 tokenizer 配置，或换一个带 BOS/EOS 的 tokenizer。"
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]

        # (1) 把文本变成 token ids；这里不自动加特殊 token，因为我们要手动加 BOS/EOS
        # truncation=True：超长就截断
        # max_length = max_seq_len - 2：预留 BOS/EOS 两个位置
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len - 2,
        ).input_ids

        # (2) 手动拼上 BOS / EOS
        token_ids = [self.bos] + token_ids + [self.eos]

        # (3) padding 到固定长度
        #     - 训练时 batch 内每条样本长度一致，才能堆成 [B, T] 的张量
        if len(token_ids) < self.max_seq_len:
            token_ids = token_ids + [self.pad] * (self.max_seq_len - len(token_ids))
        else:
            token_ids = token_ids[: self.max_seq_len]

        input_ids = torch.tensor(token_ids, dtype=torch.long)  # [T]

        # (4) labels 默认与 input_ids 相同；padding 位置设 -100（忽略 loss）
        labels = input_ids.clone()
        labels[input_ids == self.pad] = -100

        return input_ids, labels


@dataclass
class TrainBatch:
    input_ids: torch.Tensor  # [B, T]
    labels: torch.Tensor     # [B, T]


def main():
    # 让脚本“在哪运行都行”：所有默认路径都基于仓库根目录计算，而不是基于当前工作目录。
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_path = os.path.join(repo_root, "dataset", "tiny_pretrain.jsonl")
    default_tokenizer_dir = os.path.join(repo_root, "model")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="jsonl 路径，每行格式：{'text': '...'}",
    )
    parser.add_argument("--max_seq_len", type=int, default=64, help="序列最大长度 T（越小越省显存）")
    parser.add_argument("--batch_size", type=int, default=2, help="batch 大小 B（越大越吃显存）")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备：cuda 或 cpu",
    )
    args = parser.parse_args()

    # ========== 2) 初始化 tokenizer ==========
    # 这里复用仓库自带 tokenizer（model/ 目录）
    tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_dir)

    # ========== 3) 初始化模型 ==========
    # 初学者建议：把模型变小，跑起来更快，也更容易 debug
    # 你只需要记住：hidden_size / num_hidden_layers 越小，模型越小
    config = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        # vocab_size 要与 tokenizer 对齐，否则 logits 的最后一维 V 会不匹配
        vocab_size=len(tokenizer),
        # 关闭 MoE（让链路更直）
        use_moe=False,
    )
    model = MiniMindForCausalLM(config).to(args.device)
    model.train()  # 进入训练模式（会启用 dropout 等训练行为）

    # ========== 4) 数据集与 DataLoader ==========
    ds = TinyJsonlPretrainDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # 取出一个 batch（我们只跑“单步”）
    input_ids, labels = next(iter(loader))
    batch = TrainBatch(
        input_ids=input_ids.to(args.device),
        labels=labels.to(args.device),
    )

    # ========== 5) 优化器 ==========
    # 这里用 AdamW：Transformer 训练里最常见的优化器之一
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # ========== 6) 单步训练：forward → loss → backward → step ==========
    # (1) 前向：会返回一个对象（HF 的 CausalLMOutputWithPast）
    #     其中 output.loss 是 cross-entropy（next-token prediction）
    output = model(batch.input_ids, labels=batch.labels)
    loss = output.loss  # 标量

    # (2) 反向传播：计算每个参数的梯度（grad）
    loss.backward()

    # (3) 参数更新：用梯度更新参数
    optimizer.step()

    # (4) 清空梯度：下一步训练前必须清空（否则梯度会累积）
    optimizer.zero_grad()

    # ========== 7) 打印关键信息（帮助你“看见”训练） ==========
    print("✅ 单步训练完成")
    print(f"- device: {args.device}")
    print(f"- input_ids shape: {tuple(batch.input_ids.shape)}  (B, T)")
    print(f"- vocab_size: {config.vocab_size}")
    print(f"- loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()


