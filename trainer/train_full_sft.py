import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个 epoch（或从 epoch 中间断点续训的一段）。

    参数说明：
    - epoch: 当前 epoch 下标（从 0 开始）
    - loader: DataLoader（注意可能用了 SkipBatchSampler，用于续训时跳过已训练过的 batch）
    - iters: “本 epoch 视角下”的总 step 数（用于学习率调度和 ETA 估算；续训时通常=len(loader)+skip）
    - start_step: 断点续训时，本 epoch 已完成的 step 数（用于让日志/保存显示连续的 step）
    - wandb: 可选的 swanlab/wandb 记录器

    重要说明：
    - 本函数会使用外部的全局变量：`args`, `model`, `optimizer`, `scaler`, `autocast_ctx`, `lm_config`
      这是脚本式训练代码的常见写法：主函数负责组装组件，train_epoch 只负责执行训练逻辑。
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 1) 把 batch 搬到训练设备上（GPU/CPU）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 2) 计算并设置当前 step 的学习率（余弦调度；见 trainer_utils.get_lr）
        #    这里用 epoch*iters+step 作为“全局步数”，保证跨 epoch 的 lr 连续变化。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 3) 前向 + loss 计算（混合精度 autocast）
        #    - res.loss: 语言模型主损失（通常是 next-token 的交叉熵）
        #    - res.aux_loss: 额外辅助损失（比如 MoE 的 load balancing；如果不用 MoE 可能为 0/None）
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 4) 梯度累积：把一个“大 batch”拆成 accumulation_steps 个小 batch 来反传。
            #    关键点：每个小步的 loss 要除以 accumulation_steps，避免累积后梯度被放大。
            loss = loss / args.accumulation_steps

        # 5) 反向传播（配合 GradScaler 做 float16 的动态 loss scaling；bfloat16 时 scaler 基本是 no-op）
        scaler.scale(loss).backward()

        # 6) 满足梯度累积步数后，才执行一次参数更新
        #    注意：这里用的是 (step + 1) % accumulation_steps。
        #    由于本脚本的 step 从 1 开始计数，这个写法会让“更新点”相对常见写法（step % k == 0）偏移 1。
        #    如果你希望在 step=accumulation_steps、2*accumulation_steps ... 更新，可以把条件改为 (step % accumulation_steps == 0)。
        if (step + 1) % args.accumulation_steps == 0:
            # 6.1) 先把梯度从 scaled 状态还原回来，再做裁剪（否则裁剪阈值不正确）
            scaler.unscale_(optimizer)
            # 6.2) 梯度裁剪：防止梯度爆炸，提升训练稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 6.3) optimizer.step + scaler.update：真正更新参数，并更新动态缩放因子
            scaler.step(optimizer)
            scaler.update()

            # 6.4) 清梯度（set_to_none=True 更省显存/更快）
            optimizer.zero_grad(set_to_none=True)

        # 7) 日志：定期打印 loss/lr/ETA；只在主进程打印（DDP 下避免刷屏）
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 训练时我们用的是“除以 accumulation_steps 的小步 loss”，这里乘回来方便理解
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # ETA（分钟）：用当前 epoch 的平均 step 耗时估算剩余时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 8) 保存权重/断点：只在主进程做（避免多卡同时写文件）
        #    - `args.save_dir/...pth` 是“纯模型权重”
        #    - `lm_checkpoint(..._resume.pth)` 是“断点续训包”，包含模型+优化器+scaler+epoch/step+wandb_id 等
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # DDP / torch.compile 会包一层，保存时要拿到“原始模型”
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 只保存 half 精度到 CPU：省空间、保存更快（推理/再加载时也足够用）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存“可续训”的 checkpoint（额外包含 optimizer/scaler/epoch/step/wandb_id 等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        # 9) 及时释放一些中间变量，减少显存峰值（尤其是 res / loss 相关）
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 作用：
    # - 支持 DDP 多卡训练：读取环境变量 RANK/LOCAL_RANK 并初始化 nccl 进程组
    # - 固定随机种子：保证每次运行（以及每张卡）采样/打乱相对可复现
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 作用：
    # - 创建输出目录
    # - 构建模型配置（网络规模、层数、是否 MoE）
    # - 如果开启 from_resume：尝试读取 `../checkpoints/..._resume.pth`，拿到模型/优化器/scaler/epoch/step 等
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 作用：
    # - 在 CUDA 上使用 autocast 混合精度以提升吞吐、降低显存
    # - bfloat16：稳定性好（一般不需要 GradScaler）
    # - float16：更省显存/更快，但通常需要 GradScaler 防止下溢
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # 作用：
    # - 仅主进程初始化实验跟踪（避免 DDP 多进程重复写日志）
    # - 如果是断点续训：从 checkpoint 里取 wandb_id，resume='must' 继续同一个 run
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 作用：
    # - init_model：创建 MiniMind 模型 + tokenizer，并可从已有权重（如 pretrain）加载（strict=False 允许结构不完全匹配）
    # - SFTDataset：把 conversations 按 chat_template 拼成 prompt，并只对 assistant 回复部分生成 labels（其它位置为 -100 不计入 loss）
    # - DistributedSampler：多卡下把数据集切分到不同进程，避免重复训练同一条数据
    # - AdamW：常用的 Transformer 优化器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    # 作用：
    # - 断点续训时恢复：模型参数、优化器动量/状态、GradScaler 状态
    # - start_epoch/start_step：从保存的位置继续跑，避免重复训练
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    # 作用：
    # - 多卡并行训练：每个进程持有一份模型，梯度在反传时自动 all-reduce 同步
    # - `_ddp_params_and_buffers_to_ignore`：告诉 DDP 忽略某些 buffer（这里是 RoPE 相关缓存），避免不必要的同步/报错
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    # 作用：
    # - 每个 epoch 设定 sampler 的 epoch：保证不同 epoch 的 shuffle 不同，但各卡保持一致
    # - 续训时“跳过已完成 step”：用 SkipBatchSampler 跳过前 start_step 个 batch
    #   这样在同一个 epoch 内也能从中间继续，而不是只能从下一个 epoch 开始。
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # 作用：训练结束后释放 DDP 进程组资源（否则某些环境下会挂住或影响后续任务）
    if dist.is_initialized(): dist.destroy_process_group()