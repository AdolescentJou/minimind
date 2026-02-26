import os
import sys

__package__ = "trainer"
# 将项目根目录加入 Python 搜索路径，确保可以正确导入其他模块（如 model、dataset）
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
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个 epoch 的核心函数。

    参数说明：
        epoch      : 当前是第几个 epoch（从0开始计数）
        loader     : DataLoader，负责按批次提供训练数据
        iters      : 本 epoch 的总迭代步数（包含可能跳过的步数）
        start_step : 从第几步开始训练（用于断点续训时跳过已完成的步骤）
        wandb      : 可选的实验追踪工具（用于记录训练指标曲线）
    """
    start_time = time.time()

    # -------- 遍历 DataLoader，逐批取出训练数据 --------
    # enumerate 的 start 参数让 step 从 start_step+1 开始编号，方便续训时步数连续
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):

        # 【数据搬运】将输入token和标签搬到GPU上（或CPU），模型在哪训练数据就放在哪
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 【学习率调度】根据当前全局步数动态计算学习率（通常用 cosine schedule）
        # 训练初期学习率逐渐升高（warmup），之后按余弦曲线逐渐衰减
        # 这样做的好处：初期小学习率避免梯度爆炸，中期大学习率加快学习，后期小学习率精细调整
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 【前向传播 + 损失计算】在混合精度上下文中进行，节省显存并加速计算
        with autocast_ctx:
            # 将 input_ids 送入模型，同时传入 labels 让模型内部计算交叉熵损失
            # 语言模型的训练目标：根据前面的 token 预测下一个 token
            res = model(input_ids, labels=labels)

            # res.loss     : 主损失（交叉熵损失），衡量模型预测下一个 token 的准确度
            # res.aux_loss : 辅助损失（仅 MoE 模型有），用于平衡各专家的负载
            #                如果不是 MoE 模型，aux_loss 通常为 0
            loss = res.loss + res.aux_loss

            # 【梯度累积】将 loss 除以累积步数
            # 梯度累积的作用：当 GPU 显存不够大时，无法用大 batch_size 训练
            # 通过多次小 batch 前向+反向传播并累积梯度，等效于一次大 batch 训练
            # 例如 batch_size=32, accumulation_steps=8, 等效 batch_size=256
            loss = loss / args.accumulation_steps

        # 【反向传播】计算每个参数的梯度
        # scaler.scale(loss) 是混合精度训练的一部分：
        #   先将 loss 放大（scale up），避免 float16 精度下梯度过小变成 0（梯度下溢）
        #   反向传播后再缩小回来
        scaler.scale(loss).backward()

        # 【参数更新】每累积 accumulation_steps 步梯度后，才真正更新一次模型参数
        if (step + 1) % args.accumulation_steps == 0:
            # 将之前 scale up 的梯度还原回真实值
            scaler.unscale_(optimizer)

            # 【梯度裁剪】如果梯度的总范数超过 grad_clip（默认1.0），就等比例缩小所有梯度
            # 防止梯度爆炸：某些异常数据可能导致梯度特别大，会让模型参数发散
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 用优化器根据梯度更新模型参数（AdamW 算法）
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子（动态调整 scale 大小，适应训练过程中的数值范围变化）
            scaler.update()

            # 清零梯度，为下一轮梯度累积做准备
            # set_to_none=True 比 zero_grad() 更高效，直接把梯度设为 None 而不是填充 0
            optimizer.zero_grad(set_to_none=True)

        # -------- 日志打印：定期输出训练信息 --------
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 还原真实 loss（之前除以了 accumulation_steps）
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # logits_loss = 主损失（不含辅助损失），是衡量语言建模效果的核心指标
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 预估当前 epoch 还需多少分钟完成
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 如果启用了 wandb，将指标上报用于可视化
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # -------- 模型保存：定期保存模型权重（仅主进程保存，避免多卡重复写文件） --------
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式（关闭 Dropout 等随机行为）

            # 构造保存路径，MoE 模型加后缀 '_moe' 以区分
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 获取原始模型（去掉 DDP 包装和 torch.compile 包装）
            # DDP 会将模型包在 .module 属性下；torch.compile 会包在 _orig_mod 属性下
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()

            # 保存模型权重，转成 float16 以减小文件体积（约为 float32 的一半）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            # 保存完整的训练检查点（包含优化器状态、epoch、step 等），用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')

            model.train()  # 切回训练模式
            del state_dict  # 释放内存

        # 手动删除当前 step 的中间变量，帮助 Python 垃圾回收器及时释放 GPU 显存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    # =====================================================================
    #  命令行参数定义
    #  这些参数控制训练的各个方面，可以在启动训练时通过命令行灵活调整
    # =====================================================================
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境（多GPU训练时需要）
    # 如果只有单卡或CPU训练，这里会自动跳过分布式设置
    # local_rank 表示当前进程使用的是第几块 GPU
    local_rank = init_distributed_mode()
    # 如果是多卡分布式训练，将 device 设为当前进程对应的 GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子，保证实验可复现
    # 不同 rank 加不同偏移量，确保各 GPU 上的数据打乱方式不同（增加数据多样性）
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建模型权重保存目录（如果不存在）
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建模型配置：指定隐藏层维度、层数、是否使用 MoE（混合专家）架构
    # 这些超参数决定了模型的大小和能力
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果开启了断点续训（from_resume=1），尝试加载之前保存的检查点（checkpoint）
    # 检查点包含：模型权重、优化器状态、训练进度等，可以从中断处继续训练
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 混合精度训练（AMP, Automatic Mixed Precision）：
    #   训练时部分计算使用 float16/bfloat16（半精度），部分使用 float32（全精度）
    #   好处：显存占用减半，训练速度提升 1.5-2x，精度损失几乎可忽略
    #   bfloat16 vs float16：bfloat16 数值范围更大，不容易溢出，推荐在支持的GPU上使用
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 如果是 CPU 训练则不使用混合精度（CPU 不支持），使用 nullcontext 作为空上下文
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # wandb（这里实际使用 swanlab）是实验追踪工具，可以实时可视化训练过程中的：
    #   loss 曲线、学习率变化、训练耗时等指标
    # 只在主进程（rank 0）初始化，避免多卡训练时重复上报
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 如果是续训，使用之前的 wandb_id 继续同一个实验；否则创建新实验
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 初始化模型和分词器（tokenizer）
    # 分词器负责将文本转换为 token id 序列（模型只能处理数字，不能直接处理文字）
    # from_weight 不为 'none' 时，会加载预训练权重作为起点（迁移学习/继续训练）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # torch.compile 是 PyTorch 2.0+ 的编译优化：
    # 将模型的计算图编译为更高效的底层代码，可提速 10%-30%
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 创建预训练数据集：读取 jsonl 文件，将文本转为 token 序列并截断到 max_seq_len
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 分布式采样器：多卡训练时确保每块 GPU 拿到不同的数据子集，避免重复训练相同数据
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # GradScaler 配合混合精度训练使用：
    #   对 float16 训练时动态缩放 loss，防止梯度下溢（数值太小变成0）
    #   bfloat16 不需要 scaler（数值范围够大），所以只在 float16 时启用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # AdamW 优化器：Adam 的改进版，加入了权重衰减（weight decay）正则化
    # 权重衰减帮助防止过拟合，是当前训练大模型最主流的优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    # 断点续训：如果有保存的检查点，恢复模型权重、优化器状态、scaler 状态
    # 这样可以从上次中断的地方精确继续训练，不会浪费已完成的计算
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])       # 恢复模型参数
        optimizer.load_state_dict(ckp_data['optimizer']) # 恢复优化器状态（包含动量等）
        scaler.load_state_dict(ckp_data['scaler'])       # 恢复梯度缩放器状态
        start_epoch = ckp_data['epoch']                  # 恢复 epoch 进度
        start_step = ckp_data.get('step', 0)             # 恢复 step 进度
    
    # ========== 7. DDP包模型 ==========
    # DDP（DistributedDataParallel）是 PyTorch 的多卡并行训练方案：
    #   每块 GPU 各持有一份完整的模型副本
    #   前向传播时各自独立计算，反向传播时自动同步（AllReduce）梯度
    #   确保所有 GPU 上的模型参数保持一致
    if dist.is_initialized():
        # freqs_cos 和 freqs_sin 是 RoPE 位置编码的预计算缓冲区
        # 它们是固定值不参与训练，告诉 DDP 忽略它们以避免不必要的同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    # 外层循环：遍历每个 epoch（一个 epoch = 把全部训练数据看一遍）
    for epoch in range(start_epoch, args.epochs):
        # 分布式训练时，设置 epoch 让 sampler 在每个 epoch 用不同的数据打乱顺序
        # 这样每个 epoch 各 GPU 看到的数据顺序都不同，有助于模型学习
        train_sampler and train_sampler.set_epoch(epoch)

        # 用固定种子 + epoch 生成数据打乱顺序，保证可复现
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()

        # 计算需要跳过的步数（仅在续训的第一个 epoch 生效）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        # SkipBatchSampler：自定义批次采样器，支持跳过前 skip 个 batch
        # 续训时直接跳到上次训练到的位置，不需要重新遍历已经训练过的数据
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # DataLoader 负责：
        #   1. 按 batch_sampler 指定的顺序和大小取数据
        #   2. num_workers 个子进程并行加载数据，主进程专注 GPU 计算（数据加载不阻塞训练）
        #   3. pin_memory=True 将数据放在锁页内存中，加速 CPU→GPU 的数据传输
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 续训时传入总步数 = 实际加载的数据量 + 跳过的步数，保证进度条和日志正确
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布式进程 ==========
    # 训练完成后，销毁分布式进程组，释放通信资源（NCCL 后端占用的 GPU 通信通道等）
    # 如果不清理，进程可能不会正常退出
    if dist.is_initialized(): dist.destroy_process_group()
