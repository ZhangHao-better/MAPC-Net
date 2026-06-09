#!/bin/bash

echo "========================================================================"
echo "Progressive RTM Fine-tuning - Load Weights Only (不Resume)"
echo "========================================================================"
echo ""
echo "策略:"
echo "  ✅ 从iter_80000.pth加载权重"
echo "  ✅ 重新开始40k轮fine-tuning (iter 0-40000)"
echo "  ✅ 学习率调度器从头开始"
echo "  ✅ backbone: 6e-7, decode_head: 6e-6"
echo ""

# 设置环境
export PATH="/home/zhlinux/anaconda3/envs/rtm_linux/bin:$PATH"
cd /home/zhlinux/RTM-progressive/ASCFormer

echo "========================================================================"
echo "停止之前的训练进程..."
pkill -f "train.py.*ascformer_rtm_progressive_finetune"
sleep 2

# 验证checkpoint
CKPT="work_dirs/ascformer_rtm_progressive/iter_80000.pth"
if [ -f "$CKPT" ]; then
    echo "✅ Checkpoint找到: $CKPT"
else
    echo "❌ Checkpoint未找到: $CKPT"
    exit 1
fi

echo ""
echo "========================================================================"
echo "启动Fine-tuning训练 (Load Weights Only)..."
echo "========================================================================"
echo ""
echo "参数:"
echo "  - GPU: 2张"
echo "  - 端口: 29511"
echo "  - Batch size: 2"
echo "  - Resume: False (只加载权重)"
echo "  - 日志: train_finetune_loadonly.log"
echo "  - 工作目录: work_dirs/ascformer_rtm_finetune_40k"
echo ""

# 启动训练 - 不使用--resume参数
nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29511 \
  tools/train.py \
  configs/ascformer/ascformer_rtm_progressive_finetune.py \
  --launcher pytorch \
  --cfg-options train_dataloader.batch_size=2 \
  > train_finetune_loadonly.log 2>&1 &

TRAIN_PID=$!

echo ""
echo "========================================================================"
echo "✅ 训练已启动!"
echo "========================================================================"
echo ""
echo "进程ID: $TRAIN_PID"
echo "日志文件: train_finetune_loadonly.log"
echo ""
echo "监控命令:"
echo "  tail -f train_finetune_loadonly.log"
echo ""
echo "验证学习率 (等待30秒后执行):"
echo "  sleep 30"
echo "  grep 'Iter(train)' train_finetune_loadonly.log | head -5"
echo "  # 应该看到 lr: 1.xxxxxe-06 (warmup阶段)"
echo ""
echo "========================================================================"
