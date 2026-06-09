#!/bin/bash

echo "========================================================================"
echo "Progressive RTM Fine-tuning Training - 强制Resume版本"
echo "========================================================================"
echo ""
echo "修正内容:"
echo "  ✅ 使用--resume参数强制从checkpoint恢复"
echo "  ✅ backbone学习率: 6e-7"
echo "  ✅ preprocessor_sec学习率: 6e-7"
echo "  ✅ decode_head学习率: 6e-6"
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
echo "启动Fine-tuning训练 (使用--resume)..."
echo "========================================================================"
echo ""
echo "参数:"
echo "  - GPU: 2张"
echo "  - 端口: 29510"
echo "  - Batch size: 2"
echo "  - Resume: 使用--resume强制恢复"
echo "  - 日志: train_finetune_80k_to_120k_resume.log"
echo ""

# 启动训练,使用--resume参数
nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29510 \
  tools/train.py \
  configs/ascformer/ascformer_rtm_progressive_finetune.py \
  --launcher pytorch \
  --resume \
  --cfg-options train_dataloader.batch_size=2 \
  > train_finetune_80k_to_120k_resume.log 2>&1 &

TRAIN_PID=$!

echo ""
echo "========================================================================"
echo "✅ 训练已启动!"
echo "========================================================================"
echo ""
echo "进程ID: $TRAIN_PID"
echo "日志文件: train_finetune_80k_to_120k_resume.log"
echo ""
echo "监控命令:"
echo "  tail -f train_finetune_80k_to_120k_resume.log"
echo ""
echo "验证Resume (等待30秒后执行):"
echo "  sleep 30"
echo "  grep 'Iter(train)' train_finetune_80k_to_120k_resume.log | head -1"
echo ""
echo "========================================================================"
