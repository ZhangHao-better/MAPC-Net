#!/bin/bash

echo "========================================================================"
echo "Progressive RTM Fine-tuning - 保守学习率 1e-6"
echo "========================================================================"
echo ""
echo "策略:"
echo "  ✅ 基础学习率: 1e-6 (比之前的6e-6更保守)"
echo "  ✅ backbone: 1e-7, decode_head: 1e-6"
echo "  ✅ 从iter_80000.pth加载权重"
echo "  ✅ 训练40k轮"
echo ""

# 设置环境
export PATH="/home/zhlinux/anaconda3/envs/rtm_linux/bin:$PATH"
cd /home/zhlinux/RTM-progressive/ASCFormer

echo "========================================================================"
echo "停止之前的训练进程..."
pkill -f "train.py"
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
echo "启动Fine-tuning训练 (学习率 1e-6)..."
echo "========================================================================"
echo ""
echo "参数:"
echo "  - GPU: 2张"
echo "  - 端口: 29512"
echo "  - Batch size: 2"
echo "  - 基础学习率: 1e-6"
echo "  - 日志: train_finetune_1e6.log"
echo "  - 工作目录: work_dirs/ascformer_rtm_finetune_1e6"
echo ""

# 启动训练
nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29512 \
  tools/train.py \
  configs/ascformer/ascformer_rtm_finetune_1e6.py \
  --launcher pytorch \
  --cfg-options train_dataloader.batch_size=2 \
  > train_finetune_1e6.log 2>&1 &

TRAIN_PID=$!

echo ""
echo "========================================================================"
echo "✅ 训练已启动!"
echo "========================================================================"
echo ""
echo "进程ID: $TRAIN_PID"
echo "日志文件: train_finetune_1e6.log"
echo ""
echo "监控命令:"
echo "  tail -f train_finetune_1e6.log | grep 'Iter(train)'"
echo ""
echo "验证学习率 (等待30秒后执行):"
echo "  grep 'Iter(train)' train_finetune_1e6.log | head -5"
echo "  # 应该看到 lr: 约1e-07到1e-06之间"
echo ""
echo "对比:"
echo "  - 之前6e-6: 可能步子太大,破坏了已收敛权重"
echo "  - 现在1e-6: 更保守,精细打磨"
echo ""
echo "========================================================================"
