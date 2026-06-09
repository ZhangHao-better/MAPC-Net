#!/bin/bash

echo "========================================================================"
echo "RTM Contrastive Learning Fine-tuning"
echo "========================================================================"
echo ""
echo "配置:"
echo "  ✅ 对比学习: 已启用"
echo "  ✅ 多尺度特征融合: z1+z2+z3+z4"
echo "  ✅ 对比学习权重: 0.1"
echo "  ✅ 采样策略: RTM原始方式"
echo "  ✅ 基础学习率: 1e-6"
echo "  ✅ 从80k模型继续训练"
echo ""

# 设置环境
export PATH="/home/zhlinux/anaconda3/envs/rtm_linux/bin:$PATH"
cd /home/zhlinux/RTM-progressive/ASCFormer

echo "========================================================================"
echo "停止之前的训练..."
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
echo "启动对比学习Fine-tuning训练..."
echo "========================================================================"
echo ""
echo "参数:"
echo "  - GPU: 2张"
echo "  - 端口: 29513"
echo "  - Batch size: 2"
echo "  - 对比学习: 启用"
echo "  - 日志: train_contrastive.log"
echo "  - 工作目录: work_dirs/ascformer_rtm_contrastive"
echo ""

# 启动训练
nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29513 \
  tools/train.py \
  configs/ascformer/ascformer_rtm_contrastive.py \
  --launcher pytorch \
  --cfg-options train_dataloader.batch_size=2 \
  > train_contrastive.log 2>&1 &

TRAIN_PID=$!

echo ""
echo "========================================================================"
echo "✅ 训练已启动!"
echo "========================================================================"
echo ""
echo "进程ID: $TRAIN_PID"
echo "日志文件: train_contrastive.log"
echo ""
echo "监控命令:"
echo "  tail -f train_contrastive.log | grep 'Iter(train)'"
echo ""
echo "验证对比学习 (等待30秒后执行):"
echo "  grep 'loss_contrastive' train_contrastive.log | head -5"
echo "  # 应该看到对比学习损失值"
echo ""
echo "预期效果:"
echo "  - Tampered IoU: 15.6% → 18-20%"
echo "  - mIoU: 57.16% → 58-59%"
echo ""
echo "========================================================================"
