#!/bin/bash
# Fine-tuning训练启动脚本 (修正版)

echo "========================================================================"
echo "Progressive RTM Fine-tuning Training (修正版)"
echo "========================================================================"
echo ""
echo "修正内容:"
echo "  ✅ backbone学习率: 6e-7 (原RTM_backbone配置未生效)"
echo "  ✅ preprocessor_sec学习率: 6e-7"
echo "  ✅ decode_head学习率: 6e-6"
echo ""
echo "========================================================================"

# 停止之前的训练
echo "停止之前的训练进程..."
pkill -f "train.py.*ascformer_rtm_progressive_finetune"
sleep 3

# 设置环境
export PATH="/home/zhlinux/anaconda3/envs/rtm_linux/bin:$PATH"

# 检查checkpoint
CHECKPOINT="work_dirs/ascformer_rtm_progressive/iter_80000.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: checkpoint不存在: $CHECKPOINT"
    exit 1
fi
echo "✅ Checkpoint找到: $CHECKPOINT"

# 启动训练
echo ""
echo "========================================================================"
echo "启动Fine-tuning训练..."
echo "========================================================================"
echo ""
echo "参数:"
echo "  - GPU: 2张"
echo "  - 端口: 29509"
echo "  - Batch size: 2"
echo "  - 日志: train_finetune_80k_to_120k_final.log"
echo ""

nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29509 \
  tools/train.py configs/ascformer/ascformer_rtm_progressive_finetune.py \
  --launcher pytorch \
  --cfg-options train_dataloader.batch_size=2 \
  > train_finetune_80k_to_120k_final.log 2>&1 &

TRAIN_PID=$!

echo ""
echo "========================================================================"
echo "✅ 训练已启动!"
echo "========================================================================"
echo ""
echo "进程ID: $TRAIN_PID"
echo "日志文件: train_finetune_80k_to_120k_final.log"
echo ""
echo "监控命令:"
echo "  tail -f train_finetune_80k_to_120k_final.log"
echo ""
echo "验证学习率 (等待30秒后执行):"
echo "  sleep 30"
echo "  grep 'decode_head.*weight:lr=' train_finetune_80k_to_120k_final.log | head -3"
echo "  grep 'backbone.*weight:lr=' train_finetune_80k_to_120k_final.log | head -3"
echo "  grep 'preprocessor_sec.*weight:lr=' train_finetune_80k_to_120k_final.log | head -3"
echo ""
echo "========================================================================"

