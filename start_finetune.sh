#!/bin/bash
# Fine-tuning训练启动脚本
# 从iter_80000.pth继续训练到iter_120000.pth

set -e

echo "========================================================================"
echo "Progressive RTM Fine-tuning Training"
echo "========================================================================"
echo ""
echo "配置信息:"
echo "  - 起始checkpoint: work_dirs/ascformer_rtm_progressive/iter_80000.pth"
echo "  - 总迭代数: 120,000 (80k已完成 + 40k新增)"
echo "  - 验证间隔: 4,000 iterations"
echo "  - Base LR: 6e-6 (原始1/10)"
echo "  - RTM_backbone LR: 6e-7 (base_lr * 0.1)"
echo "  - decode_head LR: 6e-6 (base_lr * 1.0)"
echo ""
echo "========================================================================"
echo "检查环境..."
echo "========================================================================"

# 检查checkpoint是否存在
CHECKPOINT="work_dirs/ascformer_rtm_progressive/iter_80000.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: checkpoint文件不存在: $CHECKPOINT"
    echo "请确保80k训练已完成并保存了checkpoint"
    exit 1
fi
echo "✅ Checkpoint找到: $CHECKPOINT"

# 检查配置文件
CONFIG="configs/ascformer/ascformer_rtm_progressive_finetune.py"
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG"
    exit 1
fi
echo "✅ 配置文件找到: $CONFIG"

# 设置Python环境
export PATH="/home/zhlinux/anaconda3/envs/rtm_linux/bin:$PATH"
echo "✅ Python环境设置完成"

echo ""
echo "========================================================================"
echo "启动Fine-tuning训练..."
echo "========================================================================"
echo ""
echo "训练参数:"
echo "  - GPU数量: 2"
echo "  - 端口: 29509"
echo "  - Batch size: 2"
echo "  - 日志文件: train_finetune_80k_to_120k.log"
echo ""
echo "预期结果:"
echo "  - Checkpoint: iter_84000.pth, iter_88000.pth, ..., iter_120000.pth"
echo "  - 验证次数: 10次"
echo "  - 预计时间: ~12小时"
echo ""
echo "========================================================================"
echo "按 Ctrl+C 取消,或等待5秒后自动启动..."
echo "========================================================================"

sleep 5

nohup python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=29509 \
  tools/train.py $CONFIG \
  --launcher pytorch \
  --resume $CHECKPOINT \
  --cfg-options train_dataloader.batch_size=2 \
  > train_finetune_80k_to_120k.log 2>&1 &

TRAIN_PID=$!

echo ""
echo "========================================================================"
echo "✅ 训练已启动!"
echo "========================================================================"
echo ""
echo "进程ID: $TRAIN_PID"
echo "日志文件: train_finetune_80k_to_120k.log"
echo ""
echo "监控命令:"
echo "  查看实时日志: tail -f train_finetune_80k_to_120k.log"
echo "  查看最新50行: tail -50 train_finetune_80k_to_120k.log"
echo "  查看训练进度: grep 'Iter(train)' train_finetune_80k_to_120k.log | tail -5"
echo "  查看验证结果: grep 'mIoU' train_finetune_80k_to_120k.log"
echo ""
echo "停止训练: kill $TRAIN_PID"
echo ""
echo "========================================================================"

