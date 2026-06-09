# Fine-tuning学习率问题解决方案

## 问题发现

用户发现训练日志中显示 `lr: 0.0000e+00`,学习率全为0,模型无法学习!

## 根本原因

**MMEngine的resume机制会同时恢复:**
1. 模型权重 ✅
2. 优化器状态 ✅
3. **学习率调度器状态** ❌ ← 这是问题所在!

当`resume=True`时,从iter_80000.pth恢复的调度器状态中,学习率已经从6e-5衰减到接近0了。
即使我们修改了配置文件中的学习率调度器参数,resume会用checkpoint中保存的状态覆盖它!

## 错误尝试

### 尝试1: 修改调度器的begin/end参数
```python
param_scheduler = [
    dict(type='LinearLR', begin=80000, end=81500),  # ❌ 无效
    dict(type='PolyLR', begin=81500, end=120000),   # ❌ resume会覆盖
]
```
**结果**: 失败,因为resume恢复了旧的调度器状态

### 尝试2: 使用--resume命令行参数
```bash
python tools/train.py config.py --resume  # ❌ 问题依旧
```
**结果**: 失败,resume仍然会恢复调度器状态

## 正确解决方案

**使用`load_from`而不是`resume`:**

```python
# ============================================================================
# 关键配置
# ============================================================================
resume = False  # ← 不resume,避免恢复旧的学习率状态
load_from = 'work_dirs/ascformer_rtm_progressive/iter_80000.pth'

# 重新训练40k轮
train_cfg = dict(
    max_iters=40000,  # ← 从0开始的40k轮
    val_interval=4000
)

# 学习率调度器从头开始
param_scheduler = [
    dict(type='LinearLR', begin=0, end=1500),      # ← 从0开始
    dict(type='PolyLR', begin=1500, end=40000),    # ← 到40000结束
]
```

## 效果对比

### 使用resume=True (错误):
```
Iter(train) [ 80050/120000]  lr: 0.0000e+00  ← 学习率为0!
```

### 使用resume=False + load_from (正确):
```
Iter(train) [   50/40000]  lr: 1.1634e-07  ← 学习率正常!
```

## 实现细节

1. **只加载权重**: `resume=False` + `load_from='path/to/checkpoint.pth'`
2. **新的训练周期**: iter 0-40000 (相当于原来的80k-120k)
3. **学习率调度**: 
   - Warmup: iter 0-1500, lr从6e-7到6e-6
   - PolyLR: iter 1500-40000, lr从6e-6衰减到0
4. **差异化学习率**:
   - backbone: 6e-7 (lr_mult=0.1)
   - preprocessor_sec: 6e-7 (lr_mult=0.1)
   - decode_head: 6e-6 (lr_mult=1.0)

## Checkpoint保存

训练结果保存在新目录:
- `work_dirs/ascformer_rtm_finetune_40k/iter_4000.pth`
- `work_dirs/ascformer_rtm_finetune_40k/iter_8000.pth`
- ...
- `work_dirs/ascformer_rtm_finetune_40k/iter_40000.pth` ← 最终模型

## 经验教训

**Resume vs Load_from:**
- `resume=True`: 完整恢复训练状态(包括iter计数器、优化器、调度器)
  - 适用于: 训练中断后继续,保持完全相同的训练状态
  - 问题: 无法修改学习率策略
  
- `load_from`: 只加载模型权重
  - 适用于: Fine-tuning,需要新的学习率策略
  - 优点: 可以自由设置新的训练配置

**Fine-tuning的正确做法:**
当需要从已训练的模型继续训练,但要改变学习率策略时,应该使用`load_from`,而不是`resume`!

## 启动命令

```bash
./start_finetune_no_resume.sh
```

监控训练:
```bash
tail -f work_dirs/ascformer_rtm_finetune_40k/*/20*.log | grep 'Iter(train)'
```
