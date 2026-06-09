# RTM对比学习详解

## 📍 对比学习在哪里？

### 1. **代码位置**

对比学习相关代码分布在以下几个文件：

```
ASCFormer/
├── mmseg/models/custom_decode_heads/
│   ├── progressive_contrastive_head.py  # ← 解码头(包含对比学习开关)
│   └── contrastive_head.py              # ← 对比学习解码头实现
├── mmseg/models/losses/
│   └── contrastive_loss.py              # ← 对比学习损失函数
└── configs/ascformer/
    └── ascformer_rtm_progressive.py     # ← 配置文件(use_contrastive=False)
```

### 2. **当前状态**

🔴 **对比学习目前是关闭的！**

在配置文件中:
```python
decode_head=dict(
    type='ProgressiveContrastiveHead',
    use_contrastive=False,  # ← 关闭状态
    ...
)
```

## 🔍 对比学习的工作原理

### 架构图解

```
输入图像
   ↓
[RTM Backbone] (4个尺度的特征: s1, s2, s3, s4)
   ↓
┌──────────────────────────────────────────────┐
│  Progressive Contrastive Head                │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │ 1. NonLocal Mask模块                │    │
│  │    - 空间注意力 (Spatial Attention)  │    │
│  │    - 通道注意力 (Channel Attention)  │    │
│  │    - 生成mask + 特征z                │    │
│  └─────────────────────────────────────┘    │
│             ↓                                │
│  ┌─────────────────────────────────────┐    │
│  │ 2. 残差门控 (Residual Gating)        │    │
│  │    s_refined = s + (s * σ(mask_up)) │    │
│  │    - 从低分辨率到高分辨率逐步细化     │    │
│  └─────────────────────────────────────┘    │
│             ↓                                │
│  ┌─────────────────────────────────────┐    │
│  │ 3. 渐进式监督 (Progressive Supervision)│  │
│  │    - mask1 (512x512) weight: 0.6    │    │
│  │    - mask2 (256x256) weight: 0.8    │    │
│  │    - mask3 (128x128) weight: 1.0    │    │
│  │    - mask4 (64x64)   weight: 1.0    │    │
│  └─────────────────────────────────────┘    │
│             ↓                                │
│  ┌─────────────────────────────────────┐    │
│  │ 4. 损失计算                          │    │
│  │    - Focal Loss (分割损失)           │    │
│  │    - [Contrastive Loss] (对比学习)   │    │ ← 这里!
│  └─────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
   ↓
最终预测
```

### 对比学习在哪个环节？

**位置**: 在解码头(decode_head)的损失计算阶段

```python
class ProgressiveContrastiveHead(BaseDecodeHead):
    def __init__(self, use_contrastive=False, ...):
        self.use_contrastive = use_contrastive  # ← 开关
        
        if self.use_contrastive:
            # 初始化对比学习损失
            self.contrastive_loss = SupConLoss(...)
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        # 1. 计算分割损失 (Focal Loss)
        losses = {
            'loss_mask1': focal_loss1 * 0.6,
            'loss_mask2': focal_loss2 * 0.8,
            'loss_mask3': focal_loss3 * 1.0,
            'loss_mask4': focal_loss4 * 1.0,
        }
        
        # 2. 如果启用对比学习,计算对比损失
        if self.use_contrastive:
            contrastive_loss = self.contrastive_loss(features, labels)
            losses['loss_contrastive'] = contrastive_loss * 0.1
        
        return losses
```

## 🎯 对比学习的作用

### 1. **特征空间优化**

**传统分割损失的问题:**
- 只关心像素级分类正确性
- 不关心特征的判别性
- 特征空间可能混乱

```
特征空间 (没有对比学习):
  真实区域  篡改区域
    •   •     ×   ×
  •   ×   •   ×   •    ← 特征混在一起,难以区分
    •   ×   •   ×
```

**对比学习的优化:**
- 拉近相似样本 (同类)
- 推远不同样本 (不同类)
- 特征空间更清晰

```
特征空间 (有对比学习):
  真实区域          篡改区域
    •   •             ×   ×
  •   •   •         ×   ×   ×    ← 类内紧凑,类间分离
    •   •             ×   ×
```

### 2. **提升Tampered检测**

当前问题:
- Tampered IoU仅15.6%
- 样本不平衡 (篡改区域少)
- 特征不够判别

对比学习的帮助:
- 强化篡改区域的特征表示
- 通过对比学习,即使样本少,也能学到判别性特征
- 更好地区分真实vs篡改

### 3. **SupConLoss工作原理**

```python
# 监督对比学习损失
class SupConLoss:
    def forward(self, features, labels):
        # features: [B, N_views, D] - 特征向量
        # labels: [B] - 类别标签 (0=真实, 1=篡改)
        
        # 1. 计算特征相似度矩阵
        sim_matrix = features @ features.T / temperature
        
        # 2. 构建正负样本对
        # 正样本: 同类别的样本对
        # 负样本: 不同类别的样本对
        pos_mask = (labels == labels.T)  # 同类
        neg_mask = (labels != labels.T)  # 不同类
        
        # 3. 对比学习目标:
        # - 拉近正样本对 (增大相似度)
        # - 推远负样本对 (减小相似度)
        loss = -log(exp(sim_pos) / sum(exp(sim_all)))
        
        return loss
```

## 📊 效果对比

### 理论预期

```
场景                    | 没有对比学习 | 有对比学习 | 提升
---------------------- | ----------- | --------- | -----
Tampered IoU           | 15.6%       | 18-22%    | +15-40%
整体 mIoU              | 57.16%      | 58-60%    | +1-3%
特征判别性             | 中          | 高        | 显著提升
类别混淆               | 较多        | 较少      | 改善
```

### 为什么对比学习更有效？

1. **多目标优化**
   - 分割损失: 学习正确分类
   - 对比损失: 学习判别性特征
   - 两者互补,效果更好

2. **解决样本不平衡**
   - Tampered样本少,监督信号弱
   - 对比学习利用样本间关系,增强学习信号

3. **特征空间正则化**
   - 防止过拟合
   - 特征更加鲁棒

## ✅ 如何启用对比学习？

### 方案1: 修改配置文件 (推荐)

创建新配置 `ascformer_rtm_contrastive_finetune.py`:

```python
_base_ = './ascformer_rtm_progressive.py'

# 从80k模型加载权重
resume = False
load_from = 'work_dirs/ascformer_rtm_progressive/iter_80000.pth'

# 修改decode_head,启用对比学习
decode_head = dict(
    type='ProgressiveContrastiveHead',
    use_contrastive=True,  # ← 开启对比学习!
    contrastive_weight=0.1,  # 对比损失权重
    temperature=0.07,        # 温度参数
    ...
)

# 学习率配置
optim_wrapper = dict(
    optimizer=dict(lr=1e-6),  # 保守学习率
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'contrastive': dict(lr_mult=1.0),  # 对比学习模块正常学习率
        }
    )
)
```

### 方案2: 从头训练 (如果方案1效果不好)

如果在80k基础上加对比学习效果不明显,可能需要从头训练:

```python
# 从预训练backbone开始,加入对比学习
resume = False
load_from = 'pretrained/mit_b2.pth'

decode_head = dict(
    use_contrastive=True,
    ...
)

train_cfg = dict(max_iters=100000)  # 从头训练100k轮
```

## 🔧 实现检查清单

- [ ] 确认对比学习损失已注册 (`contrastive_loss.py`)
- [ ] 确认解码头支持对比学习 (`progressive_contrastive_head.py`)
- [ ] 创建新配置文件,设置 `use_contrastive=True`
- [ ] 设置合适的对比损失权重 (0.05-0.2)
- [ ] 调整学习率 (建议1e-6)
- [ ] 训练并监控Tampered IoU变化

## 📝 总结

**对比学习加在哪里？**
→ 在decode_head的loss_by_feat()中,作为额外的损失项

**对比学习做什么？**
→ 优化特征空间,让同类样本更近,不同类样本更远

**为什么需要它？**
→ 突破监督学习的性能瓶颈,特别是提升Tampered这种少数类的检测

**如何启用？**
→ 配置文件中设置 `use_contrastive=True`

**预期效果？**
→ Tampered IoU从15.6%提升到18-22%,整体mIoU提升1-3%

