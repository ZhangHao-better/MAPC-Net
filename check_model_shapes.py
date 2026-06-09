import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

config = Config.fromfile('configs/ascformer/ascformer_rtm.py')

# 构建模型
model = build_segmentor(config.model)
model.eval()

# 创建测试输入
batch_size = 1
H, W = 512, 512
inputs = {
    'inputs': torch.randn(batch_size, 3, H, W),
}

print("="*60)
print("原始 RTM 项目模型结构分析")
print("="*60)

# 打印模型类型
print(f"\n模型类型: {config.model.type}")
print(f"Backbone类型: {config.model.backbone.type}")

# 检查backbone输出
with torch.no_grad():
    if hasattr(model, 'extract_feat'):
        feats = model.extract_feat(inputs['inputs'])
        print(f"\n=== Backbone输出特征 ===")
        for i, feat in enumerate(feats):
            print(f"Stage {i}: shape={feat.shape} (B, C={feat.shape[1]}, H={feat.shape[2]}, W={feat.shape[3]})")
    
    # 检查decode_head配置
    print(f"\n=== Decode Head配置 ===")
    print(f"类型: {config.model.decode_head.type}")
    print(f"输入通道: {config.model.decode_head.in_channels}")
    print(f"输入索引: {config.model.decode_head.in_index}")
    print(f"融合通道: {config.model.decode_head.channels}")
    print(f"类别数: {config.model.decode_head.num_classes}")
    print(f"up_decode: {config.model.decode_head.get('up_decode', False)}")
    
    # 打印解码头输入形状
    if hasattr(model, 'decode_head'):
        print(f"\n=== Decode Head实际使用的输入 ===")
        selected_feats = [feats[i] for i in config.model.decode_head.in_index]
        for i, feat in enumerate(selected_feats):
            print(f"输入 {i}: shape={feat.shape}")

print("\n" + "="*60)
