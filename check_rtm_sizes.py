import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('configs/ascformer/ascformer_rtm.py')
model = MODELS.build(cfg.model)
model.eval()

batch_size = 2
input_size = (512, 512)
fake_input = torch.randn(batch_size, 3, input_size[0], input_size[1])

print("="*60)
print("原始RTM 模型特征图尺寸分析")
print("="*60)
print(f"\n输入图像尺寸: {fake_input.shape}")

with torch.no_grad():
    # Backbone输出
    feats = model.backbone(fake_input)
    print(f"\nBackbone 输出:")
    for i, feat in enumerate(feats):
        h, w = feat.shape[2], feat.shape[3]
        ratio = input_size[0] / h
        print(f"  Stage {i}: {feat.shape} -> H={h}, W={w} (1/{ratio}x)")
    
    # Decode head输出
    try:
        seg_logits = model.decode_head.forward(feats)[0]
        print(f"\nDecode Head 输出:")
        h, w = seg_logits.shape[2], seg_logits.shape[3]
        ratio = input_size[0] / h
        print(f"  seg_logits: {seg_logits.shape} -> H={h}, W={w} (1/{ratio}x)")
        
        print(f"\n尺寸变化:")
        print(f"  输入: 512x512")
        print(f"  输出: {h}x{w}")
        if h == 512:
            print(f"  ✅ 输出尺寸与输入一致")
        else:
            print(f"  ⚠️  输出尺寸是输入的 1/{ratio}x")
    except Exception as e:
        print(f"Decode head出错: {e}")

print("\n" + "="*60)
