from mmengine.config import Config

config = Config.fromfile('configs/ascformer/ascformer_rtm.py')

print("="*70)
print("原始 RTM 项目配置分析")
print("="*70)

print(f"\n模型类型: {config.model.type}")
print(f"\n=== Backbone (AsymCMNeXtV2) ===")
print(f"类型: {config.model.backbone.type}")
print(f"out_indices: {config.model.backbone.out_indices}")

# 查看内部backbone_main
if hasattr(config.model.backbone, 'backbone_main'):
    print(f"\n=== Inner Backbone Main ===")
    bb = config.model.backbone.backbone_main
    print(f"类型: {bb.type}")
    print(f"embed_dims: {bb.embed_dims}")
    print(f"num_stages: {bb.num_stages}")
    print(f"num_layers: {bb.num_layers}")

print(f"\n=== Decode Head ===")
print(f"类型: {config.model.decode_head.type}")
print(f"in_channels: {config.model.decode_head.in_channels}")
print(f"in_index: {config.model.decode_head.in_index}")
print(f"channels (融合后): {config.model.decode_head.channels}")
print(f"num_classes: {config.model.decode_head.num_classes}")
print(f"up_decode: {config.model.decode_head.up_decode}")
print(f"use_memory: {config.model.decode_head.use_memory}")

print(f"\n=== 预期特征图尺寸 (输入512x512) ===")
input_size = 512
stages = [(64, input_size//4), (128, input_size//8), (320, input_size//16), (512, input_size//32)]
for i, (c, s) in enumerate(stages):
    print(f"Stage {i}: C={c}, H={s}, W={s} -> shape=[B, {c}, {s}, {s}]")

print(f"\n=== Decode Head 预期输入 ===")
for i, c in enumerate(config.model.decode_head.in_channels):
    s = stages[i][1]
    print(f"输入 {i}: C={c}, H={s}, W={s}")

print(f"\n=== Decode Head 输出 (预测前) ===")
if config.model.decode_head.up_decode:
    print(f"特征融合后: C={config.model.decode_head.channels}, H=256, W=256 (上采样2倍)")
    print(f"cls_seg输出: C={config.model.decode_head.num_classes}, H=256, W=256")
else:
    print(f"特征融合后: C={config.model.decode_head.channels}, H=128, W=128")
    print(f"cls_seg输出: C={config.model.decode_head.num_classes}, H=128, W=128")

print("\n" + "="*70)
