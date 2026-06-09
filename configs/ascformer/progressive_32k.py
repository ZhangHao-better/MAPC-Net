# ascformer_rtm_progressive_ft80k_from2132.py
_base_ = './ascformer_rtm_progressive.py'

# =========================
# 1) 训练轮次（新 run 只跑 80k，相当于在 240k 基础上继续 +80k）
# =========================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)

# =========================
# 2) 学习率：重启一套 finetune schedule
#    - warmup 1% = 800 iters
#    - PolyLR 衰减到 eta_min=1e-7
# =========================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,   # 从 10% 的 base_lr 起步，避免刚开始冲击太大
        by_epoch=False,
        begin=0,
        end=400,            # 800 / 80000 = 1%
    ),
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=1.0,
        begin=400,
        end=80000,
        by_epoch=False,
    )
]

# =========================
# 3) 优化器：只把 lr 改成 finetune 的（其余保持你原来的 accum、clip、paramwise_cfg）
#    说明：
#    - 你原 base lr=6e-5
#    - finetune 建议 6~10 倍降低：这里取 1e-5（稳）
# =========================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=3,
    optimizer=dict(
        type='AdamW',
        lr=2e-6,                 # <<< finetune lr（你也可以改成 6e-6 更保守）
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'rpb': dict(decay_mult=0.),
        }
    )
)

# =========================
# 4) 可选：如果你想让 checkpoint / 验证节奏更清晰，也可以在这里显式设一下
#    （不写也行，会继承 schedule_80k.py 里的默认 checkpoint interval=8000）
# =========================
# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000)
# )

# 不要在 config 里写死 load_from，建议命令行传 --load-from
# load_from = None
# resume = False
load_from = '/home/zhlinux/RTM-progressive/ASCFormer/work_dirs/ft80k_from2132_lr1e-5/iter_80000.pth'