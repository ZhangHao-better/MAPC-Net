# ascformer_rtm_progressive_ft40k_contrastive_from2132.py
_base_ = './ascformer_rtm_progressive.py'

# =========================
# 1) 训练轮次：finetune 跑 80k（对齐 Exp19 的 finetune 时长）
#    - val_interval=8000：对齐 Exp19，便于横向对比
# =========================
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)

# =========================
# 2) 学习率：重启一套 finetune schedule（对齐 Exp19）
#    - warmup ~1% = 800 iters
#    - PolyLR 衰减到 eta_min=1e-7
# =========================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=800,
    ),
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=1.0,
        begin=800,
        end=80000,
        by_epoch=False,
    )
]

# =========================
# 3) 优化器：finetune lr（其余保持你原来的 accum/clip/paramwise_cfg）
#    经验：加了 contrastive 以后，建议比 exp19 更保守一点起步
#    - 默认给 6e-6（你也可以改回 1e-5 更激进）
# =========================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=3,
    optimizer=dict(
        type='AdamW',
        lr=6e-6,  # <<< 更保守：微调阶段更稳
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
# 4) 对比学习：finetune 建议缩短 warmup
#    - 从 ckpt 出发，分割已稳定，不必再 warmup 20k
#    - 建议 5k：既避免冲击，又能在 40k 内看到稳定影响
# =========================
model = dict(
    decode_head=dict(
        use_contrastive=True,
        contrastive_weight=0.005,
        contrastive_warmup_iters=500,
        contrastive_levels=('z1',),
        contrastive_proj_dim=128,
        contrastive_samples_per_img=160,
        contrastive_boundary_ratio=0.25,
        contrastive_loss_cfg=dict(
            type='SupConLoss',
            temperature=0.10,
            contrast_mode='all',
            base_temperature=0.10,
            loss_weight=1.0,
            gather=True,
            min_points=160,
        ),
    )
)

# =========================
# 5) checkpoint / 验证节奏（可选但建议加，便于对齐）
# =========================
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000)
)

# =========================
# 6) 权重加载
#    建议：命令行传 --load-from，更干净；
#    如果你想写死也可以：把路径换成 exp15 最强点的 pth
# =========================
load_from = '/home/zhlinux/RTM-progressive/ASCFormer/work_dirs/ascformer_rtm_progressive_contrastive/iter_224000.pth'
resume = False
