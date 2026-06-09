custom_imports = dict(imports=['mmseg.models.custom_decode_heads'], allow_failed_imports=False)

_base_ = [
    '../_base_/datasets/rtm_crop.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
num_classes = 2
quality = 80

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'

data_preprocessor = dict(
    type='SegDataPreProcessorWithExtra',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    copy_img = True,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='MyModelFull',
    merge_input=True,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='AsymCMNeXtV2',
        use_rectifier=False,
        in_stages=(1,0,0),
        extra_patch_embed=dict(
            in_channels=64,
            embed_dims=128,
            kernel_size=3,
            stride=1,
            reshape=True,
        ),
        out_indices=(0, 1, 2, 3),
        backbone_main=dict(
            type='MixVisionTransformer',
            pretrained=checkpoint,
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),
        # Sec Encoder
        backbone_extra=dict(
            type='HubVisionTransformer',
            pretrained=checkpoint,
            in_channels=3,
            embed_dims=64,
            modals=['dct', 'srm', 'ela'],
            in_modals=(2,3,3,3),    # asym input
            skip_patch_embed_stage=1,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            # mlp_ratios=(8, 8, 4, 4),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1),
        fuser=dict(
            type='NATFuserBlock',
            kernel_size=5,
            gated=True,
            post_attn=True,
            attn_mode='cross',
        ),
    ),
    # data extractor for the second stream
    preprocessor_sec=[
        [
            'dct',
            dict(
                type='DCTProcessor',
                in_channels=1,
                embed_dims=64,
                num_heads=1,
                patch_size=3,
                stride=1,
                sr_ratio=4,
                out_channels=64,
                norm_cfg=norm_cfg,
                reduce_neg=False,
            )
        ],
        [
            'ela',
            dict(
                type='NoFilter',
            )
        ],
        [
            'img',
            dict(
                type='SRMConv2d_simple',
                inc=3,
                learnable=False,
            ),
        ],
    ],
    decode_head=dict(
        type='ProgressiveContrastiveHead',  # Progressive decoder
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,  # kept for compatibility
        dropout_ratio=0.1,
        num_classes=1,  # Single channel output for BCE loss
        norm_cfg=norm_cfg,
        align_corners=False,
        crop_size=crop_size,  # Feature size alignment
        reduce_scales=[4, 2, 2, 1],  # Scale factors for NonLocal modules
        use_contrastive=False,  # Disable contrastive learning in first stage
        # Note: loss is computed internally with median-frequency weighting
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='RandomCropWithExtra',
         crop_size=crop_size,
         stride=8,
         extra_keys=('dct', 'ela'),
         cat_max_ratio=1.0,
         ensure_fg_prob=0.7,
         ensure_fg_min_pixels=128,
         ensure_fg_max_retry=20,
         ensure_fg_min_ratio = 0,
         ensure_fg_select_best = False),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='LoadAnnotations', binary=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela'))
]

train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline), num_workers=4)
val_dataloader = dict(dataset=dict(
    pipeline=test_pipeline,
))
test_dataloader = dict(dataset=dict(
    pipeline=test_pipeline,
))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=3, # 模拟更大batch_size,自己改的
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),  # 梯度裁剪防止梯度爆炸
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'rpb': dict(decay_mult=0.),
        }))

#train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=240000, val_interval=8000) # 每24000次验证一次，自己改
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2400), # 240000的1%,从800改成2400
    dict(
        type='PolyLR',
        eta_min=1e-7, # 从0改为1e-7
        power=1.0,
        begin=2400,
        end=240000, # 从80000改成240000
        by_epoch=False,
    )
]

val_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore', 'aFscore'])
test_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore'])

find_unused_parameters=True
# 覆盖default_hooks以启用完整的结果保存功能
# default_hooks = dict(
#     visualization=dict(
#         type='SegVisualizationHook',
#         draw=True,           # 启用可视化
#         interval=1,          # 每张图都保存
#         show=False           # 不显示窗口
#     ),
#     save_result=dict(
#         type='SegResultHook',
#         interval=1,
#         binary=True,
#         draw=True,           # 必须为True才会保存
#         save_mask=True,      # 保存二值掩码
#         save_prob=True,      # 保存概率热力图
#         use_sigmoid=True,    # 使用sigmoid（因为是二分类单通道输出）
#         save_dir='work_dirs/ascformer_rtm_progressive_acc3_crop0.7/result'
#     )
# )
