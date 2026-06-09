custom_imports = dict(
    imports=[
        'mmseg.models.custom_decode_heads',
        'mmseg.models.losses.contrastive_loss',
        'mmseg.datasets.t_sroie',
    ],
    allow_failed_imports=False)

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
num_classes = 2
quality = 80

dataset_type = 'TSROIEDataset'
data_root = './data/sroie'
train_json = 'sroie_train_1011.json'
test_json = 'sroie_test_1011.json'

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'

data_preprocessor = dict(
    type='SegDataPreProcessorWithExtra',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    copy_img=True,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='MyModelFull',
    merge_input=True,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='AsymCMNeXtV2',
        use_rectifier=False,
        in_stages=(1, 0, 0),
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
        backbone_extra=dict(
            type='HubVisionTransformer',
            pretrained=checkpoint,
            in_channels=3,
            embed_dims=64,
            modals=['dct', 'srm', 'ela'],
            in_modals=(2, 3, 3, 3),
            skip_patch_embed_stage=1,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
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
            dict(type='NoFilter')
        ],
        [
            'img',
            dict(type='SRMConv2d_simple', inc=3, learnable=False)
        ],
    ],
    decode_head=dict(
        type='ProgressiveContrastiveHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        crop_size=crop_size,
        reduce_scales=[4, 2, 2, 1],
        use_contrastive=True,
        contrastive_loss_cfg=dict(
            type='SupConLoss',
            temperature=0.10,
            contrast_mode='all',
            base_temperature=0.10,
            loss_weight=1.0,
            gather=True,
            min_points=320,
        ),
        contrastive_weight=0.01,
        contrastive_warmup_iters=20000,
        contrastive_levels=('z1',),
        contrastive_proj_dim=128,
        contrastive_samples_per_img=160,
        contrastive_boundary_ratio=0.25,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# NOTE:
# 1) T-SROIE contains very large receipt pages, so we resize before training
#    crop, and compute ELA/DCT *after* crop to avoid full-resolution ELA/DCT.
# 2) `LoadSROIEAnnotations` converts category `text_temp` into a binary mask.

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSROIEAnnotations', binary=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(1.0, 2.0),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomCropWithExtra',
        crop_size=crop_size,
        stride=8,
        extra_keys=(),
        cat_max_ratio=1.0,
        ensure_fg_prob=0.7,
        ensure_fg_min_pixels=128,
        ensure_fg_max_retry=20,
        ensure_fg_min_ratio=0,
        ensure_fg_select_best=False),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSROIEAnnotations', binary=True),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='SyncOriShapeWithImgShape'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_json,
        data_prefix=dict(img_path='train'),
        pipeline=train_pipeline,
        use_bbox_fallback=True,
        tampered_category='text_temp'))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_json,
        data_prefix=dict(img_path='test'),
        pipeline=test_pipeline,
        test_mode=True,
        use_bbox_fallback=True,
        tampered_category='text_temp'))

test_dataloader = val_dataloader

# Keep mask-level metrics online for training monitoring.
# Final paper numbers on T-SROIE should be reported by eval_sroie_official.py.
val_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore', 'aFscore'])
test_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore'])

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts=3,
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'rpb': dict(decay_mult=0.),
        }))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=240000, val_interval=8000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2400),
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=1.0,
        begin=2400,
        end=240000,
        by_epoch=False,
    )
]

find_unused_parameters = True

# To dump prediction masks for official T-SROIE evaluation, you can add your
# existing SegResultHook here if it has already been registered in the project.
# Example:
# custom_hooks = [
#     dict(
#         type='SegResultHook',
#         interval=1,
#         binary=True,
#         draw=True,
#         save_mask=True,
#         save_prob=False,
#         use_sigmoid=True,
#         save_dir='work_dirs/ascformer_t_sroie_progressive_exp20/result')
# ]