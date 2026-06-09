custom_imports = dict(
    imports=[
        'mmseg.models.custom_decode_heads',
        'mmseg.models.losses.contrastive_loss',
        'mmseg.datasets.t_sroie',
    ],
    allow_failed_imports=False)

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
quality = 80

dataset_type = 'TSROIEDataset'
data_root = './data/sroie'   # 如果你的目录名真的是 srioe，就改成 ./data/srioe
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

# RTM -> T-SROIE generalization evaluation
# 为避免 Resize 后 GT 与 postprocess 后 pred 尺寸不一致，
# 使用 SyncOriShapeWithImgShape，让 BinaryIoUMetric 在 resize 后坐标系中比较。
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSROIEAnnotations', binary=True),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='SyncOriShapeWithImgShape'),
    dict(type='ELA', quality=quality),
    dict(type='BlockDCT', zigzag=True),
    dict(type='PackSegInputsWithExtra', extra_keys=('dct', 'ela')),
]

test_dataloader = dict(
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

test_evaluator = dict(
    type='BinaryIoUMetric',
    iou_metrics=['mIoU', 'mFscore', 'aFscore'])

# test-only config: do not build val loop
val_dataloader = None
val_evaluator = None
val_cfg = None

# explicitly build test loop
test_cfg = dict(type='TestLoop')

default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')