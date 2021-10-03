_base_ = [
    '../../_base_/models/res50Unet.py', '../../_base_/datasets/nyu.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetV2',
        name='BiT-M-R50x3',
        pretrained='./nfs/checkpoints/BiT-M-R50x3.npz'),
    neck=dict(
        type='DepthFusionMultiLevelNeck',
        in_channels=[192, 768, 1536, 3072, 6144],
        out_channels=[192, 768, 1536, 3072, 6144],
        embedding_dim=256,
        scales=[1, 1, 1, 1, 1]),
    decode_head=dict(
        in_channels=[6144, 3072, 1536, 768, 192],
        in_index=[0, 1, 2, 3, 4],
        up_sample_channels=[3072, 1536, 768, 384, 128],
        channels=128, # last one
        min_depth=1e-3,
        max_depth=10)
    )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )

# optimizer
optimizer = dict(type='AdamW', lr=1.5e-4, betas=(0.95, 0.99), weight_decay=0.01,)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 1000,
    min_lr_ratio=1e-8,
    by_epoch=True) # test add by_epoch false
# learning policy
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=5, interval=1)
evaluation = dict(by_epoch=True, start=1, interval=1, pre_eval=True)