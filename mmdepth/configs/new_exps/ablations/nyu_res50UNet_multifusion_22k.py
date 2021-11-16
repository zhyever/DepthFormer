_base_ = [
    '../../_base_/models/res50Unet.py', '../../_base_/datasets/nyu.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetV2',
        name='BiT-M-R50x1',
        pretrained='/mnt/10-5-108-187/xxx/workspace/python_workspace/mmsegmentation/BiT-M-R50x1.npz'),
    neck=dict(
        type='DepthFusionMultiLevelNeck',
        in_channels=[64, 256, 512, 1024, 2048],
        out_channels=[64, 256, 512, 1024, 2048],
        embedding_dim=768,
        scales=[1, 1, 1, 1, 1]),
    decode_head=dict(
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    )

# optimizer
optimizer = dict(type='AdamW', lr=1.5e-4, betas=(0.95, 0.99), weight_decay=0.01,)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-8,
    by_epoch=True) # test add by_epoch false
# learning policy
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=5, interval=1)
evaluation = dict(by_epoch=True, start=1, interval=1, pre_eval=True)