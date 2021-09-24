_base_ = [
    '../../_base_/models/convswin_base.py', '../../_base_/datasets/kitti_benchmark.py',
    '../../_base_/iter_runtime.py', '../../_base_/schedules/schedule_cos24x_iter.py'
]

model = dict(
    pretrained='./checkpoints/swin_large_patch4_window12_384_22k.pth', # noqa
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12),
    neck=dict(
        type='DepthFusionMultiLevelNeck',
        in_channels=[64, 192, 384, 768, 1536],
        out_channels=[64, 192, 384, 768, 1536],
        embedding_dim=512, # 384?
        scales=[1, 1, 1, 1, 1]),
    decode_head=dict(
        type='UpsampleHead',
        in_channels=[1536, 768, 384, 192, 64],
        in_index=[0, 1, 2, 3],
        up_sample_channels=[1536, 768, 384, 192, 64],
        channels=64,
        min_depth=1e-3,
        max_depth=80,
        att_fusion=False,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, min_depth=1e-3, max_depth=80)
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    type='AdamW',
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )

find_unused_parameters=True

# search the best
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=400, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel_all',
                  greater_keys=("a1_all", "a2_all", "a3_all"), 
                  less_keys=("abs_rel_all", "rmse_all"))

# change 1/10 warmup_ratio to converge
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1600 * 8,
    warmup_ratio=1.0 / 1000,
    min_lr_ratio=1e-8,
    by_epoch=False)

# change interval to 10 to check convergement
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])