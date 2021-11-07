_base_ = [
    '../_base_/models/convswin_base.py', '../_base_/datasets/kitti_benchmark.py',
    '../_base_/iter_runtime.py', '../_base_/schedules/schedule_benchmark.py'
]

model = dict(
    pretrained='./nfs/checkpoints/swin_large_patch4_window7_224_22k.pth', # noqa
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    neck=dict(
        type='DepthFusionMultiLevelNeck',
        in_channels=[64, 192, 384, 768, 1536],
        out_channels=[64, 192, 384, 768, 1536],
        embedding_dim=512,
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
    workers_per_gpu=6,
    )

find_unused_parameters=False

# search the best abs_rel
evaluation = dict(by_epoch=False, 
                  start=22400,
                  interval=400,
                  pre_eval=True, 
                  rule='less', 
                  save_best='silog_all',
                  greater_keys=("a1_all", "a2_all", "a3_all"), 
                  less_keys=("abs_rel_all", "rmse_all", "silog_all"))

# small warmup_ratio to ensure convergence
lr_config = dict(
    warmup_ratio=1.0 / 1000
    )

# change interval to 10 to check convergence
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])