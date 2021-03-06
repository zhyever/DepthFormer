_base_ = [
    '../../_base_/models/swin_base.py', '../../_base_/datasets/nyu.py',
    '../../_base_/iter_runtime.py', '../../_base_/schedules/schedule_cos24x_iter.py'
]

model = dict(
    pretrained='./nfs/checkpoints/swin_large_patch4_window12_384_22k.pth', # noqa
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12),
    neck=dict(
        type='DepthMultiLevelNeck',
        in_channels=[192, 384, 768, 1536],
        out_channels=[192, 384, 768, 1536],
        scales=[1, 1, 1, 1]),
    decode_head=dict(
        in_channels=[1536, 768, 384, 192],
        in_index=[0, 1, 2, 3],
        up_sample_channels=[1536, 768, 384, 192],
        channels=192, # last one
        min_depth=1e-3,
        max_depth=10,
        att_fusion=False
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )

find_unused_parameters=True