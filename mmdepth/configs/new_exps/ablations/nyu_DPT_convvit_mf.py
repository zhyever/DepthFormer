_base_ = [
    '../../_base_/models/convvit_base.py', '../../_base_/datasets/nyu.py',
    '../../_base_/iter_runtime.py', '../../_base_/schedules/schedule_cos24x_iter.py'
]

model = dict(
    neck=dict(
        type='DepthFusionMultiLevelNeck',
        in_channels=[64, 768, 768, 768, 768],
        out_channels=[64, 768, 768, 768, 768],
        embedding_dim=256,
        scales=[1, 4, 2, 1, 0.5]),
    decode_head=dict(
        type='UpsampleHead',
        in_channels=[768, 768, 768, 768, 64],
        in_index=[0, 1, 2, 3],
        up_sample_channels=[256, 256, 256, 256, 128],
        channels=128,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, min_depth=1e-3, max_depth=10))
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    type='AdamW',
    lr=0.00006,
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