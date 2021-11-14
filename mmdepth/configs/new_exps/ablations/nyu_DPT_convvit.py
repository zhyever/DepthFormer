_base_ = [
    '../../_base_/models/convvit_base.py', '../../_base_/datasets/nyu.py',
    '../../_base_/iter_runtime.py', '../../_base_/schedules/schedule_cos24x_iter.py'
]

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
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