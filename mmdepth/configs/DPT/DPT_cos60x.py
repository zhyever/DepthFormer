_base_ = [
    '../_base_/models/DPT_base.py', '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos60x.py'
]

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
    ))

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=8,
    )

optimizer = dict(
    # _delete_=True,
    # type='AdamW',
    # lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

