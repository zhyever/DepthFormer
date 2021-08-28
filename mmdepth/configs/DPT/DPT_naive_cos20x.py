_base_ = [
    '../_base_/models/DPT_base.py', '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos20x.py'
]

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, min_depth=1e-3, max_depth=80)),
    )

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=8,
    )
