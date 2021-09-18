_base_ = [
    '../_base_/models/res101Unet3.py', '../_base_/datasets/kitti.py',
    '../_base_/iter_runtime.py', '../_base_/schedules/schedule_cos24_iter.py'
]

model = dict(
    pretrained='torchvision://resnet101',
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, min_depth=1e-3, max_depth=80)),
    )


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.95, 0.99), weight_decay=0.01,)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )

find_unused_parameters=True