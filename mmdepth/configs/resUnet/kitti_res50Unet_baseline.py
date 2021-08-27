_base_ = [
    '../_base_/models/res50Unet.py', '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos_20e.py'
]

model = dict(
    pretrained='torchvision://resnet50',
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0, min_depth=1e-3, max_depth=80)),
    )
