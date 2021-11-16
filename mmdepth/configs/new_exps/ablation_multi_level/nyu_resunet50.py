_base_ = [
    '../../_base_/models/res50Unet.py', '../../_base_/datasets/nyu.py',
    '../../_base_/runtime_epoch.py', '../../_base_/schedules/schedule_cos20x.py'
]

model = dict(
    pretrained='torchvision://resnet50',
    neck=dict(
        type='DepthFusionMultiLevelNeckSA',
        in_channels=[64, 256, 512, 1024, 2048],
        out_channels=[64, 256, 512, 1024, 2048],
        scales=[1, 1, 1, 1, 1],
        embedding_dim=768,
        abl_single_level=True),
    decode_head=dict(
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )

find_unused_parameters=True

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
)