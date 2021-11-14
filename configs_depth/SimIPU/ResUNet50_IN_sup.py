_base_ = [
    '../_base_/models/res50Unet.py', '../_base_/datasets/nyu.py',
    '../_base_/epoch_runtime.py', '../_base_/schedules/schedule_cos20x.py'
]

model = dict(
    pretrained='https://download.pytorch.org/models/resnet50-19c8e357.pth', # noqa
    decode_head=dict(
        max_depth=10,
    ))