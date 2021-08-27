# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='DepthResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        style='pytorch',
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='UpsampleHead',
        in_channels=[2048, 1024, 512, 256, 64],
        in_index=[0, 1, 2, 3, 4],
        up_sample_channels=[2048, 1024, 512, 256, 128],
        channels=128, # last one
        align_corners=True, # for upsample
        input_transform="resize_concat",
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
