_base_ = [
    '../_base_/models/convswin_base.py', '../_base_/datasets/kitti.py',
    '../_base_/iter_runtime.py', '../_base_/schedules/schedule_cos24_iter.py'
]

model = dict(
    pretrained='./checkpoints/swin_base_patch4_window12_384_22k.pth', # noqa
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12),
    neck=dict(
        type='DepthFusionMultiLevelNeck', # interact
        in_channels=[64, 128, 256, 512, 1024],
        out_channels=[64, 128, 256, 512, 1024],
        scales=[1, 1, 1, 1, 1],
        embedding_dim=64),
    decode_head=dict(
        in_channels=[1024, 512, 256, 128, 64],
        in_index=[0, 1, 2, 3, 4], # no use, align to mmseg.
        up_sample_channels=[1024, 512, 256, 128, 64],
        min_depth=1e-3,
        max_depth=80,
        att_fusion=False
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))


# By default, models are trained on 8 GPUs with 2 images per GPU
# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=8,
#     )
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )

find_unused_parameters=True