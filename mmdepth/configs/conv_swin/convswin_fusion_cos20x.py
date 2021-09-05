_base_ = [
    '../_base_/models/convswin_base.py', '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_cos20x.py'
]

model = dict(
    pretrained='./checkpoints/swin_tiny_patch4_window7_224.pth', # noqa
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        pretrain_style='official'),
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
        att_fusion=True
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
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    )

find_unused_parameters=True