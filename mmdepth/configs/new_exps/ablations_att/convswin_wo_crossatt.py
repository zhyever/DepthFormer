_base_ = [
    '../../_base_/models/convswin_base.py', '../../_base_/datasets/nyu.py',
    '../../_base_/iter_runtime.py', '../../_base_/schedules/schedule_30k.py'
]

model = dict(
    pretrained='./nfs/checkpoints/swin_tiny_patch4_window7_224.pth', # noqa
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        pretrain_style='official'),
    neck=dict(
        type='DepthFusionMultiLevelNeck',
        in_channels=[64, 96, 192, 384, 768],
        out_channels=[64, 96, 192, 384, 768],
        embedding_dim=256,
        scales=[1, 1, 1, 1, 1],
        cross_att=False),
    decode_head=dict(
        type='UpsampleHead',
        in_channels=[768, 384, 192, 96, 64],
        in_index=[0, 1, 2, 3],
        up_sample_channels=[768, 384, 192, 96, 64],
        channels=64,
        min_depth=1e-3,
        max_depth=10,
        att_fusion=False
    ))

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


# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )

find_unused_parameters=True

# search the best
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=400, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel_all',
                  greater_keys=("a1_all", "a2_all", "a3_all"), 
                  less_keys=("abs_rel_all", "rmse_all"))