# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        pretrain_style='official'),
    neck=dict(
        type='DepthMultiLevelNeck',
        in_channels=[96, 192, 384, 768],
        out_channels=[96, 192, 384, 768],
        scales=[1, 1, 1, 1]),
    decode_head=dict(
        type='UpsampleHead',
        in_channels=[768, 384, 192, 96],
        in_index=[0, 1, 2, 3],
        up_sample_channels=[768, 384, 192, 96],
        channels=96, # last one
        align_corners=True, # for upsample
        input_transform="resize_concat",
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable



