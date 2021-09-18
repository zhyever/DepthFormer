# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
conv_stem_norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ConvSwinTransformer',
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
        pretrain_style='official',
        conv_norm_cfg=conv_stem_norm_cfg,
        depth=50,
        num_stages=0),
    decode_head=dict(
        type='Unet3UpsampleHead',
        in_channels=[1024, 512, 256, 128, 64],
        in_index=[0, 1, 2, 3, 4],
        mid_channel=64,
        channels=320, # last one
        align_corners=True, # for upsample
        input_transform="resize_concat",
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable



