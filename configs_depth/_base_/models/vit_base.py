# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',  # noqa
    backbone=dict(
        type='VisionTransformer',
        img_size=(352, 704),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    neck=dict(
        type='DepthMultiLevelNeck',
        in_channels=[768, 768, 768, 768],
        out_channels=[768, 768, 768, 768],
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UpsampleHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        up_sample_channels=[256, 256, 256, 256],
        channels=256, # last one
        align_corners=True, # for upsample
        input_transform="resize_concat",
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
