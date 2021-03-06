# optimizer
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.95, 0.99), weight_decay=0.01,)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-8,
    by_epoch=True) # test add by_epoch false
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=5, interval=1)
evaluation = dict(by_epoch=True, start=1, interval=1, pre_eval=True)