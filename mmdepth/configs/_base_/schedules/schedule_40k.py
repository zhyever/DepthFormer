# optimizer
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.95, 0.99), weight_decay=0.01,)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=5, interval=1000)
evaluation = dict(by_epoch=False, 
                  interval=1000,
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel_all',
                  greater_keys=("a1_all", "a2_all", "a3_all"), 
                  less_keys=("abs_rel_all", "rmse_all"))