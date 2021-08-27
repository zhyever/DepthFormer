_base_ = [
    '../_base_/models/res50Unet.py', '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_10k.py'
]

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
    ))

optimizer = dict(type='AdamW', lr=3e-5, betas=(0.95, 0.99), weight_decay=0.01,)