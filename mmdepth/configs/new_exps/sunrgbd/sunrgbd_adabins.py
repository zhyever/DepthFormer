_base_ = [
    '../../_base_/datasets/sun_rgbd.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_cos20x.py'
]

model = dict(
    type = 'Adabins',
    # model training and testing settings
    max_val=10,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

find_unused_parameters=True