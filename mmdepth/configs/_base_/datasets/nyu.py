# dataset settings
dataset_type = 'NYUDataset'
data_root = './data/nyu/'
# data_root = '/nfs/lizhenyu1/data_depth_annotated/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='DepthNYUCrop', depth=True),
    dict(type='DepthRandomRotate', prob=0.5, degree=2.5),
    dict(type='DepthRandomFlip', prob=0.5),
    dict(type='DepthRandomCrop', crop_size=(416, 544)),
    dict(type='DepthColorAug', prob=1, gamma_range=[0.9, 1.1], brightness_range=[0.75, 1.25], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DepthDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='DepthNYUCrop', depth=False),
    # dict(type='DepthNYUTestCrop'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(0, 0),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_train.txt',
        pipeline=train_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_test.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='nyu_video.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10))

