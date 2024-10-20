_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 데이터셋 설정
dataset_type = 'CocoDataset'
data_root = '/path/to/your/dataset/'  # 실제 데이터셋 경로로 변경해주세요
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

model = dict(
    type='YOLOv8',
    backbone=dict(
        type='YOLOv8Backbone',
        depth_multiple=0.33,
        width_multiple=0.50,
        return_levels=3
    ),
    neck=dict(
        type='YOLOv8Neck',
        in_channels=[256, 512, 1024],
        out_channels=256,
    ),
    bbox_head=dict(
        type='YOLOv8Head',
        num_classes=10,
        in_channels=[256, 256, 256],
        feat_channels=256,
        stacked_convs=2,
        loss=dict(
            type='YOLOv8Loss',
            num_classes=10,
        ),
    ),
    train_cfg=dict(),  # 여기에 있어야 합니다
    test_cfg=dict(     # 여기에 있어야 합니다
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True),  # 이 줄을 추가합니다
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True),  # 이 줄을 추가합니다
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# 데이터 로더 설정
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)
    ),
    pin_memory=True
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)
    ),
    pin_memory=True  # pin_memory=True는 데이터 로딩 시 GPU 메모리에 페이지 잠금을 활성화하여 데이터 전송 속도를 향상시킵니다.
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator

# 옵티마이저 설정
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# 학습 스케줄 설정
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 학습률 스케줄 설정
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=500,
        by_epoch=True,
        milestones=[350, 450],
        gamma=0.1)
]

# 기본 런타임 설정
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False