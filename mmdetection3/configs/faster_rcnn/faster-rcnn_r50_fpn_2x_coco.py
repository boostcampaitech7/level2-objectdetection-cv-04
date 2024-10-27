_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# # 모델 설정
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=10)  # 클래스 수를 여러분의 데이터셋에 맞게 조정
#     )
# )

# classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
#            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# # 데이터셋 설정
# data_root ='/data/ephemeral/home/dataset/'
# dataset_type = 'CocoDataset'  # 또는 여러분의 데이터셋 유형

# 데이터셋 설정 수정
# train_dataloader = dict(
#     dataset=dict(
#         ann_file=data_root + 'train.json',
#         data_prefix=dict(img=data_root + 'train'),
#         metainfo=dict(classes=classes)
#     )
# )   

# val_dataloader = dict(
#     dataset=dict(
#         ann_file=data_root + 'val.json',
#         data_prefix=dict(img=data_root + 'val'),
#         metainfo=dict(classes=classes)
#     )
# )

# test_dataloader = dict(
#     dataset=dict(
#         ann_file=data_root + 'test.json',
#         data_prefix=dict(img=data_root + 'test'),
#         metainfo=dict(classes=classes)
#     )
# )

# # 평가기 설정 - 안하면 오류남
# val_evaluator = dict(
#     ann_file=data_root + 'val.json',  # 수정됨
#     backend_args=None,
#     format_only=False,
#     metric='bbox',
#     type='CocoMetric')

# # 테스트 평가기도 같은 방식으로 수정
# test_evaluator = dict(
#     ann_file=data_root + 'test.json',  # 수정됨
#     backend_args=None,
#     format_only=False,
#     metric='bbox',
#     type='CocoMetric')

# # 옵티마이저 설정
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# )

# # 파라미터 스케줄러 설정
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
# ]

# # 훈련 설정
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# # 기본 훈련 설정
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', interval=1),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='DetVisualizationHook')
# )

# # 작업 디렉토리 설정
# work_dir = './work_dirs/faster-rcnn_r50_fpn_2x_coco_trash'
