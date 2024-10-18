_base_ = './retinanet_r50_fpn_1x_coco.py'

# from albumentations import HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast
# import albumentations as A
# import numpy as np
# import mmcv

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'))
)
# # Add the log_config to integrate WandB logging
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(
#             type='WandbLoggerHook',
#             init_kwargs=dict(
#                 project='retinanet_x101_project',
#                 entity='jongseo001111-naver',  # Replace with your WandB username
#                 config=dict(
#                     lr = 0.01, 
#                     batch_size = 4,  
#                     num_epochs = 12,  
#                     backbone ='ResNeXt',
#                     depth = 101
#                 )
#             )
#         )
#     ]
# )

# # Custom Albumentations Transform
# class AlbumentationsTransform:
#     def __init__(self):
#         self.transform = A.Compose([
#             HorizontalFlip(p=0.5),
#             ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5),
#             RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
#         ])

#     def __call__(self, results):
#         img = results['img']
#         augmented = self.transform(image=img)
#         results['img'] = augmented['image']
#         return results

# # Instantiate Albumentations Transform
# albumentations_transform = AlbumentationsTransform()

# # Define training pipeline with custom Albumentations
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     albumentations_transform,  # Use the custom Albumentations transform here as a callable
#     dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),  # Add resize step
#     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

# # Update data pipeline settings without duplicating data keys
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type='CocoDataset',
#         ann_file='/data/ephemeral/home/dataset/train.json',
#         img_prefix='/data/ephemeral/home/dataset/train',
#         pipeline=train_pipeline
#     ),
#     val=dict(
#         type='CocoDataset',
#         ann_file='/data/ephemeral/home/dataset/val.json',
#         img_prefix='/data/ephemeral/home/dataset/val',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(1024, 1024),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img']),
#                 ])
#         ]
#     ),
#     test=dict(
#         type='CocoDataset',
#         ann_file='/data/ephemeral/home/dataset/test.json',
#         img_prefix='/data/ephemeral/home/dataset/test',
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(
#                 type='MultiScaleFlipAug',
#                 img_scale=(1024, 1024),
#                 flip=False,
#                 transforms=[
#                     dict(type='Resize', keep_ratio=True),
#                     dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#                     dict(type='Pad', size_divisor=32),
#                     dict(type='ImageToTensor', keys=['img']),
#                     dict(type='Collect', keys=['img']),
#                 ])
#         ]
#     )
# )