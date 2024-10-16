_base_ = './retinanet_r50_fpn_1x_coco.py'

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

# Add the log_config to integrate WandB logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='retinanet_x101_project',
                entity='your_wandb_username',  # Replace with your WandB username
                config=dict(
                    lr=0.01,  # or cfg.optimizer['lr'] if available
                    batch_size=2,  # or cfg.data.samples_per_gpu if available
                    num_epochs=12,  # or cfg.runner.max_epochs if available
                    backbone='ResNeXt',
                    depth=101
                )
            )
        )
    ]
)
