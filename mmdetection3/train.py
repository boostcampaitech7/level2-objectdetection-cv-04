import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os.path as osp

def parse_args():
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="DETR 모델 훈련")
    parser.add_argument('--config', default='./configs/detr/detr_r50_8xb2-150e_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/detr_r50_8xb2-150e_coco_trash', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='/data/ephemeral/home/dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=24, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=64, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
=======
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 훈련")
    parser.add_argument('--config', default='./configs/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/CascadeSWINKFold0', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='../dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=20, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=4, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.0001, help='학습률')
>>>>>>> develop
    parser.add_argument('--num-classes', type=int, default=10, help='클래스 수')
    parser.add_argument('--seed', type=int, default=2022, help='랜덤 시드')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='사용할 GPU ID')
    parser.add_argument('--pretrained', default='./pretrained/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth', help='Pretrained 모델 경로 (Optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()
    # 설정 파일 로드
    cfg = Config.fromfile(args.config)
    # 클래스 정의
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
<<<<<<< HEAD

    # DETR 특정 설정
    cfg.model.type = 'DETR'
    cfg.model.backbone.type = 'ResNet'
    cfg.model.backbone.depth = 50
    cfg.model.backbone.num_stages = 4
    cfg.model.backbone.out_indices = (3,)
    cfg.model.backbone.frozen_stages = 1
    cfg.model.neck = dict(type='ChannelMapper', in_channels=[2048], out_channels=256, num_outs=1)
    cfg.model.bbox_head.type = 'DETRHead'
    cfg.model.bbox_head.num_classes = args.num_classes
    cfg.model.bbox_head.embed_dims = 256
    
    # num_queries를 bbox_head에서 제거하고 모델 레벨로 이동
    cfg.model.num_queries = 100

    # # 이미지 해상도 조절을 위한 파이프라인 수정
    # # 훈련 파이프라인만 수정
    # cfg.train_pipeline = [
    #     dict(type='LoadImageFromFile', backend_args=None),
    #     dict(type='LoadAnnotations', with_bbox=True),
    #     dict(type='Resize', scale=(224, 224), keep_ratio=False),
    #     dict(type='RandomFlip', prob=0.5),
    #     dict(type='PackDetInputs')
    # ]
=======
    #
    # 모델 설정 수정
    for settings in cfg.model.roi_head.bbox_head:
        settings.num_classes = args.num_classes
>>>>>>> develop


    ## 임시
    if args.pretrained is not None:
        cfg.load_from = args.pretrained
    ##

    # Pipeline 설정
    
    img_size = 1024
    backend_args = None
    train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=[(img_size,img_size), (img_size, 800)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=18,  # 비활성화 (무작위 밝기 조정 제거)
        contrast_range=(0.5, 1.5),  # 비활성화 (무작위 대비 조정 제거)
        saturation_range=(0.5, 1.5),  # 채도 조정만 활성화
        hue_delta=18  # 색상 조정만 활성화
    ),
    dict(type='PackDetInputs')
]
    test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(img_size,img_size), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
    cfg.train_dataloader.dataset.pipeline = train_pipeline
    cfg.val_dataloader.dataset.pipeline = test_pipeline
    cfg.test_dataloader.dataset.pipeline = test_pipeline
    # 데이터셋 설정 수정
    
    cfg.train_dataloader.dataset.ann_file = osp.join(args.data_root, 'train_fold_0.json')
    cfg.train_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'train')
    cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.train_dataloader.batch_size = args.batch_size

    cfg.val_dataloader.dataset.ann_file = osp.join(args.data_root, 'val_fold_0.json')
    cfg.val_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'train')
    cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.val_dataloader.batch_size = args.batch_size

    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'test')
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.batch_size = args.batch_size

<<<<<<< HEAD
    # # 평가기 수행 설정
    # # 이 부분은 검증(validation)과 테스트 데이터셋에 대한 평가 설정을 정의합니다.
    
    # # 검증 데이터셋에 대한 평가기 설정
    # cfg.val_evaluator = dict(
    #     type='CocoMetric',  # COCO 형식의 평가 지표를 사용합니다.
    #     ann_file=osp.join(args.data_root, 'val.json'),  # 검증 데이터셋의 주석 파일 경로
    #     metric='bbox',  # 바운딩 박스 평가 지표를 사용합니다.
    #     format_only=False,  # 결과를 COCO 형식으로만 저장하지 않고 실제 평가도 수행합니다.
    #     backend_args=None  # 백엔드 인자가 필요 없는 경우 None으로 설정합니다.
    # )
    
    # # 테스트 데이터셋에 대한 평가기 설정
    # cfg.test_evaluator = dict(
    #     type='CocoMetric',  # COCO 형식의 평가 지표를 사용합니다.
    #     ann_file=osp.join(args.data_root, 'test.json'),  # 테스트 데이터셋의 주석 파일 경로
    #     metric='bbox',  # 바운딩 박스 평가 지표를 사용합니다.
    #     format_only=False,  # 결과를 COCO 형식으로만 저장하지 않고 실제 평가도 수행합니다.
    #     backend_args=None  # 백엔드 인자가 필요 없는 경우 None으로 설정합니다.
    # )
    
    # # 이 설정들은 모델의 성능을 평가하는 데 사용되며, 
    # # 검증 및 테스트 단계에서 객체 검출의 정확도를 측정하는 데 중요합니다.
=======

        # Multi
    # cfg.train_dataloader.dataset = dict(
    # # use MultiImageMixDataset wrapper to support mosaic and mixup
    # type='MultiImageMixDataset',
    # max_refetch = 1,
    # dataset=dict(
    #     type='CocoDataset',
    #     data_root=args.data_root,
    #     ann_file=osp.join(args.data_root, 'train.json'),
    #     data_prefix=dict(img = osp.join(args.data_root, 'train')),
    #     pipeline=[
    #         dict(type='LoadImageFromFile', backend_args=backend_args),
    #         dict(type='LoadAnnotations', with_bbox=True)
    #     ],
    #     filter_cfg=dict(filter_empty_gt=False, min_size=32),
    #     backend_args=backend_args),
    # pipeline=train_pipeline)
    # cfg.train_dataloader.dataset.pipeline = train_pipeline
    # cfg.test_dataloader.dataset.pipeline = test_pipeline
    #
>>>>>>> develop

    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val_fold_0.json')
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')
    cfg.val_evaluator.classwise = True

<<<<<<< HEAD
    # 옵티마이저 설정 수정
    cfg.optim_wrapper.optimizer = dict(type='AdamW', lr=args.lr, weight_decay=0.0001)
    cfg.optim_wrapper.clip_grad = dict(max_norm=0.1, norm_type=2)

  # 학습 스케줄러 설정
    cfg.param_scheduler = [
        dict(type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=500),
        dict(type='MultiStepLR', begin=0, end=args.epochs, by_epoch=True, milestones=[10, 20], gamma=0.1)
    ]

    cfg.train_cfg.max_epochs = 30
    cfg.train_cfg.val_interval = 1

    # cfg.default_hooks.timer = dict(type='IterTimerHook')
    # cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    # cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')

    # 기본 훈련 설정의 hook 변경 가능
    # cfg.default_hooks = dict(
    #     timer=dict(type='IterTimerHook'),
    #     logger=dict(type='LoggerHook', interval=50),
    #     param_scheduler=dict(type='ParamSchedulerHook'),
    #     checkpoint=dict(type='CheckpointHook', interval=1),
    #     sampler_seed=dict(type='DistSamplerSeedHook'),
    #     visualization=dict(type='DetVisualizationHook')
    # )
    
=======
    cfg.optim_wrapper.optimizer = dict(type='AdamW', 
                         lr=args.lr, 
                         weight_decay=0.0001)
    cfg.optim_wrapper.type='OptimWrapper'

    cfg.param_scheduler[0] = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
    cfg.param_scheduler[1] = dict(
    type='MultiStepLR',
    milestones=[6,9,11,13,15,18,19],  # Specify the epochs at which to decrease the learning rate
    gamma=0.5,               # Factor by which the learning rate will be reduced
    begin=1,                 # Start iteration
    end=args.epochs,                  # End iteration (adjust as needed)
    by_epoch=True            # Whether to apply this by epoch
)

    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = 1

    cfg.default_hooks.timer = dict(type='IterTimerHook')
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')
###
    cfg.backbone = dict(
    type='SwinTransformer',
    pretrain_img_size=384,  # Adjust if required; for pretrained weights, 384 is often used
    in_channels=3,
    embed_dims=192,
    patch_size=4,
    window_size=12,
    mlp_ratio=4,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    strides=(4, 2, 2, 2),
    out_indices=(0, 1, 2, 3),
    qkv_bias=True,
    qk_scale=None,
    patch_norm=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    use_abs_pos_embed=False,
    act_cfg=dict(type='GELU'),
    norm_cfg=dict(type='LN'),
    init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')
)
    # cfg.model.rpn_head.anchor_generator.ratios = [0.25, 0.35, 0.5, 1.0, 1.5, 2.0]
###
    cfg.model.neck = dict(
    type='PAFPN',
    in_channels=[256, 512, 1024, 2048],  # Update based on actual feature map sizes
    out_channels=256,  # Output channel dimension for each level
    num_outs=5  # This indicates 5 output feature maps (typically for FPN)
)

    # 기본 훈련 설정의 hook 변경 가능
    cfg.default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50',  # Track the best bbox_mAP
        rule='greater',
        greater_keys=['coco/bbox_mAP_50'],  # mAP는 높을수록 좋다
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

    # 옵티마이저 설정 수정
    cfg.optim_wrapper.optimizer.lr = args.lr

    # 훈련 설정 수정
    cfg.train_cfg.max_epochs = args.epochs

>>>>>>> develop
    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids
    
<<<<<<< HEAD
    # 체크포인트 훅 설정 수정
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3
    )

=======
>>>>>>> develop
    # Runner 생성 및 훈련 시작
    runner = Runner.from_cfg(cfg)
    runner.train()
    
if __name__ == '__main__':
    main()