import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 훈련")
    parser.add_argument('--config', default='./configs/dcn/cascade-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/DCNkFold', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='../dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=15, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.0001, help='학습률')
    parser.add_argument('--num-classes', type=int, default=10, help='클래스 수')
    parser.add_argument('--seed', type=int, default=2022, help='랜덤 시드')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='사용할 GPU ID')
    parser.add_argument('--pretrained', default='./pretrained/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth', help='Pretrained 모델 경로 (Optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()
    # 설정 파일 로드
    cfg = Config.fromfile(args.config)
    # 클래스 정의
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    #
    # 모델 설정 수정
    for settings in cfg.model.roi_head.bbox_head:
        settings.num_classes = args.num_classes


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
        brightness_delta=0,  # 비활성화 (무작위 밝기 조정 제거)
        contrast_range=(0.5, 1.5),  # 비활성화 (무작위 대비 조정 제거)
        saturation_range=(0.5, 1.5),  # 채도 조정만 활성화
        hue_delta=18  # 색상 조정만 활성화
    ),
    dict(type='PackDetInputs')
]
    test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='MultiScaleFlipAug', scale=[(img_size,img_size), (img_size, 800)], keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
    # 데이터셋 설정 수정
    
    cfg.train_dataloader.dataset.ann_file = osp.join(args.data_root, 'train.json')
    cfg.train_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'train')
    cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.train_dataloader.batch_size = args.batch_size

    cfg.val_dataloader.dataset.ann_file = osp.join(args.data_root, 'val.json')
    cfg.val_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'val')
    cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.val_dataloader.batch_size = args.batch_size

    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'test')
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.batch_size = args.batch_size


        # Multi2
    # cfg.train_dataloader.dataset = dict(
    #     # use MultiImageMixDataset wrapper to support mosaic and mixup
    #     type='MultiImageMixDataset',
    #     max_refetch = 15,
    #     dataset=dict(
    #         type='CocoDataset',
    #         data_root=args.data_root,
    #         ann_file='../dataset/train.json',
    #         metainfo = dict(classes=classes),
    #         data_prefix=dict(img = osp.join(args.data_root, 'train')),
    #         pipeline=[
    #             dict(type='LoadImageFromFile', backend_args=backend_args),
    #             dict(type='LoadAnnotations', with_bbox=True)
    #         ],
    #         filter_cfg=dict(filter_empty_gt=True, min_size=32),
    #         backend_args=backend_args),
    #     pipeline=train_pipeline)
    # cfg.train_dataloader.dataset.pipeline = train_pipeline
    # cfg.test_dataloader.dataset.pipeline = test_pipeline
    #

    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val.json')
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')

    cfg.optim_wrapper.optimizer = dict(type='AdamW', 
                         lr=args.lr, 
                         weight_decay=0.0001)
    cfg.optim_wrapper.type='OptimWrapper'

    cfg.param_scheduler[0] = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
    cfg.param_scheduler[1] = dict(
    type='CosineAnnealingLR',
    T_max=args.epochs,              # 학습률이 한 주기에서 감소하는 최대 단계 (보통 epoch 수와 동일하게 설정)
    eta_min=0.000000005,       # 학습률의 최소값 (필요시 조정)
    begin=1,               # 학습 시작 epoch
    end=args.epochs,                # 학습 종료 epoch
    by_epoch=True          # 에폭 단위로 적용
    )

    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = 1

    cfg.default_hooks.timer = dict(type='IterTimerHook')
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')
###
#     cfg.neck = dict(
#     type='RFP',
#     rfp_steps=3,  # Number of recursive steps, can be adjusted as needed
#     aspp_out_channels=256,  # The number of output channels for ASPP layer, commonly set to 256
#     rfp_backbone=dict(
#         type='SwinTransformer',
#         pretrain_img_size=384,
#         in_channels=3,
#         embed_dims=192,
#         patch_size=4,
#         window_size=12,
#         mlp_ratio=4,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         strides=(4, 2, 2, 2),
#         out_indices=(0, 1, 2, 3),
#         qkv_bias=True,
#         qk_scale=None,
#         patch_norm=True,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.3,
#         use_abs_pos_embed=False,
#         act_cfg=dict(type='GELU'),
#         norm_cfg=dict(type='LN'),
#         init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')
#     ),
#     in_channels=[384, 768, 1536, 3072],
#     out_channels=256,
#     num_outs=5
# )
    # cfg.model.rpn_head.anchor_generator.ratios = [0.25, 0.35, 0.5, 1.0, 1.5, 2.0]
###

    # 기본 훈련 설정의 hook 변경 가능
    cfg.default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    # param_scheduler=dict(type='ParamSchedulerHook'),
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

    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids

    # Runner 생성 및 훈련 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()