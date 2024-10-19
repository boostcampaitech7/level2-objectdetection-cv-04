import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os
import os.path as osp
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

config_name = 'retinanet_x101_64x4d_fpn_1x_coco.py'
workdir_name = 'workdir_' + config_name
os.makedirs(workdir_name, exist_ok = True)

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 훈련")
    parser.add_argument('--config', default='/data/ephemeral/home/KenLee/level2-objectdetection-cv-04/mmdetection3/configs/retinanet/retinanet_x101-64x4d_fpn_1x_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='/data/ephemeral/home/KenLee/level2-objectdetection-cv-04/mmdetection3/'+workdir_name, help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='/data/ephemeral/home/KenLee/level2-objectdetection-cv-04/dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=20, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=4, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.02, help='학습률')
    parser.add_argument('--num-classes', type=int, default=10, help='클래스 수')
    parser.add_argument('--seed', type=int, default=2022, help='랜덤 시드')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='사용할 GPU ID')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()

    # 설정 파일 로드
    cfg = Config.fromfile(args.config)

    # 클래스 정의
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # 2stage 모델에서 사용
    # cfg.model.roi_head.bbox_head.num_classes = 10
    
    # 1stage 모델에서 사용
    cfg.model.bbox_head.num_classes = 10

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

    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val.json')
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')
    cfg.val_evaluator.classwise = True

    #cfg.optim_wrapper.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    #cfg.optim_wrapper.type='OptimWrapper'
    cfg.optim_wrapper = dict(
        type = 'OptimWrapper',
        optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001), # 0.02
        # Experiments show that there is no need to turn on clip_grad.
        # paramwise_cfg=dict(norm_decay_mult=0.))
        clip_grad=dict(max_norm=10, norm_type=2),
        accumulative_counts = 4
        )

    # cfg.param_scheduler = [
    #     dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    #     dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
    # ]

    cfg.default_hooks.timer = dict(type='IterTimerHook')
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')

    img_size = 512

    #pipeline 변경
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=cfg.backend_args),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs')
    ]
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=cfg.backend_args),
        dict(type='Resize', scale=(img_size, img_size), keep_ratio=True),
        # If you don't have a gt annotation, delete the pipeline
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
        ]
    cfg.img_scales = [(256, 256), (512, 512), (1024, 1024)]
    cfg.tta_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(
            type='TestTimeAug',
            transforms=[[
                dict(type='Resize', scale=s, keep_ratio=True) for s in cfg.img_scales
            ], [
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ], [dict(type='LoadAnnotations', with_bbox=True)],
                        [
                            dict(
                                type='PackDetInputs',
                                meta_keys=('img_id', 'img_path', 'ori_shape',
                                        'img_shape', 'scale_factor', 'flip',
                                        'flip_direction'))
                        ]])
    ]

    #기본 훈련 설정의 hook 변경 가능
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(type='CheckpointHook', interval=1),
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