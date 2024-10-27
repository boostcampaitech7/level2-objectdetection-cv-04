import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os.path as osp
import torch
from mmengine.model import BaseModel

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 훈련")
    parser.add_argument('--config', default='./configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/fold1', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='/data/ephemeral/home/dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=10, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=2, help='배치 크기')
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
    cfg.train_dataloader.dataset.ann_file = osp.join(args.data_root, 'train_fold_1.json')
    cfg.train_dataloader.dataset.data_prefix.img = osp.join(args.data_root, '')
    cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.train_dataloader.batch_size = args.batch_size

    cfg.val_dataloader.dataset.ann_file = osp.join(args.data_root, 'val_fold_1.json')
    cfg.val_dataloader.dataset.data_prefix.img = osp.join(args.data_root, '')
    cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.val_dataloader.batch_size = args.batch_size

    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'test')
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.batch_size = args.batch_size

    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val_fold_1.json')
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')

    # cfg.optim_wrapper.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    # cfg.optim_wrapper.type='OptimWrapper'

    # cfg.param_scheduler[0] = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
    # cfg.param_scheduler[1] = dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)

    cfg.train_cfg.max_epochs = 10
    cfg.train_cfg.val_interval = 1
    
    

    cfg.default_hooks.timer = dict(type='IterTimerHook')
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')


    # 옵티마이저 설정 수정
    # cfg.optim_wrapper.optimizer.lr = args.lr

    # 훈련 설정 수정
    cfg.train_cfg.max_epochs = args.epochs

    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids

    # 
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',  # Use checkpoint hook
            interval=1,
            max_keep_ckpts=3,
            save_best='coco/bbox_mAP_50',  # Track the best bbox_mAP
            rule='greater',
            greater_keys=['coco/bbox_mAP_50'],  # mAP는 높을수록 좋다
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook')
    )

    cfg.val_evaluator.classwise = True

    # 이전 체크포인트에서 재개할 경우
    cfg.resume = True
    
    cfg.load_from = '/data/ephemeral/home/hanseonglee/level2-objectdetection-cv-04/mmdetection3/work_dirs/fold1/best_coco_bbox_mAP_50_epoch_8.pth'

    # 데이터 로더에 파이프라인 적용
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline


    # Runner 생성
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()

