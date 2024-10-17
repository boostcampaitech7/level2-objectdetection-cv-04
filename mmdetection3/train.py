import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="DETR 모델 훈련")
    parser.add_argument('--config', default='./configs/detr/detr_r50_8xb2-150e_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/detr_r50_8xb2-150e_coco_trash', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='/data/ephemeral/home/dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=24, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=64, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
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

    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val.json')
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')

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
    
    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids
    
    # 체크포인트 훅 설정 수정
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3
    )

    # Runner 생성 및 훈련 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
