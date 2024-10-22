import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 훈련")
    parser.add_argument('--config', default='./configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco_default_dataset.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/atss_kFold', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='../dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=20, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=2, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.00005, help='학습률')
    parser.add_argument('--num-classes', type=int, default=10, help='클래스 수')
    parser.add_argument('--seed', type=int, default=2022, help='랜덤 시드')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='사용할 GPU ID')
    parser.add_argument('--pretrained', default='./pretrained/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth', help='Pretrained 모델 경로 (Optional)')
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
    cfg.model.bbox_head.num_classes = args.num_classes


    ## 임시
    if args.pretrained is not None:
        cfg.load_from = args.pretrained
        # cfg.model.init_cfg = dict(type='Pretrained', checkpoint=args.pretrained, prefix='backbone', strict=False)
    ##

    # 데이터셋 설정 수정
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.val_dataloader.batch_size = args.batch_size

    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'test')
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.batch_size = args.batch_size


    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val.json')
    cfg.val_evaluator.classwise = True
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')

    cfg.param_scheduler[0] = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
    cfg.param_scheduler[1] = dict(
        type='CosineAnnealingLR',
        T_max=20,  # Specify the epochs at which to decrease the learning rate
        eta_min=0.0000002,               # Factor by which the learning rate will be reduced
        begin=1,                 # Start iteration
        end=20,                  # End iteration (adjust as needed)
        by_epoch=True            # Whether to apply this by epoch
    )

    cfg.default_hooks.timer = dict(type='IterTimerHook')
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')

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

    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids
    # Runner 생성 및 훈련 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()