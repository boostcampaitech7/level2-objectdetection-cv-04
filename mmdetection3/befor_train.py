import os
import json
import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model")
    # Config 관련 argument
    parser.add_argument('--config', default='./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', help='config file path')
    parser.add_argument('--work-dir', default='./work_dirs/faster-rcnn_r50_fpn_1x_trash', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='ids of gpus to use')
    parser.add_argument('--samples-per-gpu', type=int, default=4, help='samples per gpu')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)

    # work_dir 설정
    cfg.work_dir = args.work_dir

    # 데이터셋 설정 수정
    cfg.data_root = '/data/ephemeral/home/dataset/'
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.ann_file = 'train.json'
    cfg.train_dataloader.dataset.data_prefix = dict(img='train/')

    # 검증 데이터셋을 사용하지 않을 경우
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.val_cfg = None

    # 디버깅 정보 출력
    print(f"Train annotation file: {os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file)}")
    print(f"Data root: {cfg.data_root}")
    print(f"Ann file: {cfg.train_dataloader.dataset.ann_file}")
    print(f"Data prefix: {cfg.train_dataloader.dataset.data_prefix}")
    print(f"Work directory: {cfg.work_dir}")
    
    # 주석 파일 내용 확인
    with open(os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file), 'r') as f:
        train_data = json.load(f)
    print(f"Train data keys: {train_data.keys()}")
    print(f"Number of train images: {len(train_data.get('images', []))}")
    print(f"Number of train annotations: {len(train_data.get('annotations', []))}")

    # 모델 설정 수정
    cfg.model.pop('num_classes', None)  # 최상위 레벨에서 num_classes 제거
    cfg.model.roi_head.bbox_head.num_classes = 10  # bbox_head에 num_classes 추가

    # optimizer 설정 수정
    if 'clip_grad' in cfg.optim_wrapper:
        clip_grad = cfg.optim_wrapper.pop('clip_grad')
        cfg.optimizer_config = dict(grad_clip=clip_grad)
    cfg.optim_wrapper.optimizer.update(dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)

    # Runner 생성 및 실행
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()