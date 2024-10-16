import argparse
import wandb
import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device


#파일 경로 지정
config_name = 'retinanet_x101_64x4d_fpn_1x_coco.py'
workdir_name = 'workdir_' + config_name

#Train 파일 돌 때 새로운 dir 파일 생성
os.makedirs(workdir_name, exist_ok = True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model")
    # Config 관련 argument
    parser.add_argument('--config', default='/data/ephemeral/home/level2-objectdetection-cv-04/mmdetection/configs/retinanet/' + config_name, help='config file path')
    parser.add_argument('--work-dir', default='/data/ephemeral/home/level2-objectdetection-cv-04/mmdetection/' + workdir_name, help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='ids of gpus to use')
    parser.add_argument('--samples-per-gpu', type=int, default=4, help='samples per gpu')
    return parser.parse_args()

def main():
    args = parse_args()
    # 10 classes
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    cfg = Config.fromfile(args.config)

    root = '/data/ephemeral/home/dataset/'

    img_size = 512

    # Modify dataset config
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json'
    cfg.data.train.pipeline[2]['img_scale'] = (img_size,img_size)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (img_size,img_size)

    cfg.data.samples_per_gpu = args.samples_per_gpu

    cfg.seed = args.seed
    cfg.gpu_ids = args.gpu_ids
    cfg.work_dir = args.work_dir

    # 2stage 모델에서 사용
    # cfg.model.roi_head.bbox_head.num_classes = 10
    
    # 1stage 모델에서 사용
    cfg.model.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)
    model.init_weights()

    # Train the model
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)

if __name__ == '__main__':
    main()
