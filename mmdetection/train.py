import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model")
    # Config 관련 argument
    parser.add_argument('--config', default='./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help='config file path')
    parser.add_argument('--work-dir', default='./work_dirs/cv2', help='the dir to save logs and models')
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

    root = '../dataset/'
    ### Validation을 위한 세팅
    cfg.workflow = [('train', 1), ('val', 1)]
    
    ###
    # Modify dataset config
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root + 'train'
    cfg.data.train.ann_file = root + 'train.json'
    cfg.data.train.pipeline[2]['img_scale'] = (128,128)

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root + 'val'
    cfg.data.val.ann_file = root + 'val.json'
    cfg.data.val.pipeline[2]['img_scale'] = (128,128)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (128,128)
    cfg.data.samples_per_gpu = args.samples_per_gpu

    cfg.seed = args.seed
    cfg.gpu_ids = args.gpu_ids
    cfg.work_dir = args.work_dir

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    # Build dataset
    datasets = [build_dataset(cfg.data.train),build_dataset(cfg.data.val)]

    # Build the detector
    model = build_detector(cfg.model)
    model.init_weights()

    # Train the model
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()
