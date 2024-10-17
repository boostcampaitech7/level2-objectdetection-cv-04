import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model")
    # Config 관련 argument
    parser.add_argument('--config', default='./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help='config file path')
    parser.add_argument('--work-dir', default='./work_dirs/faster_rcnn_r50_fpn_1x_trash', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='ids of gpus to use')
    parser.add_argument('--samples-per-gpu', type=int, default=4, help='samples per gpu')
    return parser.parse_args()

def main():
    args = parse_args()
    # 10 classes
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    cfg = Config.fromfile(args.config)

    root = '../dataset/'
    cfg.data_root = root
    cfg.model.bbox_head.num_classes = len(classes)
    
    # Modify dataset config
    cfg.data.train.classes = classes
    # cfg.data.train.img_prefix = root + 'train/' # edit
    # cfg.data.train.ann_file = root + 'train.json'
    
    cfg.data.sample_per_gpu = 128#16 # = real batch size
    cfg.data.worker_per_gpu = 4 # 
    cfg.runner.max_epochs = 28 # real epoch = max_epochs * times
    cfg.auto_scale_lr.base_batch_size = 128 #  (8 GPUs) x (16 samples per GPU)
    
    # cfg.data.train.pipeline[2]['img_scale'] = (512,512)
    cfg.data.train.times = 1 # 5 : train reapeat
    cfg.data.train.dataset= dict(
            type=cfg.dataset_type ,
            classes = classes,
            ann_file =  root + 'train.json',
            img_prefix = root , 
            pipeline = cfg.train_pipeline)
    # cfg.data.train.dataset.ann_file = root + 'train.json'
    
    # cfg.data.train.dataset.img_prefix=  root + 'train/'

    # cfg.data.val.img_prefix = root + 'train/'
    # cfg.data.val.ann_file = root + 'train.json' #edit

    # cfg.data.test.classes = classes
    # cfg.data.test.img_prefix = root+ 'test/'
    # cfg.data.test.ann_file = root + 'test.json'
    # cfg.data.test.pipeline[1]['img_scale'] = (512,512)
    # print(f" data -train-dataset : { cfg.data.train.dataset.ann_file} , {cfg.data.train.dataset.img_prefix}")

    cfg.data.samples_per_gpu = args.samples_per_gpu

    cfg.seed = args.seed
    cfg.gpu_ids = args.gpu_ids
    cfg.work_dir = args.work_dir
    
    # cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()


    #Wandb
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='centernet_mmdetectionV2_try1', # Set Project name 
                entity='jongseo001111-naver', 
                config=dict(
                    lr = cfg.auto_scale_lr,#  0.01,
                    batch_size =128 ,
                    num_epochs =140,#140,##base_batch_size,#28*5,#140 = workers_per_gpu * sampe
                    backbone ='ResNet', 
                    depth = 18
                )
            )
        )       
    ]

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    print(f"Dataset length: {len(datasets[0])} , {type(datasets)}")
    # logger.info(f"Dataset length: {len(datasets[0])} , {datasets[0][0]}")

    # logger.info(f" data-train-dataset : {cfg.data.train.dataset.ann_file},{cfg.data.train.dataset.img_prefix}")

    # Build the detector
    model = build_detector(cfg.model)
    
    model.init_weights()

    # Train the model
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)

if __name__ == '__main__':
    main()
