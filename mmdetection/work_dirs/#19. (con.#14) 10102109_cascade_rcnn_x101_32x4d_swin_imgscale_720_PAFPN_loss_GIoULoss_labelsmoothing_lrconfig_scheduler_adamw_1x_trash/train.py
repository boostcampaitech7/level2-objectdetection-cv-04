import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Train a atss_r50_fpn model")
    # Config 관련 argument
    parser.add_argument('--config', default='./configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py', help='config file path')
    parser.add_argument('--work-dir', default='./work_dirs/#19. (con.#14) 10102109_cascade_rcnn_x101_32x4d_swin_imgscale_720_PAFPN_loss_GIoULoss_labelsmoothing_lrconfig_scheduler_adamw_1x_trash', help='the dir to save logs and models')
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

    # Modify dataset config
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json'
    cfg.data.train.pipeline[2]['img_scale'] = (720,720)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (720,720)

    args.samples_per_gpu = 16
    cfg.data.samples_per_gpu = args.samples_per_gpu
    cfg.optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
    cfg.seed = args.seed
    cfg.gpu_ids = args.gpu_ids
    cfg.work_dir = args.work_dir
    ### 수정
    for settings in cfg.model.roi_head.bbox_head:
        settings.num_classes = 10
    cfg.backbone = dict(
    type='SwinTransformer',
    embed_dims=192,  
    depths=[2, 2, 18, 2],  
    num_heads=[4, 8, 16, 32],  # 헤드 수 조정
    window_size=7,
    mlp_ratio=4.,
    out_indices=(0, 1, 2, 3),
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3, 
    ape=False,
    patch_norm=True,
    out_channels=[384, 768, 1536, 3072],
    init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')),
    cfg.neck=dict(
        type='PAFPN',
        in_channels=[384, 768, 1536, 3072],
        out_channels=256,
        num_outs=5)
    # cfg.model.rpn_head.anchor_generator.ratios = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    # cfg.model.rpn_head.loss_cls=dict(
    #         type='FocalLoss')
    cfg.model.rpn_head.loss_cls=dict(
            type='LabelSmoothingCrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, label_smoothing = 0.1)
    cfg.model.rpn_head.loss_bbox=dict(type='DIoULoss')
    cfg.lr_config.step = [12, 16]
    cfg.lr_config.gamma = 0.1
    # cfg.optimizer.lr = 2e-2
    cfg.runner.max_epochs = 20
    ### 수정
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
