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
    parser.add_argument('--work-dir', default='./work_dirs/10071624_cascade_rcnn_x101_32x4d_swin_focalloss_imgscale_1024_1x_trash', help='the dir to save logs and models')
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
    cfg.data.train.pipeline[2]['img_scale'] = (1024,1024)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (1024,1024)

    cfg.data.samples_per_gpu = args.samples_per_gpu

    cfg.seed = args.seed
    cfg.gpu_ids = args.gpu_ids
    cfg.work_dir = args.work_dir
    ### 수정
    for settings in cfg.model.roi_head.bbox_head:
        settings.num_classes = 10
    cfg.backbone =dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_channels=[96, 192, 384, 768],
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth')),
    cfg.neck = dict(
    type='RFP',
    rfp_steps=3,  # Number of feedback steps (tuning this can enhance performance but increases computation)
    aspp_out_channels=256,  # Set according to the feature pyramid needs
    aspp_dilations=(1, 3, 6, 1),  # Dilation rates for ASPP if used
    in_channels=[256, 512, 1024, 2048],  # Based on Swin Transformer output stages
    out_channels=256,  # Desired output channels for FPN
    num_outs=5,  # Typically the number of levels in the FPN
    rfp_backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_channels=[96, 192, 384, 768],
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth')
    )
)
    # cfg.model.rpn_head.loss_cls=dict(
    #         type='FocalLoss')
    # cfg.optimizer.lr = 8e-5
    cfg.runner.max_epochs = 12
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
