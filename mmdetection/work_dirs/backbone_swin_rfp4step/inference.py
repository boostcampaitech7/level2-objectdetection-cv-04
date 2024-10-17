import os
import argparse
import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN Inference")
    parser.add_argument('--config', default='./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help='config file path')
    parser.add_argument('--checkpoint', default='latest', help='checkpoint to use')
    parser.add_argument('--work-dir', default='./work_dirs/faster_rcnn_r50_fpn_1x_trash', help='the dir to save logs and models')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--root', default='../dataset/', help='root directory of dataset')
    return parser.parse_args()

def main():
    args = parse_args()

    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    cfg = Config.fromfile(args.config)
    cfg.backbone = dict(
    type='SwinTransformer',
    pretrain_img_size=384,  # Adjust if required; for pretrained weights, 384 is often used
    in_channels=3,
    embed_dims=192,
    patch_size=4,
    window_size=7,
    mlp_ratio=4,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    strides=(4, 2, 2, 2),
    out_indices=(0, 1, 2, 3),
    qkv_bias=True,
    qk_scale=None,
    patch_norm=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    use_abs_pos_embed=False,
    act_cfg=dict(type='GELU'),
    norm_cfg=dict(type='LN'),
    init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')
)

    cfg.neck = dict(
    type='RFP',
    rfp_steps=4,  # Number of recursive steps, can be adjusted as needed
    aspp_out_channels=256,  # The number of output channels for ASPP layer, commonly set to 256
    rfp_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        in_channels=3,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')
    ),
    in_channels=[384, 768, 1536, 3072],
    out_channels=256,
    num_outs=5
)
    cfg.model.rpn_head.anchor_generator.ratios = [0.25, 0.35, 0.5, 1.0, 1.5, 2.0]
    cfg.runner.max_epochs = 20
    for settings in cfg.model.roi_head.bbox_head:
        settings.num_classes = 10
    # Modify dataset config
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = args.root
    cfg.data.test.ann_file = os.path.join(args.root, 'test.json')
    cfg.data.test.pipeline[1]['img_scale'] = (512,512)
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.gpu_ids = [args.gpu_id]
    cfg.work_dir = args.work_dir

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{args.checkpoint}.pth')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[args.gpu_id])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    # Process predictions
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += f"{j} {o[4]} {o[0]} {o[1]} {o[2]} {o[3]} "
        
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    # Create submission
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{args.checkpoint}.csv'), index=None)
    print(submission.head())

if __name__ == '__main__':
    main()
