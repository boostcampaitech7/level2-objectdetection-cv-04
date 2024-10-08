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
    parser.add_argument('--config', default='./configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py', help='config file path')
    parser.add_argument('--checkpoint', default='latest', help='checkpoint to use')
    parser.add_argument('--work-dir', default='./work_dirs/10071624_cascade_rcnn_x101_32x4d_swin_focalloss_imgscale_1024_1x_trash', help='the dir to save logs and models')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--root', default='../dataset/', help='root directory of dataset')
    return parser.parse_args()

def main():
    args = parse_args()

    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    cfg = Config.fromfile(args.config)

    # Modify dataset config
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = args.root
    cfg.data.test.ann_file = os.path.join(args.root, 'test.json')
    cfg.data.test.pipeline[1]['img_scale'] = (512,512)
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.gpu_ids = [args.gpu_id]
    cfg.work_dir = args.work_dir
    ### 수정 부분
    # cfg.model.bbox_head.num_classes = 10
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
    ))
    # cfg.runner.max_epochs = 16
    ###
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
    ### 수정
    cfg.model.test_cfg.rpn.nms_pre = 850
    # cfg.model.test_cfg.rcnn.score_thr = 0.3
    ###
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
