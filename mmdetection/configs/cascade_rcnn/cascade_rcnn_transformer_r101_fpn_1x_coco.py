_base_ = './cascade_transformer_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='SwinTransformer',
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict( type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')))