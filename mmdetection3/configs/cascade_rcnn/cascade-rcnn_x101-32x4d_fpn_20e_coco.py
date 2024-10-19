_base_ = './cascade-rcnn_r50_fpn_20e_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'cascade-rcnn_x101-32x4d',
            'group': 'faster-rcnn_r50-caffe_fpn_1x_coco',
            'entity': 'jongseo001111-naver'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')