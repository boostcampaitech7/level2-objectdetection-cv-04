_base_ = '../cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py'
# _base_ = '../cascade_rcnn/(DCN)cascade-rcnn_x101_64x4d_fpn_20e_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'cascade-rcnn_r101-dconv-c3-c5',
            'entity': 'jongseo001111-naver'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')