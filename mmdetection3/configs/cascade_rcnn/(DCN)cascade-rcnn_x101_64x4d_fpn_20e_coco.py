_base_ = './cascade-rcnn_r50_fpn_20e_coco.py'

# vis_backends = [
#     dict(type='LocalVisBackend'),
#     dict(type='WandbVisBackend',
#          init_kwargs={
#             'project': 'cascade-rcnn_x101-32x4d',
#             'entity': 'jongseo001111-naver'
#          })
# ]
# visualizer = dict(
#     type='DetLocalVisualizer',
#     vis_backends=vis_backends,
#     name='visualizer')

