_base_ = './dino-5scale_swin-l_8xb2-12e_coco.py'
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[6, 10],
        gamma=0.1)
]



load_from = '/data/ephemeral/home/hanseonglee/level2-objectdetection-cv-04/mmdetection3/dino-5scale_swin-l_8xb2-36e_coco_pretrained.pth'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'dino_with_k_fold1',
            'entity': 'jongseo001111-naver'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')