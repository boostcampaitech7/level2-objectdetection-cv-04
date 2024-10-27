import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os.path as osp
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 훈련")
    # /data/ephemeral/home/Hongjoo/level2-objectdetection-cv-04/mmdetection3/configs/centernet/centernet-update_r101_fpn_8xb8-amp-lsj-200e_coco.py
    parser.add_argument('--config', default='/data/ephemeral/home/hanseonglee/level2-objectdetection-cv-04/mmdetection3/configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/centernet_r50_epoch25_mosaic', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='/data/ephemeral/home/dataset/', help='데                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=25, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=4, help='배치 크기')# 64
    parser.add_argument('--lr', type=float, default=0.0001, help='학습률')
    parser.add_argument('--num-classes', type=int, default=10, help='클래스 수')
    parser.add_argument('--seed', type=int, default=2022, help='랜덤 시드')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='사용할 GPU ID')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()

    # 설정 파일 로드
    cfg = Config.fromfile(args.config)

    # 클래스 정의
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # 모델 설정 수정
    cfg.model.bbox_head.num_classes = args.num_classes
    
    cfg.data_root = args.data_root
    
    # 데이터셋 설정 수정
    # cfg.train_dataloader =dict(
    #     batch_size=args.batch_size,
    #     num_workers=2,
    #     persistent_workers=True,
    #     sampler=dict(type='DefaultSampler', shuffle=True),
    #     batch_sampler=dict(type='AspectRatioBatchSampler'),
    #     dataset=dict(
    #         type=cfg.dataset_type,
    #         data_root=args.data_root,
    #         ann_file= osp.join(args.data_root, 'train.json'),
    #         data_prefix=dict(img=osp.join(args.data_root, 'train')),
    #         filter_cfg=dict(filter_empty_gt=True, min_size=32),
    #         pipeline= cfg.train_pipeline,
    #         backend_args=cfg.backend_args))
    
    


    # cfg.dataset_wrapper = dict(type='MultiImageMixDataset'), # mosaic 사용
    # cfg.train_dataloader.dataset.ann_file = osp.join(args.data_root, 'train.json')
    # cfg.train_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'train')
    # cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
    # cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_cfg.max_epochs = args.epochs


    
    cfg.val_dataloader.dataset.ann_file = osp.join(args.data_root, 'val.json')
    cfg.val_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'val/')
    cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.val_dataloader.batch_size = args.batch_size
    cfg.val_dataloader.dataset.data_root = args.data_root
    cfg.val_dataloader.dataset.data_root = args.data_root

    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'test/')
    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.batch_size = args.batch_size
#       

    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val.json')
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')
    cfg.val_evaluator.classwise = True

    # cfg.optim_wrapper.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    # cfg.optim_wrapper.type='OptimWrapper'

    #cfg.param_scheduler[0] = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
    #cfg.param_scheduler[1] = dict(type='MultiStepLR', by_epoch=True, milestones=[2], gamma=0.1)

    # cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 1

    # cfg.default_hooks.timer = dict(type='IterTimerHook')
    # cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    # cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')

    # 기본 훈련 설정의 hook 변경 가능
    # 시각화
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            interval=1,
            # save_best='auto',
            max_keep_ckpts=3,
            save_best='coco/bbox_mAP_50',  # Track the best bbox_mAP
            rule='greater',
            greater_keys=['coco/bbox_mAP_50'],  # mAP는 높을수록 좋다
            ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook')
    )
    # Wandb 시각화
    cfg.vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs={
                'project': 'centernet_r50_epoch25_mosaic',
                'entity': 'jongseo001111-naver'
            })
    ]
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=cfg.vis_backends,
        name='visualizer'
    )
    # cfg.img_size = cfg.img_size # (1024,1024)
    # augmentation
    train_pipeline = [
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        # dict(type='RandomFlip', prob=0.5),
        # dict(type='PackDetInputs'),
        dict(type='Mosaic', 
            img_scale=(1024,1024),
            pad_val=114.0 ,
            prob = 0.3,
            bbox_clip_border=True,
            center_ratio_range=(
                0.5,
                1.5,
            ),
        ),
        # dict(
        # type='MixUp',
        # img_scale=img_scale,
        # ratio_range=(0.8, 1.6),
        # pad_val=114.0),
        dict(
            keep_ratio=True,
            ratio_range=(0.1,2.0),
            scale=(1024,1024),
            type='RandomResize'),
        dict(
            allow_negative_crop=True,
            crop_size=(1024,1024,),
            crop_type='absolute_range',
            recompute_bbox=True,
            type='RandomCrop'),
        dict(min_gt_bbox_wh=(0.01,0.01,), type='FilterAnnotations'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PackDetInputs'),
    ]
    
    train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
        type='MultiImageMixDataset',
        
        dataset=dict(
            type=cfg.dataset_type,
            data_root=args.data_root,
            ann_file= osp.join(args.data_root,'train.json'),
            data_prefix= dict( img=osp.join(args.data_root, 'train/') ),
            metainfo = dict(classes=classes) ,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=cfg.backend_args),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
        ),
        
        pipeline=train_pipeline ,
        
    )   
        #이전 dataloader
    #  cfg.train_dataloader = dict(
    #     batch_size=args.batch_size,
    #     num_workers=2,
    #     persistent_workers=True,
    #     sampler=dict(type='DefaultSampler', shuffle=True),
    #     batch_sampler=dict(type='AspectRatioBatchSampler'),
    #     dataset=dict(
    #         type=cfg.dataset_type,
    #         data_root=args.data_root,
    #         ann_file = osp.join(args.data_root, 'train.json'),
    #         data_prefix=dict(img=osp.join(args.data_root, 'train/')),
    #         pipeline=cfg.train_pipeline,
    #         metainfo = dict(classes=classes),
    #         backend_args=cfg.backend_args
    #     )
    # )
        # pipeline , dynamic_scale , skip_type_keys max_refetch
    

    cfg.train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_sampler=dict(type='AspectRatioBatchSampler'),
        dataset = train_dataset,
    )
        # dataset=dict(
        #     type=cfg.dataset_type,
        #     data_root=args.data_root,
        #     ann_file = osp.join(args.data_root, 'train.json'),
        #     data_prefix=dict(img=osp.join(args.data_root, 'train/')),
        #     pipeline=cfg.train_pipeline,
        #     metainfo = dict(classes=classes),
        #     backend_args=cfg.backend_args
        #     )
        
    #     # learning rate
    # param_scheduler = [
    #     dict(
    #         # use quadratic formula to warm up 5 epochs
    #         # and lr is updated by iteration
    #         # TODO: fix default scope in get function
    #         type='mmdet.QuadraticWarmupLR',
    #         by_epoch=True,
    #         begin=0,
    #         end=5,
    #         convert_to_iter_based=True),
    #     dict(
    #         # use cosine lr from 5 to 285 epoch
    #         type='CosineAnnealingLR',
    #         eta_min=base_lr * 0.05,
    #         begin=5,
    #         T_max=max_epochs - num_last_epochs,
    #         end=max_epochs - num_last_epochs,
    #         by_epoch=True,
    #         convert_to_iter_based=True),
    #     dict(
    #         # use fixed lr during last 15 epochs
    #         type='ConstantLR',
    #         by_epoch=True,
    #         factor=1,
    #         begin=max_epochs - num_last_epochs,
    #         end=max_epochs,
    #     )
    # ]


    # val 마다 best model 3개 체크포인트 저장
    # cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3)
    # cfg.val_evaluator.classwise = True
    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids
    #batchsize
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=args.batch_size)
    #optimizer
    cfg.optim_wrapper = dict(
        type = 'OptimWrapper',
        optimizer=dict(type='SGD', lr=args.lr, momentum=0.9, weight_decay=0.0001 , nesterov=True), # 0.02
        # Experiments show that there is no need to turn on clip_grad.
        # paramwise_cfg=dict(norm_decay_mult=0.))
        clip_grad=dict(max_norm=35, norm_type=2),
        )
    # cfg.optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
    # cfg.optim_wrapper.accumulative_counts = 2  # 2번의 반복마다 그래디언트 업데이트
    # real_batch effect =  {accumulative_counts} * {batch_size}
    # Runner 생성 및 훈련 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()

    