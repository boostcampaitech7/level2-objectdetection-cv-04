import argparse
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import os.path as osp
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import os
import re
from mmengine.hooks import CheckpointHook
@HOOKS.register_module()
class CustomCheckpointHook(CheckpointHook):
    def after_train_epoch(self, runner):
        # Get the current epoch and bbox_mAP score
        # print('Whole Log---------------',runner.message_hub.runtime_info)
        # epoch = runner.message_hub.get_info('epoch') + 1  # +1 because epoch is 0-indexed
        # bbox_mAP = runner.message_hub.get_info('coco/bbox_mAP_50')
        # print('epoch',runner.message_hub.get_info('epoch'))
        # print('bbox_mAP50',runner.message_hub.get_info('coco/bbox_mAP_50'))
        # # Customize filename template
        # self.filename_tmpl = f'epoch_{epoch}_mAP_{bbox_mAP:.4f}.pth'
        # self.max_keep_ckpts=3
        # Call parent method to save the checkpoint
        super().after_train_epoch(runner)

# def print_last_non_empty_line(filename):
#     with open(filename, 'rb') as file:  # 바이너리 모드로 열기
#         file.seek(0, 2)  # 파일 끝으로 이동
#         position = file.tell()  # 현재 위치 저장
        
#         # 마지막 줄을 찾기 위해 거꾸로 탐색
#         last_line = None
#         while position >= 0:
#             file.seek(position)
#             if file.read(1) == b'\n' and position != file.tell():
#                 # 줄바꿈을 찾았을 때
#                 if last_line:  # 마지막 줄이 이미 설정되어 있을 경우
#                     break
#             else:
#                 # 마지막 줄 업데이트
#                 file.seek(position)
#                 last_line = file.readline().decode('utf-8').strip()
            
#             position -= 1
        
#         if last_line:  # 빈 줄이 아닌 마지막 줄이 있는 경우에만 출력
#             print(last_line.split())
#             return last_line.split()


# @HOOKS.register_module()
# class BestCheckpointHook(Hook):
#     '''Validation mAP50 기준으로 모델을 저장합니다
#     parameters
#     max_keep_ckpts: mAP50을 기준으로 몇 개의 모델을 저장할 지 결정 (Default: 3)
#     '''
#     def __init__(self, max_keep_ckpts = 3, file_name = None):
#         self.best_score = None
#         self.max_keep_ckpts = max_keep_ckpts
#         self.stack = []
#         self.file_name = file_name

#     def after_train_epoch(self, runner):
#         print('Evaluating whether to save the model')
#         log_dir = os.path.join(runner.work_dir, self.file_name)
#         current_score = float(print_last_non_empty_line(os.path.join(log_dir, f'{self.file_name}.log'))[12])
#         if len(self.stack) < self.max_keep_ckpts:
#             self.stack.append((current_score, runner.epoch+1))
#             self.stack.sort()
#             self._save_checkpoint(runner, current_score)
#         if len(self.stack) >= self.max_keep_ckpts and current_score > self.stack[0][0]:
#         # 가장 낮은 점수의 체크포인트 찾기
#             lowest_score, epoch = self.stack.pop(0)
#             lowest_checkpoint_path = os.path.join(runner.work_dir, f'epoch_{epoch}_mAP_{lowest_score}.pth')
#             if os.path.exists(lowest_checkpoint_path):
#                 os.remove(lowest_checkpoint_path)  # 파일 삭제
#             self.stack.append((current_score, runner.epoch+1))
#             self.stack.sort()
#             self._save_checkpoint(runner, current_score)
#         if len(self.stack) and self.stack[-1][0] == current_score:
#             self._save_bestmodel(runner, current_score)
#         print('Evaluation of saving model done!')
            
#     def _save_checkpoint(self, runner,current_score):
#         # 체크포인트 저장
#         checkpoint_path = os.path.join(runner.work_dir, f'epoch_{runner.epoch + 1}_mAP_{current_score}.pth')
#         runner.save_checkpoint(runner.work_dir, checkpoint_path)
#         print(f'Epoch {runner.epoch + 1} model has been saved.')
        
    
#     def _save_bestmodel(self, runner, current_score):
#         best_model_path = os.path.join(runner.work_dir, 'best_model.pth')
#         runner.save_checkpoint(runner.work_dir, best_model_path)
#         print(f'Current Best Model: Epoch {runner.epoch + 1} model, mAP: {current_score}')
def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 훈련")
    parser.add_argument('--config', default='./configs/cascade_rcnn/cascade-rcnn_x101-32x4d_fpn_20e_coco.py', help='설정 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/v3', help='로그와 모델을 저장할 디렉토리')
    parser.add_argument('--data-root', default='../dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--epochs', type=int, default=18, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
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
    for settings in cfg.model.roi_head.bbox_head:
        settings.num_classes = args.num_classes


    ## 임시
    cfg.rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0))
    ##

    # Pipeline 설정
    img_size = 256
    backend_args = None
    train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(img_size,img_size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
    test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(img_size,img_size), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
    cfg.train_dataloader.dataset.pipeline = train_pipeline
    cfg.test_dataloader.dataset.pipeline = test_pipeline
    #
    # 데이터셋 설정 수정
    cfg.train_dataloader.dataset.ann_file = osp.join(args.data_root, 'train.json')
    cfg.train_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'train')
    cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.train_dataloader.batch_size = args.batch_size

    cfg.val_dataloader.dataset.ann_file = osp.join(args.data_root, 'val.json')
    cfg.val_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'val')
    cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.val_dataloader.batch_size = args.batch_size

    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'test')
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.batch_size = args.batch_size

    # 평가기 설정 수정
    cfg.val_evaluator.ann_file = osp.join(args.data_root, 'val.json')
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')

    cfg.optim_wrapper.optimizer = dict(type='AdamW', 
                         lr=args.lr, 
                         weight_decay=0.0001)
    cfg.optim_wrapper.type='OptimWrapper'

    cfg.param_scheduler[0] = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
    cfg.param_scheduler[1] = dict(type='MultiStepLR', by_epoch=True, milestones=[11,15], gamma=0.1)

    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 1

    cfg.default_hooks.timer = dict(type='IterTimerHook')
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=50)
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')
###
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
    rfp_steps=3,  # Number of recursive steps, can be adjusted as needed
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
###

    # 기본 훈련 설정의 hook 변경 가능
    cfg.default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CustomCheckpointHook',  # Use custom checkpoint hook
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50',  # Track the best bbox_mAP
        rule='greater',
        greater_keys=['coco/bbox_mAP_50'],  # mAP는 높을수록 좋다
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

    # 옵티마이저 설정 수정
    cfg.optim_wrapper.optimizer.lr = args.lr

    # 훈련 설정 수정
    cfg.train_cfg.max_epochs = args.epochs

    # 작업 디렉토리 설정
    cfg.work_dir = args.work_dir

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids
    # numeric_dirs = [d for d in os.listdir(cfg.work_dir) if re.match(r'^\d', d)]

# 필터된 항목을 역순으로 정렬하고 마지막 항목 선택
    # latest_dir = sorted(numeric_dirs, reverse=True)[0] if numeric_dirs else None
    # CustomHook 추가
    # cfg.custom_hooks = [
    #     dict(type='CustomCheckpointHook')
    # ]

    # Runner 생성 및 훈련 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()