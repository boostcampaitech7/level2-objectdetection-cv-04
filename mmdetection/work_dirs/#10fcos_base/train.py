import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
from mmcv.runner import Hook, HOOKS
import os
import shutil

def print_last_non_empty_line(filename):
    with open(filename, 'rb') as file:  # 바이너리 모드로 열기
        file.seek(0, 2)  # 파일 끝으로 이동
        position = file.tell()  # 현재 위치 저장
        
        # 마지막 줄을 찾기 위해 거꾸로 탐색
        last_line = None
        while position >= 0:
            file.seek(position)
            if file.read(1) == b'\n' and position != file.tell():
                # 줄바꿈을 찾았을 때
                if last_line:  # 마지막 줄이 이미 설정되어 있을 경우
                    break
            else:
                # 마지막 줄 업데이트
                file.seek(position)
                last_line = file.readline().decode('utf-8').strip()
            
            position -= 1
        
        if last_line:  # 빈 줄이 아닌 마지막 줄이 있는 경우에만 출력
            return last_line.split()


@HOOKS.register_module()
class BestCheckpointHook(Hook):
    '''Validation mAP50 기준으로 모델을 저장합니다
    parameters
    max_keep_ckpts: mAP50을 기준으로 몇 개의 모델을 저장할 지 결정 (Default: 3)
    '''
    def __init__(self, max_keep_ckpts = 3):
        self.best_score = None
        self.max_keep_ckpts = max_keep_ckpts
        self.stack = []

    def after_train_epoch(self, runner):
        print('Evaluating whether to save the model')
        current_score = float(print_last_non_empty_line(os.path.join(runner.work_dir,'None.log.json'))[11].strip(','))
        if len(self.stack) < self.max_keep_ckpts:
            self.stack.append((current_score, runner.epoch+1))
            self.stack.sort()
            self._save_checkpoint(runner, current_score)
        if len(self.stack) >= self.max_keep_ckpts and current_score > self.stack[0][0]:
        # 가장 낮은 점수의 체크포인트 찾기
            lowest_score, epoch = self.stack.pop(0)
            lowest_checkpoint_path = os.path.join(runner.work_dir, f'epoch_{epoch}_mAP_{lowest_score}.pth')
            if os.path.exists(lowest_checkpoint_path):
                os.remove(lowest_checkpoint_path)  # 파일 삭제
            self.stack.append((current_score, runner.epoch+1))
            self.stack.sort()
            self._save_checkpoint(runner, current_score)
        if len(self.stack) and self.stack[-1][0] == current_score:
            self._save_bestmodel(runner, current_score)
        print('Evaluation of saving model done!')
            
    def _save_checkpoint(self, runner,current_score):
        # 체크포인트 저장
        checkpoint_path = os.path.join(runner.work_dir, f'epoch_{runner.epoch + 1}_mAP_{current_score}.pth')
        runner.save_checkpoint(runner.work_dir, checkpoint_path)
        print(f'Epoch {runner.epoch + 1} model has been saved.')
        
    
    def _save_bestmodel(self, runner, current_score):
        best_model_path = os.path.join(runner.work_dir, 'best_model.pth')
        runner.save_checkpoint(runner.work_dir, best_model_path)
        print(f'Current Best Model: Epoch {runner.epoch + 1} model, mAP: {current_score}')



def parse_args():
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model")
    # Config 관련 argument
    parser.add_argument('--config', default='./configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py', help='config file path')
    parser.add_argument('--work-dir', default='./work_dirs/fcos_r50_base', help='the dir to save logs and models')
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

    # CustomHook 추가
    cfg.custom_hooks = [
        dict(
            type='BestCheckpointHook',
            max_keep_ckpts = 3,
            priority='VERY_LOW'
        )
    ]

    root = '../dataset/'
    # Modify dataset config
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root + 'train'
    cfg.data.train.ann_file = root + 'train.json'
    cfg.data.train.pipeline[2]['img_scale'] = (256,256)

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root + 'val'
    cfg.data.val.ann_file = root + 'val.json'
    cfg.data.val.pipeline[1]['img_scale'] = (256,256)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (256,256)
    cfg.data.samples_per_gpu = args.samples_per_gpu

    cfg.seed = args.seed
    cfg.gpu_ids = args.gpu_ids
    cfg.work_dir = args.work_dir

    # cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)
    cfg.device = get_device()

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]


    # Build the detector
    model = build_detector(cfg.model)
    model.init_weights()
    # Train the model
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()
