import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import mmcv
import os
from tqdm import tqdm
import torch
import os.path as osp
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="ceneternet2모델 추론")
    parser.add_argument('--config', default='./configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py', help='설정 파일 경로')
    parser.add_argument('--checkpoint', default='/data/ephemeral/home/Hongjoo/level2-objectdetection-cv-04/mmdetection3/work_dirs/centernet-update_r101_fpn_8xb8-amp-lsj-200e_coco_t1/epoch_1.pth', help='체크포인트 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/t1_inference', help='작업 디렉토리')
    parser.add_argument('--data-root', default='/data/ephemeral/home/dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--output-dir', default='inference_results/test1', help='결과 저장 디렉토리')
    parser.add_argument('--score-thr', type=float, default=0.05, help='점수 임계값')
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
    cfg.model.bbox_head.num_classes = 10
    
    # 설정 수정
    cfg.data_root = args.data_root
    cfg.work_dir = args.work_dir  # 이 줄을 추가
    cfg.load_from = args.checkpoint
    cfg.data_root = args.data_root
    
    # test 데이터셋 설정
    cfg.train_dataloader.dataset.test_mode = False
    cfg.val_dataloader.dataset.test_mode = False

    # 데이터셋 설정 수정
    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = args.data_root
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.dataset.data_root = args.data_root
    
    # 평가기 설정 수정
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_evaluator.backend_args = None
    cfg.test_evaluator.format_only = False
    cfg.test_evaluator.metric = 'bbox'
    cfg.test_evaluator.type = 'CocoMetric'

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids

    # Runner 생성
    runner = Runner.from_cfg(cfg)

    # 결과 저장 디렉토리
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 결과 저장을 위한 리스트
    prediction_strings = []
    file_names = []

    # 테스트 실행
    # results = runner.test() # 이거 주석처리 해야함

    # 결과 처리 및 시각화
    for idx, data_sample in enumerate(tqdm(runner.test_dataloader)):
        # 이미지 파일 경로
        img_path = data_sample['data_samples'][0].img_path

        # 이미지 로드
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        # 추론 수행
        with torch.no_grad():
            result = runner.model.test_step(data_sample)[0]

        # Pascal VOC 형식으로 결과 변환
        prediction_string = ''
        if hasattr(result.pred_instances, 'bboxes'):
            for label, bbox, score in zip(result.pred_instances.labels, result.pred_instances.bboxes, result.pred_instances.scores):
                if score < args.score_thr:
                    continue
                x1, y1, x2, y2 = bbox.tolist()
                prediction_string += f"{label} {score:.10f} {x1:.10f} {y1:.10f} {x2:.10f} {y2:.10f} "

        prediction_strings.append(prediction_string.strip())
        file_names.append((os.path.join('test', os.path.basename(img_path))))

        # 시각화 (옵션)
        runner.visualizer.add_datasample(
            f'result_{idx}',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, f'result_{idx}.png'),
            pred_score_thr=args.score_thr
        )

    # 제출 파일 생성
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(args.work_dir, 'submission.csv'), index=None)
    print(f"제출 파일이 '{args.work_dir}/submission.csv'에 저장되었습니다.")

    print(f"추론 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")

if __name__ == '__main__':
    main()