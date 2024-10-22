import argparse
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
import mmcv
import os
from tqdm import tqdm
import cv2
import os.path as osp
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 추론")
    parser.add_argument('--config', default='./configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco_default_dataset.py', help='설정 파일 경로')
    parser.add_argument('--checkpoint', default='./work_dirs/atssv1/best_coco_bbox_mAP_50_epoch_13.pth', help='체크포인트 파일 경로')
    parser.add_argument('--work-dir', default='./work_dirs/atssv1', help='작업 디렉토리')
    parser.add_argument('--data-root', default='../dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--output-dir', default='inference_results/atssv1', help='결과 저장 디렉토리')
    parser.add_argument('--score-thr', type=float, default=0.05, help='점수 임계값')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='사용할 GPU ID')
    parser.add_argument('--num-classes', type=int, default=10, help='클래스 수')
    parser.add_argument('--batch-size', type=int, default=1, help='배치 크기')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
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

    
    # 설정 수정
    cfg.work_dir = args.work_dir  # 이 줄을 추가
    cfg.load_from = args.checkpoint
    cfg.data_root = args.data_root

    # Pipeline 설정
    img_size = 1024
    backend_args = None
    train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=[(img_size,img_size), (img_size, 800)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
    test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(img_size,img_size), keep_ratio=True),
    # dict(type='MultiScaleFlipAug',
    #     scales=[(img_size, img_size), (img_size, 800)],  # List of different scales
    #     allow_flip=False,  # Whether to apply flip augmentations
    #     transforms=[
    #         dict(type='RandomFlip'),  # Randomly flip images
    #         dict(type='PackDetInputs')
    #     ]),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
    cfg.test_dataloader.dataset.pipeline = test_pipeline
    #
    # 데이터셋 설정 수정
    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = osp.join(args.data_root, 'test')
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
    cfg.test_dataloader.batch_size = args.batch_size
    ##
    # test 데이터셋 설정
    cfg.train_dataloader.dataset.test_mode = False
    cfg.val_dataloader.dataset.test_mode = False

    # 데이터셋 설정 수정
    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = args.data_root
    cfg.test_dataloader.dataset.metainfo = dict(classes=classes)

    # 평가기 설정 수정
    cfg.test_evaluator.ann_file = osp.join(args.data_root, 'test.json')
    cfg.test_evaluator.backend_args = None
    cfg.test_evaluator.format_only = False
    cfg.test_evaluator.metric = 'bbox'
    cfg.test_evaluator.type = 'CocoMetric'

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids

    # # Runner 생성
    # runner = Runner.from_cfg(cfg)

    # 모델 초기화
    model = init_detector(cfg, args.checkpoint, device=args.device)

    # 시각화 도구 초기화
    visualizer = DetLocalVisualizer()

    # 결과 저장 디렉토리
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 결과 저장을 위한 리스트
    prediction_strings = []
    file_names = []

    # # 테스트 실행
    # results = runner.test()

        # 테스트 이미지 목록 가져오기
    test_img_dir = os.path.join(args.data_root, 'test')
    test_imgs = [os.path.join(test_img_dir, img) for img in os.listdir(test_img_dir) if img.endswith(('.jpg', '.png'))]

    # 추론 실행
    for idx, img_path in enumerate(tqdm(test_imgs)):
        # 이미지에 대해 예측 수행
        result = inference_detector(model, img_path)

        # 이미지 로드
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        # Pascal VOC 형식으로 결과 변환
        prediction_string = ''
        if hasattr(result.pred_instances, 'bboxes'):
            for label, bbox, score in zip(result.pred_instances.labels, result.pred_instances.bboxes, result.pred_instances.scores):
                if score < args.score_thr:
                    continue
                x1, y1, x2, y2 = bbox.tolist()
                prediction_string += f"{label} {score:.10f} {x1:.10f} {y1:.10f} {x2:.10f} {y2:.10f} "

        prediction_strings.append(prediction_string.strip())
        file_names.append(os.path.join('test', os.path.basename(img_path)).replace("\\", "/"))

        # 결과 시각화
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(args.output_dir, f'result_{idx}.png'),
            pred_score_thr=args.score_thr
        )
        
        # 점수 임계값 표시
        img_with_text = mmcv.imconvert(img, 'rgb', 'bgr')
        cv2.putText(img_with_text, f'Score Threshold: {args.score_thr}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        mmcv.imwrite(img_with_text, os.path.join(args.output_dir, f'result_{idx}_with_threshold.png'))

     # 제출 파일 생성
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(args.output_dir, 'submission.csv'), index=None)
    print(f"제출 파일이 '{args.output_dir}/submission.csv'에 저장되었습니다.")

    print(f"추론 결과가 '{args.output_dir}' 디렉토리에 저장되었습니다.")

if __name__ == '__main__':
    main()