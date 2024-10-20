import argparse
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
import mmcv
import os
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="DETR 모델 추론")
    parser.add_argument('--config', default='/data/ephemeral/home/hanseonglee/level2-objectdetection-cv-04/mmdetection3/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py', help='설정 파일 경로')
    parser.add_argument('--checkpoint', default='/data/ephemeral/home/hanseonglee/level2-objectdetection-cv-04/mmdetection3/work_dirs/dino/epoch_12.pth', help='체크포인트 파일 경로')
    parser.add_argument('--data-root', default='/data/ephemeral/home/dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--output-dir', default='inference_results', help='결과 저장 디렉토리')
    parser.add_argument('--score-thr', type=float, default=0.1, help='점수 임계값')
    parser.add_argument('--device', default='cuda:0', help='사용할 디바이스')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()

    # 클래스 정의 
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # 설정 파일 로드 및 수정
    cfg = Config.fromfile(args.config)

    # 설정 파일 내용 출력
    print(cfg.pretty_text)

    cfg.model.bbox_head.num_classes = 10
    
    # 모델 초기화
    model = init_detector(cfg, args.checkpoint, device=args.device)

    # 시각화 도구 초기화
    visualizer = DetLocalVisualizer()

    # 결과 저장 디렉토리
    os.makedirs(args.output_dir, exist_ok=True)

    # 결과 저장을 위한 리스트
    prediction_strings = []
    file_names = []

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
        file_names.append(os.path.join('test', os.path.basename(img_path)))

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
        img_with_text = mmcv.imconvert(img
        , 'rgb', 'bgr')
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