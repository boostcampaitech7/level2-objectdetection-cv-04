import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import mmcv
import os
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN 모델 추론")
    parser.add_argument('--config', default='configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py', help='설정 파일 경로')
    parser.add_argument('--checkpoint', default='/data/ephemeral/home/backup/mmdetection3/work_dirs/faster-rcnn_r50_fpn_2x_trash/epoch_10.pth', help='체크포인트 파일 경로')
    parser.add_argument('--work-dir', default='/data/ephemeral/home/backup/mmdetection3/work_dirs/', help='작업 디렉토리')
    parser.add_argument('--data-root', default='/data/ephemeral/home/dataset/', help='데이터셋 루트 디렉토리')
    parser.add_argument('--output-dir', default='inference_results', help='결과 저장 디렉토리')
    parser.add_argument('--score-thr', type=float, default=0.3, help='점수 임계값')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0], help='사용할 GPU ID')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()

    # 설정 파일 로드
    cfg = Config.fromfile(args.config)
    
    # 설정 수정
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    # test 데이터셋 설정
    cfg.test_dataloader.dataset.ann_file = os.path.join(args.data_root, 'test.json')
    cfg.test_dataloader.dataset.data_prefix.img = args.data_root
    cfg.test_dataloader.dataset.test_mode = True

    # GPU 설정
    cfg.gpu_ids = args.gpu_ids

    # Runner 생성
    runner = Runner.from_cfg(cfg)

    # 클래스 이름 설정
    class_names = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
                   "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    # 결과 저장 디렉토리
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 테스트 실행
    results = runner.test()

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

        # 시각화
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

        # 결과 출력 (옵션)
        print(f"이미지: {img_path}")
        if hasattr(result.pred_instances, 'bboxes'):
            for label, bbox in zip(result.pred_instances.labels, result.pred_instances.bboxes):
                print(f"클래스: {class_names[label]}, 박스: {bbox.tolist()}")
        print("---")

    print(f"추론 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")

if __name__ == '__main__':
    main()