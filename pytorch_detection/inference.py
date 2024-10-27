import argparse
from src.inference import run_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Faster R-CNN Inference")
    parser.add_argument('--annotation', type=str, default='../dataset/test.json', help='path to annotation file')
    parser.add_argument('--data_dir', type=str, default='../dataset', help='path to data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8, help='number of worker threads for dataloader')
    parser.add_argument('--num_classes', type=int, default=11, help='number of classes (including background)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/faster_rcnn_torchvision_checkpoints.pth', help='path to model checkpoint')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='score threshold for predictions')
    parser.add_argument('--output_csv', type=str, default='./faster_rcnn_torchvision_submission.csv', help='path to output CSV file')
    
    args = parser.parse_args()
    
    run_inference(args)