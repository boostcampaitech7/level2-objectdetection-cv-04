import os
import argparse
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from tqdm import tqdm
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, annotation, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)

        return image
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())

def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs

def main(args):
    test_dataset = CustomDataset(args.annotation, args.data_dir)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(args.annotation)

    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > args.score_threshold: 
                prediction_string += f"{label-1} {score} {box[0]} {box[1]} {box[2]} {box[3]} "
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(args.output_csv, index=None)
    print(submission.head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Faster R-CNN Inference")
    # 데이터 경로
    parser.add_argument('--annotation', type=str, default='../dataset/test.json', help='path to annotation file')
    parser.add_argument('--data_dir', type=str, default='../dataset', help='path to data directory')
    # 학습 관련 
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8, help='number of worker threads for dataloader')
    parser.add_argument('--num_classes', type=int, default=11, help='number of classes (including background)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/faster_rcnn_torchvision_checkpoints.pth', help='path to model checkpoint')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='score threshold for predictions')
    parser.add_argument('--output_csv', type=str, default='./faster_rcnn_torchvision_submission.csv', help='path to output CSV file')
    
    args = parser.parse_args()
    
    main(args)
