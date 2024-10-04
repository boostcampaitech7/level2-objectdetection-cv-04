import torch
import torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
import pandas as pd
from src.utils import CustomDataset

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