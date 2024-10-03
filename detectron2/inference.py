import os
import copy
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    cfg.DATASETS.TEST = ('coco_trash_test',)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.weights)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh_test
    return cfg

def register_dataset(args):
    try:
        register_coco_instances('coco_trash_test', {}, args.test_json, args.dataset_dir)
    except AssertionError:
        pass

def my_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    dataset_dict['image'] = image
    return dataset_dict

def process_predictions(predictor, test_loader):
    prediction_strings = []
    file_names = []

    for data in tqdm(test_loader):
        prediction_string = ''
        data = data[0]
        outputs = predictor(data['image'])['instances']
        
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        for target, box, score in zip(targets, boxes, scores):
            prediction_string += f"{target} {score} {box[0]} {box[1]} {box[2]} {box[3]} "
        
        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace(args.dataset_dir, ''))

    return prediction_strings, file_names

def create_submission(prediction_strings, file_names, output_dir):
    submission = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })
    submission.to_csv(os.path.join(output_dir, 'submission_det2.csv'), index=None)

def main(args):
    setup_logger()
    register_dataset(args)
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', my_mapper)
    prediction_strings, file_names = process_predictions(predictor, test_loader)
    create_submission(prediction_strings, file_names, cfg.OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster R-CNN Inference")
    parser.add_argument("--config_file", type=str, default="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", help="Config file")

    # 기본 서버 CPU 수 8개
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    # config 관련
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--weights", type=str, default="model_final.pth", help="Model weights file name")
    parser.add_argument("--roi_batch_size_per_image", type=int, default=128, help="ROI batch size per image")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--score_thresh_test", type=float, default=0.3, help="Score threshold for test")
    parser.add_argument("--test_json", type=str, default="../../dataset/test.json", help="Path to test JSON file")
    parser.add_argument("--dataset_dir", type=str, default="../../dataset/", help="Dataset directory")
    args = parser.parse_args()
    
    
    main(args)
