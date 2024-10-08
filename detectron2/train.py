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
    cfg.merge_from_file(model_zoo.get_config_file(f'{args.model}.yaml'))
    cfg.DATASETS.TEST = ('coco_trash_test',)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{args.model}.yaml')
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = args.steps
    cfg.SOLVER.GAMMA = args.gamma
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.TEST.EVAL_PERIOD = args.eval_period
    return cfg

def register_dataset():
    try:
        register_coco_instances('coco_trash_test', {}, '../dataset/test.json', '../dataset/')
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
    class_num = 10

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
        file_names.append(data['file_name'].replace('../../dataset/', ''))

    return prediction_strings, file_names

def create_submission(prediction_strings, file_names, output_dir):
    submission = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    submission.to_csv(os.path.join(output_dir, 'submission_det2.csv'), index=None)

def main(args):
    setup_logger()
    register_dataset()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', my_mapper)
    prediction_strings, file_names = process_predictions(predictor, test_loader)
    create_submission(prediction_strings, file_names, cfg.OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster R-CNN Training")
    parser.add_argument("--model", type=str, default="COCO-Detection/faster_rcnn_R_101_FPN_3x", help="Model name")
    # 기본 서버 CPU 수 8개로 설정
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")

    # Config 관련
    parser.add_argument("--ims_per_batch", type=int, default=4, help="Images per batch")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--max_iter", type=int, default=15000, help="Maximum number of iterations")
    parser.add_argument("--steps", nargs='+', type=int, default=[8000, 12000], help="Steps for learning rate decay")
    parser.add_argument("--gamma", type=float, default=0.005, help="Gamma for learning rate decay")
    parser.add_argument("--checkpoint_period", type=int, default=3000, help="Period to save checkpoints")

    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")

    parser.add_argument("--roi_batch_size_per_image", type=int, default=128, help="ROI batch size per image")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")

    parser.add_argument("--eval_period", type=int, default=3000, help="Evaluation period")
    args = parser.parse_args()
    
    main(args)
