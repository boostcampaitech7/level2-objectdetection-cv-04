from mmcv.runner import HOOKS
from mmcv.runner.hooks import WandbLoggerHook
import wandb
import numpy as np
from sklearn.metrics import precision_recall_curve
 
@HOOKS.register_module()
class CustomWandbLoggerHook(WandbLoggerHook):
    def __init__(self, *args, **kwargs):
        super(CustomWandbLoggerHook, self).__init__(*args, **kwargs)
        self.predictions = []
        self.ground_truths = []

    def after_val_iter(self, runner):
        # Assuming runner.outputs contains bbox predictions in the format [x1, y1, x2, y2, score]
        if 'bbox_results' in runner.outputs and 'gt_bboxes' in runner.data_batch:
            bbox_results = runner.outputs['bbox_results']
            gt_bboxes = runner.data_batch['gt_bboxes']

            # Collect predictions and ground truth for later processing
            self.predictions.extend(bbox_results)
            self.ground_truths.extend(gt_bboxes)

    def after_val_epoch(self, runner):
        # Call the original WandbLoggerHook after_val_epoch to log default metrics
        super().after_val_epoch(runner)

        # Compute and log the PR curve
        if self.predictions and self.ground_truths:
            precision, recall = self.compute_precision_recall(self.predictions, self.ground_truths)

            if len(precision) > 0 and len(recall) > 0:
                # Use W&B to log the PR curve
                wandb.log({"PR Curve": wandb.plot.line_series(
                    xs=[r for r in recall],
                    ys=[p for p in precision],
                    keys=["Precision"],
                    title="Precision-Recall Curve",
                    xname="Recall"
                )})

    def compute_precision_recall(self, predictions, ground_truths, iou_threshold=0.5):
        # Initialize true positives (TP), false positives (FP), and false negatives (FN)
        tp, fp, fn = 0, 0, 0

        # Flatten the lists of predictions and ground truths
        for img_preds, img_gts in zip(predictions, ground_truths):
            matched_gt = set()
            for pred in img_preds:
                iou_max = 0
                best_gt_idx = -1

                # Compute IoU between prediction and each ground truth box
                for idx, gt in enumerate(img_gts):
                    iou = self.compute_iou(pred[:4], gt)
                    if iou > iou_max:
                        iou_max = iou
                        best_gt_idx = idx

                if iou_max >= iou_threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            # Count false negatives for unmatched ground truths
            fn += len(img_gts) - len(matched_gt)

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return [precision], [recall]

    def compute_iou(self, box1, box2):
        # Compute the intersection over union (IoU) of two bounding boxes
        x1, y1, x2, y2 = box1
        x1_gt, y1_gt, x2_gt, y2_gt = box2

        # Determine the coordinates of the intersection rectangle
        xi1 = max(x1, x1_gt)
        yi1 = max(y1, y1_gt)
        xi2 = min(x2, x2_gt)
        yi2 = min(y2, y2_gt)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Compute the area of both the prediction and ground-truth rectangles
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

        # Compute the IoU
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou