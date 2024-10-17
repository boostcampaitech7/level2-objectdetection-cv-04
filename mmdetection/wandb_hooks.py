from mmcv.runner import HOOKS, Hook, EvalHook
import wandb
import numpy as np
from mmdet.apis import single_gpu_test

@HOOKS.register_module()
class CustomEvalHook(EvalHook):
    def _do_evaluate(self, runner):
        """Perform evaluation and save the results in runner.outputs."""
        if not self._should_evaluate(runner):
            return

        # Run the single GPU test
        results = single_gpu_test(runner.model, self.dataloader, show=False)

        # Store the results in runner.outputs to make them available for hooks
        runner.outputs['results'] = results

        # Proceed with the rest of the evaluation
        key_score = self.evaluate(runner, results)
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class WandBPrecisionRecallHook(Hook):
    def after_val_epoch(self, runner):
        # Get dataset from the runner's data loader
        dataset = runner.data_loader.dataset
        # Get validation results from runner's outputs
        results = runner.outputs.get('results', None)
        # If results are not available, skip this hook
        if results is None:
            return
        
        ground_truth = []
        predictions = []

        for idx, (img_info, pred) in enumerate(zip(dataset.data_infos, results)):
            ann = dataset.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            for label in gt_labels:
                ground_truth.append(label)

            for bbox in pred:
                x1, y1, x2, y2, score = bbox[:5]
                label = int(bbox[5]) if bbox.shape[0] > 5 else 0  # Assuming label is present in bbox data
                predictions.append({
                    'score': score,
                    'label': label
                })

        # Log precision-recall curve to WandB
        pr_curve = wandb.plot.pr_curve(
            ground_truth=ground_truth,
            predictions=predictions,
            labels=["General trash", "Paper", "Paper pack", "Metal", "Glass",
                    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
        )
        wandb.log({"precision_recall_curve": pr_curve})
