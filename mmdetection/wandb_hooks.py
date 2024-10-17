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
        # Get validation results from runner's outputs
        results = runner.outputs.get('results', None)
        # If results are not available, skip this hook
        if results is None:
            return
        
        ground_truth = []
        predictions = []

        for idx, (gt, pred) in enumerate(zip(runner.data_loader.dataset, results)):
            # Extracting ground truth from dataset
            gt_labels = gt['ann']['labels'].tolist()  # Extract labels as a list of integers
            ground_truth.extend(gt_labels)  # Append all ground truth labels

            # Extracting predictions from results
            pred_labels = pred[:, 4]  # Extract predicted class labels
            pred_confidences = pred[:, 4]  # Extract prediction scores
            predictions.extend(list(zip(pred_labels, pred_confidences)))  # Create tuples and append

        # Log precision-recall curve to WandB
        pr_curve = wandb.plot.pr_curve(
            ground_truth=ground_truth,
            predictions=predictions,
            labels=["General trash", "Paper", "Paper pack", "Metal", "Glass",
                    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
        )
        wandb.log({"precision_recall_curve": pr_curve})
