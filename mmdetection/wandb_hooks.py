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
    def after_train_epoch(self, runner):
        # Get dataset from the runner's data loader
        dataset = runner.data_loader.dataset

        # Get validation results from runner's outputs
        results = runner.outputs.get('results', None)

        # If results are not available, skip this hook
        if results is None:
            print("No validation results found, skipping PR curve drawing.")
            return

        ground_truth_labels = []
        prediction_scores = []

        # Extract ground truth labels and prediction scores
        for idx, (img_info, pred) in enumerate(zip(dataset.data_infos, results)):
            if pred is None or len(pred) == 0:
                print(f"Unexpected bbox format with length 0 for image index {idx}: {pred}")
                continue

            ann = dataset.get_ann_info(idx)
            gt_labels = ann.get('labels', [])

            if len(gt_labels) == 0:
                print(f"No ground truth labels for image index {idx}.")
                continue

            # Append ground truth labels
            for label in gt_labels:
                ground_truth_labels.append(label)

            # Append predicted scores for the bounding boxes
            for bbox in pred:
                if len(bbox) >= 5:  # Ensure bbox has at least 5 elements: [x1, y1, x2, y2, score]
                    score = bbox[4]  # The 5th value is the score
                    prediction_scores.append(score)
                else:  # Log a warning if the bbox size isn't as expected
                    print(f"Unexpected bbox format with length {len(bbox)}: {bbox}")

        # Ensure that ground truth labels and prediction scores have some valid data to log
        if len(ground_truth_labels) == 0 or len(prediction_scores) == 0:
            print("No valid data available for drawing PR curve.")
            return

        # Log precision-recall curve to WandB
        try:
            print("Attempting to draw PR curve in WandB...")
            # If ground truth and predictions lengths don't match, use the minimum length
            min_length = min(len(ground_truth_labels), len(prediction_scores))
            filtered_ground_truth = ground_truth_labels[:min_length]
            filtered_scores = prediction_scores[:min_length]

            data = [[filtered_ground_truth[i], filtered_scores[i]] for i in range(min_length)]
            table = wandb.Table(data=data, columns=["ground_truth", "score"])
            pr_curve = wandb.plot.pr_curve(
                table,
                "ground_truth",
                "score"
            )
            wandb.log({"precision_recall_curve": pr_curve})
            print("PR curve drawn successfully.")
        except Exception as e:
            print("Error drawing PR curve:", str(e))
