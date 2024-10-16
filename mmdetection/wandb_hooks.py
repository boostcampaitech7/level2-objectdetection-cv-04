from mmcv.runner import Hook
import wandb

class WandBPrecisionRecallHook(Hook):
    def after_val_epoch(self, runner):
        # Assuming validation results are accessible
        predictions = runner.outputs['preds']
        ground_truth = runner.outputs['gt']

        # Log precision-recall curve
        pr_curve = wandb.plot.pr_curve(
            ground_truth, predictions,
            labels=["General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"] 
        )
        wandb.log({"precision_recall_curve": pr_curve})
