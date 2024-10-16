from mmcv.runner import HOOKS, Hook
import wandb

@HOOKS.register_module()
class WandBPrecisionRecallHook(Hook):
    def after_train_epoch(self, runner):
        # Assuming validation results are accessible
        print("runner latest results:", runner.log_buffer.output)
        predictions = runner.outputs['preds']
        print("outpurs preds:", runner.outputs['preds'])
        ground_truth = runner.outputs['gt']
        print("outputs ground truth:", runner.outputs['gt'])

        # Log precision-recall curve
        pr_curve = wandb.plot.pr_curve(
            ground_truth, predictions,
            labels=["General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"] 
        )
        wandb.log({"precision_recall_curve": pr_curve})
