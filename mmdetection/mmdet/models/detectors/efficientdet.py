# EfficientDetHead Implementation for MMDetection

import torch.nn as nn
from mmdet.models.detectors.single_stage import SingleStageDetector

class EfficientDet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=dict(allowed_border=0),
                 test_cfg=dict(allowed_border=0),
                 pretrained=None):
        super(EfficientDet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        for i, feat in enumerate(x):
            print(f"Backbone output {i}: shape = {feat.shape}")
        x = self.neck(x)
        return x