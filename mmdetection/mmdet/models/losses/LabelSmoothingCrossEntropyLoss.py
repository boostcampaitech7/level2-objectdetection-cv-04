import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss 

@LOSSES.register_module()
class LabelSmoothingCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, label_smoothing=0.1, **kwargs):
        super(LabelSmoothingCrossEntropyLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        # 클래스 수 계산
        n_classes = cls_score.size(-1)
        
        # 클래스 수가 1이면 label smoothing을 적용하지 않음
        if n_classes == 1:
            return super(LabelSmoothingCrossEntropyLoss, self).forward(
                cls_score,
                label,
                weight,
                avg_factor,
                reduction_override,
                ignore_index,
                **kwargs
            )

        if self.label_smoothing > 0:
            # Label smoothing 적용
            smoothing = self.label_smoothing / (n_classes - 1)
            one_hot = torch.full_like(cls_score, smoothing).scatter_(
                -1, label.unsqueeze(-1), 1.0 - self.label_smoothing
            )
            log_prob = F.log_softmax(cls_score, dim=-1)
            loss = (-one_hot * log_prob).sum(dim=-1)

            # reduction 방식 적용
            if reduction_override == 'mean':
                return loss.mean()
            elif reduction_override == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            # label smoothing이 적용되지 않은 경우, 부모 클래스의 forward 호출
            return super(LabelSmoothingCrossEntropyLoss, self).forward(
                cls_score,
                label,
                weight,
                avg_factor,
                reduction_override,
                ignore_index,
                **kwargs
            )