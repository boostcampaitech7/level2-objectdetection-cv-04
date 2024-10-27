import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = ConvBNSiLU(in_channels, mid_channels, 1, 1, 0)
        self.conv3 = ConvBNSiLU(mid_channels, mid_channels, 1, 1, 0)
        self.conv4 = ConvBNSiLU(mid_channels, mid_channels, 1, 1, 0)
        self.bottlenecks = nn.Sequential(*[ConvBNSiLU(mid_channels, mid_channels, 3, 1, 1) for _ in range(num_bottlenecks)])
        self.conv5 = ConvBNSiLU(mid_channels * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.bottlenecks(y2)
        y2 = self.conv3(y2)
        y = torch.cat([y2, y1], dim=1)
        y = self.conv5(y)
        return y

class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNSiLU(3, 32, 3, 1, 1)
        self.conv2 = ConvBNSiLU(32, 64, 3, 2, 1)
        self.csp1 = CSPLayer(64, 64, 1)
        self.conv3 = ConvBNSiLU(64, 128, 3, 2, 1)
        self.csp2 = CSPLayer(128, 128, 2)
        self.conv4 = ConvBNSiLU(128, 256, 3, 2, 1)
        self.csp3 = CSPLayer(256, 256, 8)
        self.conv5 = ConvBNSiLU(256, 512, 3, 2, 1)
        self.csp4 = CSPLayer(512, 512, 8)
        self.conv6 = ConvBNSiLU(512, 1024, 3, 2, 1)
        self.csp5 = CSPLayer(1024, 1024, 4)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.csp1(x)
        x = self.conv3(x)
        x = self.csp2(x)
        x = self.conv4(x)
        x = self.csp3(x)
        outputs.append(x)
        x = self.conv5(x)
        x = self.csp4(x)
        outputs.append(x)
        x = self.conv6(x)
        x = self.csp5(x)
        outputs.append(x)
        return outputs

class YOLOv8Head(nn.Module):
    def __init__(self, num_classes, in_channels=[256, 512, 1024]):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.stems = nn.ModuleList([
            ConvBNSiLU(in_ch, in_ch, 1, 1, 0) for in_ch in in_channels
        ])

        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(in_ch, in_ch, 3, 1, 1),
                ConvBNSiLU(in_ch, in_ch, 3, 1, 1)
            ) for in_ch in in_channels
        ])

        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(in_ch, in_ch, 3, 1, 1),
                ConvBNSiLU(in_ch, in_ch, 3, 1, 1)
            ) for in_ch in in_channels
        ])

        self.cls_preds = nn.ModuleList([
            nn.Conv2d(in_ch, self.num_classes, 1) for in_ch in in_channels
        ])

        self.reg_preds = nn.ModuleList([
            nn.Conv2d(in_ch, 4, 1) for in_ch in in_channels
        ])


    def forward(self, inputs):
        outputs = []
        for i, x in enumerate(inputs):
            x = self.stems[i](x)

            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)

            cls_output = self.cls_preds[i](cls_feat)
            reg_output = self.reg_preds[i](reg_feat)

            # 출력을 [batch_size, 5 + num_classes, height, width] 형태로 변경
            output = torch.cat([reg_output, cls_output], 1)
            outputs.append(output)

        return outputs

class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = YOLOv8Backbone()
        self.head = YOLOv8Head(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs  # 이 부분이 [batch_size, 5 + num_classes, height, width] 형태여야 합니다

class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOULoss()
        self.strides = [8, 16, 32]  # 여러 스케일의 특징 맵에 대한 스트라이드

    def forward(self, predictions, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        for i, pred in enumerate(predictions):
            batch_size, _, grid_h, grid_w = pred.shape
            pred = pred.view(batch_size, 4 + self.num_classes, grid_h, grid_w)
            pred_boxes, pred_cls = pred.split([4, self.num_classes], dim=1)
            
            matched_gt_boxes, matched_gt_cls, mask = self.match_targets_to_predictions(
                pred_boxes, targets, (grid_h, grid_w), self.strides[i]
            )
            
            # 박스 손실 계산
            if mask.sum() > 0:
                pred_boxes_masked = pred_boxes.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)[mask]
                matched_gt_boxes_masked = matched_gt_boxes[mask]
                iou = self.iou_loss(pred_boxes_masked, matched_gt_boxes_masked)
                lbox += (1.0 - iou).mean()
            
            # 분류 손실 계산
            pred_cls_masked = pred_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)[mask]
            matched_gt_cls_masked = matched_gt_cls[mask]
            lcls += self.bce(pred_cls_masked, matched_gt_cls_masked).mean()
            
            # 객체성 손실 계산 (간단한 구현을 위해 마스크를 객체성 점수로 사용)
            pred_obj = pred_cls.max(dim=1, keepdim=True)[0]  # 최대 클래스 점수를 객체성 점수로 사용
            lobj += self.bce(pred_obj.view(batch_size, -1), mask.float()).mean()
        
        # 손실 가중치 조정
        lbox *= 5.0
        lcls *= 1.0
        lobj *= 1.0
        
        loss = lbox + lcls + lobj
        return loss, torch.cat((lbox, lcls, lobj, loss)).detach()

    def match_targets_to_predictions(self, pred_boxes, targets, grid_size, stride):
        device = pred_boxes.device
        batch_size, _, height, width = pred_boxes.shape
        num_classes = self.num_classes

        # 그리드 생성
        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).to(device).float()

        # 타겟 변환
        gt_boxes = targets[:, :, 1:5] * torch.tensor([width, height, width, height], device=device)
        gt_classes = targets[:, :, 0].long()

        matched_gt_boxes = torch.zeros((batch_size, height, width, 4), device=device)
        matched_gt_cls = torch.zeros((batch_size, height, width, num_classes), device=device)
        mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=device)

        for b in range(batch_size):
            for t in range(targets.shape[1]):
                if targets[b, t].sum() == 0:  # 패딩된 타겟 무시
                    continue

                # 중심점 및 크기 계산
                gx, gy = gt_boxes[b, t, 0], gt_boxes[b, t, 1]
                gw, gh = gt_boxes[b, t, 2], gt_boxes[b, t, 3]

                # 동적 할당: 객체 크기에 따라 할당 범위 조정
                radius = max(3, int(max(gw, gh) / stride))
                
                gi, gj = (gx / stride).long(), (gy / stride).long()
                gi = torch.clamp(gi, 0, width - 1)
                gj = torch.clamp(gj, 0, height - 1)

                # 품질 평가 및 포지티브 샘플 선택
                for i in range(max(0, gi-radius), min(width, gi+radius+1)):
                    for j in range(max(0, gj-radius), min(height, gj+radius+1)):
                        # 간단한 중심성 기반 품질 점수 계산 (실제로는 IoU 등을 사용할 수 있음)
                        quality = 1 - ((i - gx/stride)**2 + (j - gy/stride)**2) / (2 * radius**2)
                        if quality > 0:
                            mask[b, j, i] = True
                            matched_gt_boxes[b, j, i] = torch.tensor([gx, gy, gw, gh], device=device)
                            matched_gt_cls[b, j, i, gt_classes[b, t]] = 1

        matched_gt_boxes = matched_gt_boxes.view(batch_size, -1, 4)
        matched_gt_cls = matched_gt_cls.view(batch_size, -1, num_classes)
        mask = mask.view(batch_size, -1)

        return matched_gt_boxes, matched_gt_cls, mask

class IOULoss(nn.Module):
    def __init__(self, reduction="none"):
        super(IOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)
        target_area = (target_right - target_left) * (target_bottom - target_top)

        w_intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
        h_intersect = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = pred_area + target_area - area_intersect

        iou = area_intersect / area_union

        loss = 1 - iou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

# class YOLOv8Dataset(Dataset):
#     def __init__(self, img_dir, anno_path, transform=None):
#         # 데이터셋 초기화

#     def __len__(self):
#         # 데이터셋 길이 반환

#     def __getitem__(self, idx):
#         # 이미지와 레이블 로드 및 전처리
#         return img, labels
