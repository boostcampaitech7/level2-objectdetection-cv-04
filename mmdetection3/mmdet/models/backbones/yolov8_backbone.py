import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
import math

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

@MODELS.register_module()
class YOLOv8Backbone(BaseModule):
    def __init__(self, out_indices=(2, 3, 4)):
        super().__init__()
        self.out_indices = out_indices
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
        if 2 in self.out_indices:
            outputs.append(x)
        x = self.conv5(x)
        x = self.csp4(x)
        if 3 in self.out_indices:
            outputs.append(x)
        x = self.conv6(x)
        x = self.csp5(x)
        if 4 in self.out_indices:
            outputs.append(x)
        return tuple(outputs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            try:
                self.load_state_dict(torch.load(pretrained))
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # SiLU에 대한 근사 게인 값 사용
                    gain = math.sqrt(2.0 / (1 + 0.25**2))
                    nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)