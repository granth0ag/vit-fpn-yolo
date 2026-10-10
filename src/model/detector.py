import torch
import torch.nn as nn
from types import SimpleNamespace
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import ops


class YoloHead(nn.Module):
    """Takes [p3, p4, p5] (all 128 ch) -> raw YOLOv8 predictions."""
    def __init__(self, num_classes=80, ch=(128, 128, 128), strides=(8, 16, 32)):
        super().__init__()
        self.detect = Detect(nc=num_classes, ch=ch)
        self.detect.stride = torch.tensor(strides, dtype=torch.float32)
        self.detect.bias_init()          # sensible init for cls/box biases

    def forward(self, feats):            # feats = [p3, p4, p5]
        return self.detect(list(feats))
        # train mode -> list of 3 raw maps [B, 64+nc, H, W]
        # eval  mode -> (decoded [B, 4+nc, num_anchors], raw maps)


class DetectionModel(nn.Module):
    """Thin wrapper because v8DetectionLoss expects model.model[-1] and model.args."""
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone, self.neck, self.head = backbone, neck, head
        self.model = nn.ModuleList([head.detect])                       # loss looks here
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)          # loss gains

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))