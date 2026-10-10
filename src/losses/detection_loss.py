# detector_loss.py
import torch
from ultralytics.utils.loss import v8DetectionLoss


def make_batch(targets, device):
    """Tere list-of-tensors targets -> ultralytics ka batch dict."""
    idx = torch.cat([torch.full((len(t),), i, dtype=torch.float32)
                     for i, t in enumerate(targets)])
    t = torch.cat(targets)
    return {
        "batch_idx": idx.to(device),       # [N]
        "cls":       t[:, :1].to(device),  # [N, 1]
        "bboxes":    t[:, 1:].to(device),  # [N, 4] cx, cy, w, h (0-1)
    }


class DetectionLoss:
    """Thin wrapper: model + targets list -> loss."""
    def __init__(self, model):
        self.criterion = v8DetectionLoss(model)   # model ko device pe bhejne ke BAAD banana

    def __call__(self, preds, targets, device):
        loss, loss_items = self.criterion(preds, make_batch(targets, device))
        return loss.sum(), loss_items.detach()    # scalar for backward, [box, cls, dfl] for logging