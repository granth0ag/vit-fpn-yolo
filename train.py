# train.py
import random
import numpy as np
import torch

from src.dataset import build_dataloaders
from src.model.vit import ViTDetBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel
from src.losses.detection_loss import DetectionLoss

# ---------------- config ----------------
IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 100
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3          # neck + head
WEIGHT_DECAY = 0.05
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- data ----------------
train_loader, _ = build_dataloaders(batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=False)

# ---------------- model ----------------
backbone = ViTDetBackbone(img_size=IMG_SIZE)
neck = SimpleFeaturePyramid(in_ch=384, out_ch=128)
head = YoloHead(num_classes=80, ch=(128, 128, 128))
model = DetectionModel(backbone, neck, head).to(device)

criterion = DetectionLoss(model)                 # model .to(device) ke BAAD

# ---------------- optimizer ----------------
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
    {"params": model.neck.parameters(),     "lr": LR_HEAD},
    {"params": model.head.parameters(),     "lr": LR_HEAD},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ---------------- training ----------------
def train_one_epoch():
    model.train()
    total, items = 0.0, torch.zeros(3)
    for images, targets, _ in train_loader:
        images = images.to(device)
        preds = model(images)                                    # 3 raw maps
        loss, loss_items = criterion(preds, targets, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total += loss.item()
        items += loss_items.cpu()
    n = len(train_loader)
    return total / n, items / n


best = float("inf")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_items = train_one_epoch()
    scheduler.step()

    print(f"epoch {epoch:3d} | loss {tr_loss:8.3f} "
          f"(box {tr_items[0]:.3f} cls {tr_items[1]:.3f} dfl {tr_items[2]:.3f})")

    if tr_loss < best:
        best = tr_loss
        torch.save(model.state_dict(), "best.pt")

torch.save(model.state_dict(), "last.pt")