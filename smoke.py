# smoke.py
import time
import torch

from src.dataset import build_dataloaders
from src.model.vit import ViTDetBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel
from src.losses.detection_loss import DetectionLoss
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

device = "cuda" if torch.cuda.is_available() else "cpu"
print("1) device:", device, "| torch", torch.__version__)

# ---- data ----
train_loader, _ = build_dataloaders(batch_size=8, img_size=320)
images, targets, paths = next(iter(train_loader))
print("2) data:", images.shape, "| targets:", len(targets), targets[0].shape)
ids = torch.cat([t[:, 0] for t in targets])
assert ids.min() >= 0 and ids.max() < 80, "class id range galat"

# ---- model ----
model = DetectionModel(
    ViTDetBackbone(img_size=320),
    SimpleFeaturePyramid(in_ch=384, out_ch=128),
    YoloHead(num_classes=80, ch=(128, 128, 128)),
).to(device)
criterion = DetectionLoss(model)

images = images.to(device)

# ---- forward shapes ----
model.train()
feat = model.backbone(images)
print("3) backbone:", feat.shape)                       # [8, 384, 20, 20]
print("   neck    :", [p.shape for p in model.neck(feat)])
preds = model(images)
print("   head    :", [p.shape for p in preds])         # [8,144,40,40], [8,144,20,20], [8,144,10,10]

# ---- loss + gradients ----
loss, items = criterion(preds, targets, device)
print("4) loss:", loss.item(), "| items:", items)
assert torch.isfinite(loss), "loss NaN/inf"
loss.backward()
for name, m in [("backbone", model.backbone), ("neck", model.neck), ("head", model.head)]:
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    print(f"   grad in {name}: {has_grad}")

# ---- overfit ONE batch ----
opt = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.neck.parameters(),     "lr": 1e-3},
    {"params": model.head.parameters(),     "lr": 1e-3},
], weight_decay=0.05)

print("5) overfit 1 batch, 40 steps")
t0 = time.time()
for step in range(40):
    preds = model(images)
    loss, items = criterion(preds, targets, device)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    if step % 10 == 0 or step == 39:
        print(f"   step {step:2d} loss {loss.item():8.3f}")
if device == "cuda":
    torch.cuda.synchronize()
dt = (time.time() - t0) / 40
print(f"   {dt:.3f} s/step -> ~{dt * 16:.1f} s per coco128 epoch (16 steps)")

# ---- eval path ----
model.eval()
with torch.no_grad():
    y, _ = model(images)
print("6) eval y:", y.shape)                            # [8, 84, 2100]
dets = non_max_suppression(y, conf_thres=0.001, iou_thres=0.65)
print("   NMS ok, dets per image:", [len(d) for d in dets][:4])

# ---- checkpoint roundtrip ----
torch.save(model.state_dict(), "smoke.pt")
model.load_state_dict(torch.load("smoke.pt", map_location=device))
print("7) save/load ok\nSMOKE TEST PASSED")