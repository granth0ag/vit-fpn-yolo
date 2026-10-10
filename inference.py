# inference.py
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:                                  # purane ultralytics
    from ultralytics.utils.ops import non_max_suppression

from src.model.vit import ViTDetBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel

# ---------------- config ----------------
IMG_SIZE = 320
CKPT = "best.pt"
CONF_THRES = 0.25
IOU_THRES = 0.45
OUT_DIR = Path("outputs")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
device = "cuda" if torch.cuda.is_available() else "cpu"

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def load_model():
    model = DetectionModel(
        ViTDetBackbone(img_size=IMG_SIZE),
        SimpleFeaturePyramid(in_ch=384, out_ch=128),
        YoloHead(num_classes=80, ch=(128, 128, 128)),
    ).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    return model.eval()


def letterbox(image, S=IMG_SIZE):
    """dataset.py wala same letterbox. Returns canvas, scale, pad_x, pad_y."""
    w0, h0 = image.size
    scale = S / max(w0, h0)
    new_w, new_h = round(w0 * scale), round(h0 * scale)
    pad_x, pad_y = (S - new_w) // 2, (S - new_h) // 2
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


@torch.no_grad()
def detect(model, image):
    """PIL image -> [n, 6] (x1, y1, x2, y2, conf, cls) in ORIGINAL image pixels."""
    w0, h0 = image.size
    canvas, scale, pad_x, pad_y = letterbox(image)
    x = to_tensor(canvas).unsqueeze(0).to(device)

    y, _ = model(x)                                              # [1, 84, 2100]
    det = non_max_suppression(y, conf_thres=CONF_THRES, iou_thres=IOU_THRES)[0].cpu()

    # letterbox pixels -> original pixels
    det[:, [0, 2]] = ((det[:, [0, 2]] - pad_x) / scale).clamp(0, w0)
    det[:, [1, 3]] = ((det[:, [1, 3]] - pad_y) / scale).clamp(0, h0)
    return det


def draw(image, det):
    image = image.copy()
    d = ImageDraw.Draw(image)
    for x1, y1, x2, y2, conf, cls in det.tolist():
        label = f"{COCO_NAMES[int(cls)]} {conf:.2f}"
        d.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        d.text((x1 + 2, y1 + 2), label, fill="white")
    return image


if __name__ == "__main__":
    # usage: python inference.py img1.jpg img2.jpg   (ya koi folder)
    paths = []
    for a in sys.argv[1:]:
        p = Path(a)
        paths += sorted(p.glob("*.jpg")) if p.is_dir() else [p]
    if not paths:                                                # default: coco128 ki pehli 4 images
        paths = sorted(Path("data/coco128/images/train2017").glob("*.jpg"))[:4]

    OUT_DIR.mkdir(exist_ok=True)
    model = load_model()
    for p in paths:
        image = Image.open(p).convert("RGB")
        det = detect(model, image)
        draw(image, det).save(OUT_DIR / p.name)
        print(f"{p.name}: {len(det)} detections")