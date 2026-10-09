

import timm
import torch 

model = timm.create_model(
      "vit_small_patch16_224",
    pretrained=True,
    num_classes=0,  # no classification head
)
