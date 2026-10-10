import torch
import torch.nn as nn


class SimpleFeaturePyramid(nn.Module):
    """ViT stride-16 map [B, 384, 20, 20] -> P3 [B,C,40,40], P4 [B,C,20,20], P5 [B,C,10,10]"""

    def __init__(self, in_ch=384, out_ch=128):
        super().__init__()
        # three parallel branches, all from the SAME input map
        self.to_p3 = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)  # 20 -> 40
        self.to_p4 = nn.Identity()                                                    # 20 -> 20
        self.to_p5 = nn.MaxPool2d(kernel_size=2, stride=2)                            # 20 -> 10

        # per-level head: 1x1 conv -> norm -> 3x3 conv -> norm, so every level has out_ch channels
        def head(c):
            return nn.Sequential(
                nn.Conv2d(c, out_ch, 1, bias=False), nn.GroupNorm(8, out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(8, out_ch),
            )

        self.head_p3 = head(in_ch // 2)
        self.head_p4 = head(in_ch)
        self.head_p5 = head(in_ch)

    def forward(self, x):
        p3 = self.head_p3(self.to_p3(x))   # [B, out_ch, 40, 40]
        p4 = self.head_p4(self.to_p4(x))   # [B, out_ch, 20, 20]
        p5 = self.head_p5(self.to_p5(x))   # [B, out_ch, 10, 10]
        return p3, p4, p5