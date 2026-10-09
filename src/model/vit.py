import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def window_partition(x, ws):
    """[B, H, W, C] -> [B * num_windows, ws*ws, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_unpartition(x, ws, H, W):
    """[B * num_windows, ws*ws, C] -> [B, H, W, C]"""
    B = x.shape[0] // ((H // ws) * (W // ws))
    x = x.view(B, H // ws, W // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


def run_attention(attn, x):
    """Reuse a timm Attention module's weights. x: [B, N, C]"""
    B, N, C = x.shape
    h = attn.num_heads
    q, k, v = attn.qkv(x).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
    out = F.scaled_dot_product_attention(q, k, v)
    return attn.proj(out.transpose(1, 2).reshape(B, N, C))


class ViTDetBackbone(nn.Module):
    def __init__(self, img_size=320, window_size=10, global_idx=(2, 5, 8, 11)):
        super().__init__()
        # pretrained pos_embed gets resampled to the 320 grid by timm
        self.vit = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=0, img_size=img_size
        )
        self.patch_size = 16
        self.embed_dim = self.vit.embed_dim
        self.window_size = window_size
        self.global_idx = set(global_idx)

        g = img_size // self.patch_size
        assert g % window_size == 0, f"grid {g} must be divisible by window_size {window_size}"

        # drop the CLS position, keep the patch positions as a [1, g, g, C] grid
        pos = self.vit.pos_embed[:, 1:, :].detach().clone()
        self.pos_embed = nn.Parameter(pos.reshape(1, g, g, -1))

    def forward(self, x):
        x = self.vit.patch_embed.proj(x).permute(0, 2, 3, 1)   # [B, H, W, C]
        x = x + self.pos_embed
        B, H, W, C = x.shape

        for i, blk in enumerate(self.vit.blocks):
            y = blk.norm1(x)
            if i in self.global_idx:                           # cross-window propagation
                y = run_attention(blk.attn, y.reshape(B, H * W, C)).view(B, H, W, C)
            else:                                              # windowed attention
                y = window_unpartition(
                    run_attention(blk.attn, window_partition(y, self.window_size)),
                    self.window_size, H, W)
            x = x + blk.drop_path1(blk.ls1(y))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

        x = self.vit.norm(x)
        return x.permute(0, 3, 1, 2)                           # [B, C, H, W]