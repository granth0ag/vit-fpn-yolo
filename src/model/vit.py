import torch 
import timm 
import torch.nn as nn

class ViTBackBone(nn.Module):
    def __init__(self,img_size = 320):
        super().__init__()

        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            img_size=img_size
        )
        self.embed_dim = self.vit.embed_dim
        self.patch_size = 16

    def forward(self,x):

        tokens = self.vit.forward_features(x)
        # Remove CLS token
        tokens = tokens[:, 1:, :]            
        B,N,C =tokens.shape
        H = W = int(N ** 0.5)
        # [B, N, C] -> [B, C, H, W]
        features = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return features



