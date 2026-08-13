"""
backbones.py
DenseNet-121 and ViT feature extractors for CSAF-Net.

Both backbones output a sequence of tokens (B, N, D) so they can be
fed directly into the gated cross-attention fusion module.

Usage:
    from backbones import DenseNetBackbone, ViTBackbone, get_device

    device = get_device()
    cnn_backbone = DenseNetBackbone().to(device)
    vit_backbone = ViTBackbone().to(device)

    cnn_feats = cnn_backbone(images)   # (B, N_cnn, proj_dim)
    vit_feats = vit_backbone(images)   # (B, N_vit, proj_dim)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel


PROJ_DIM = 256  # common embedding dim both backbones project into


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DenseNetBackbone(nn.Module):
    """
    DenseNet-121 feature extractor.
    Takes the last conv feature map (B, C, H, W), flattens spatial dims
    into a token sequence (B, H*W, C), then projects to PROJ_DIM.
    """

    def __init__(self, proj_dim=PROJ_DIM, pretrained=True, freeze=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        # features: everything up to (and including) the final conv block,
        # excludes the classifier head
        self.features = densenet.features
        out_channels = 1024  # DenseNet-121's final feature map channel count

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(out_channels, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        # x: (B, 3, H, W) -- expects 224x224 input
        feat_map = self.features(x)               # (B, 1024, h, w)
        feat_map = torch.relu(feat_map)            # DenseNet convention before pooling
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)   # (B, H*W, C)
        tokens = self.proj(tokens)                      # (B, H*W, proj_dim)
        tokens = self.norm(tokens)
        return tokens


class ViTBackbone(nn.Module):
    """
    ViT feature extractor (google/vit-base-patch16-224-in21k).
    Returns patch token embeddings (excludes CLS token by default),
    projected to PROJ_DIM.
    """

    def __init__(self, proj_dim=PROJ_DIM, pretrained_name="google/vit-base-patch16-224-in21k",
                 freeze=True, include_cls=False):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_name)
        out_dim = self.vit.config.hidden_size  # 768 for vit-base
        self.include_cls = include_cls

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(out_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        # x: (B, 3, 224, 224), already normalized/preprocessed
        outputs = self.vit(pixel_values=x)
        tokens = outputs.last_hidden_state   # (B, 197, 768) -- 1 CLS + 196 patches

        if not self.include_cls:
            tokens = tokens[:, 1:, :]        # drop CLS token -> (B, 196, 768)

        tokens = self.proj(tokens)           # (B, N, proj_dim)
        tokens = self.norm(tokens)
        return tokens


if __name__ == "__main__":
    # quick smoke test
    device = get_device()
    print(f"Using device: {device}")

    dummy = torch.randn(2, 3, 224, 224).to(device)

    print("\nLoading DenseNet-121 backbone...")
    cnn_backbone = DenseNetBackbone().to(device)
    cnn_out = cnn_backbone(dummy)
    print(f"  CNN token output shape: {cnn_out.shape}")   # (2, 49, 256) for 224x224 input

    print("\nLoading ViT backbone...")
    vit_backbone = ViTBackbone().to(device)
    vit_out = vit_backbone(dummy)
    print(f"  ViT token output shape: {vit_out.shape}")   # (2, 196, 256)

    print("\nBoth backbones ran successfully.")
