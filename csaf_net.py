"""
csaf_net.py
Full CSAF-Net model: DenseNet-121 + ViT backbones -> Gated Cross-Attention
Fusion -> classification head.

Also defines lightweight wrapper models for your ablation study
(CNN-only, ViT-only, concat, ungated cross-attention, scalar-gated
cross-attention) so every variant shares the same backbones and head --
only the fusion strategy changes.

Usage:
    from csaf_net import build_model, get_device

    device = get_device()
    model = build_model(variant="full_csaf").to(device)
    logits, gate = model(images)   # images: (B, 3, 224, 224)
"""

import torch
import torch.nn as nn

from backbones import DenseNetBackbone, ViTBackbone, get_device, PROJ_DIM
from csaf_module import (
    GatedCrossAttentionFusion,
    ConcatFusion,
    UngatedCrossAttentionFusion,
    ScalarGatedCrossAttentionFusion,
)


class ClassificationHead(nn.Module):
    """Pools fused tokens and outputs a single crack/no-crack logit."""

    def __init__(self, dim=PROJ_DIM, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, fused_tokens):
        # fused_tokens: (B, N, dim) -> pool over tokens -> (B, dim)
        pooled = fused_tokens.mean(dim=1)
        return self.net(pooled).squeeze(-1)   # (B,) raw logit


class CSAFNet(nn.Module):
    """
    Full CSAF-Net: dual backbone + gated cross-scale attention fusion + head.

    variant controls which fusion module is used, so this one class serves
    every ablation entry except the single-backbone baselines.
    """

    VARIANT_MODULES = {
        "full_csaf": GatedCrossAttentionFusion,
        "concat": ConcatFusion,
        "ungated_cross_attn": UngatedCrossAttentionFusion,
        "scalar_gate": ScalarGatedCrossAttentionFusion,
    }

    def __init__(self, variant="full_csaf", dim=PROJ_DIM, num_heads=8,
                 freeze_backbones=True):
        super().__init__()
        assert variant in self.VARIANT_MODULES, f"Unknown variant: {variant}"
        self.variant = variant

        self.cnn_backbone = DenseNetBackbone(proj_dim=dim, freeze=freeze_backbones)
        self.vit_backbone = ViTBackbone(proj_dim=dim, freeze=freeze_backbones)

        fusion_cls = self.VARIANT_MODULES[variant]
        if variant == "concat":
            self.fusion = fusion_cls(dim=dim)
        else:
            self.fusion = fusion_cls(dim=dim, num_heads=num_heads)

        self.head = ClassificationHead(dim=dim)

    def forward(self, images):
        cnn_tokens = self.cnn_backbone(images)   # (B, N_cnn, dim)
        vit_tokens = self.vit_backbone(images)   # (B, N_vit, dim)
        fused, gate = self.fusion(cnn_tokens, vit_tokens)   # (B, N, dim), gate or None
        logits = self.head(fused)   # (B,)
        return logits, gate


class SingleBackboneNet(nn.Module):
    """
    CNN-only or ViT-only baseline for the ablation ladder.
    Same classification head as CSAFNet for a fair comparison.
    """

    def __init__(self, backbone_type="cnn", dim=PROJ_DIM, freeze_backbone=True):
        super().__init__()
        assert backbone_type in ("cnn", "vit")
        self.backbone_type = backbone_type

        if backbone_type == "cnn":
            self.backbone = DenseNetBackbone(proj_dim=dim, freeze=freeze_backbone)
        else:
            self.backbone = ViTBackbone(proj_dim=dim, freeze=freeze_backbone)

        self.head = ClassificationHead(dim=dim)

    def forward(self, images):
        tokens = self.backbone(images)   # (B, N, dim)
        logits = self.head(tokens)   # (B,)
        return logits, None


def build_model(variant="full_csaf", **kwargs):
    """
    Factory function used by train.py / run_ablations.py.

    variant options:
        "cnn_only"            -> DenseNet-121 + head
        "vit_only"            -> ViT + head
        "concat"              -> ConcatFusion ablation
        "ungated_cross_attn"  -> UngatedCrossAttentionFusion ablation
        "scalar_gate"         -> ScalarGatedCrossAttentionFusion ablation
        "full_csaf"           -> full CSAF-Net (your novelty)
    """
    if variant == "cnn_only":
        return SingleBackboneNet(backbone_type="cnn", **kwargs)
    elif variant == "vit_only":
        return SingleBackboneNet(backbone_type="vit", **kwargs)
    elif variant in CSAFNet.VARIANT_MODULES:
        return CSAFNet(variant=variant, **kwargs)
    else:
        raise ValueError(f"Unknown model variant: {variant}")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}\n")

    dummy = torch.randn(2, 3, 224, 224).to(device)

    for variant in ["cnn_only", "vit_only", "concat", "ungated_cross_attn", "scalar_gate", "full_csaf"]:
        print(f"Testing variant: {variant}")
        model = build_model(variant=variant).to(device)
        model.eval()
        with torch.no_grad():
            logits, gate = model(dummy)
        print(f"  logits shape: {logits.shape}")   # (2,)
        if gate is not None:
            print(f"  gate shape: {gate.shape}")
        print()

    print("All model variants ran successfully end-to-end.")