"""
csaf_module.py
Gated Cross-Scale Attention Fusion (CSAF) module.

This is the novelty of CSAF-Net:
1. Bidirectional cross-attention between CNN tokens and ViT tokens
   (CNN queries ViT for global context, ViT queries CNN for local detail)
2. A learned gate, conditioned on BOTH attended outputs, decides per-token
   how much to trust the CNN-attended branch vs the ViT-attended branch.

Also included: simpler fusion variants (concat, add, no-gate cross-attention)
used for your ablation study, so all variants live in one place and share
the same interface.

Usage:
    from csaf_module import GatedCrossAttentionFusion

    fusion = GatedCrossAttentionFusion(dim=256, num_heads=8)
    fused, gate_map = fusion(cnn_tokens, vit_tokens)
    # fused: (B, N_cnn, dim) -- fused representation aligned to CNN token grid
    # gate_map: (B, N_cnn, dim) -- per-token, per-channel gate values (for viz)
"""

import torch
import torch.nn as nn


class GatedCrossAttentionFusion(nn.Module):
    """
    Full CSAF module: bidirectional cross-attention + learned gate.

    CNN tokens (N_cnn, e.g. 49) and ViT tokens (N_vit, e.g. 196) can have
    different sequence lengths -- MultiheadAttention handles this natively
    since Q and K/V don't need matching sequence length, only matching dim.

    Output is aligned to the CNN token grid (N_cnn tokens) since that's
    typically the smaller, spatially-meaningful grid for a classification head.
    Change `query_grid` to 'vit' if you'd rather align to the ViT grid.
    """

    def __init__(self, dim=256, num_heads=8, query_grid="cnn", dropout=0.1):
        super().__init__()
        assert query_grid in ("cnn", "vit")
        self.query_grid = query_grid

        self.cross_attn_c2v = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.cross_attn_v2c = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )

        self.norm_c2v = nn.LayerNorm(dim)
        self.norm_v2c = nn.LayerNorm(dim)

        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, cnn_tokens, vit_tokens):
        """
        cnn_tokens: (B, N_cnn, dim)
        vit_tokens: (B, N_vit, dim)
        """
        # CNN queries ViT -> pulls in global context, output aligned to CNN grid
        attn_c2v, _ = self.cross_attn_c2v(cnn_tokens, vit_tokens, vit_tokens)
        attn_c2v = self.norm_c2v(attn_c2v + cnn_tokens)   # residual, (B, N_cnn, dim)

        # ViT queries CNN -> pulls in local detail, output aligned to ViT grid
        attn_v2c, _ = self.cross_attn_v2c(vit_tokens, cnn_tokens, cnn_tokens)
        attn_v2c = self.norm_v2c(attn_v2c + vit_tokens)   # residual, (B, N_vit, dim)

        if self.query_grid == "cnn":
            # need attn_v2c aligned to N_cnn tokens to combine with attn_c2v
            # pool ViT-branch output down to N_cnn tokens via adaptive pooling
            attn_v2c_aligned = self._align_tokens(attn_v2c, target_len=attn_c2v.shape[1])
            branch_a, branch_b = attn_c2v, attn_v2c_aligned
        else:
            attn_c2v_aligned = self._align_tokens(attn_c2v, target_len=attn_v2c.shape[1])
            branch_a, branch_b = attn_c2v_aligned, attn_v2c

        gate_input = torch.cat([branch_a, branch_b], dim=-1)   # (B, N, 2*dim)
        gate = self.gate_fc(gate_input)                         # (B, N, dim), in [0,1]

        fused = gate * branch_a + (1 - gate) * branch_b
        return fused, gate

    @staticmethod
    def _align_tokens(tokens, target_len):
        """
        Adaptive pool along the token dimension to match target_len.
        tokens: (B, N, dim) -> (B, target_len, dim)
        """
        B, N, D = tokens.shape
        if N == target_len:
            return tokens
        tokens_t = tokens.transpose(1, 2)          # (B, dim, N)
        pooled = nn.functional.adaptive_avg_pool1d(tokens_t, target_len)  # (B, dim, target_len)
        return pooled.transpose(1, 2)              # (B, target_len, dim)


# ---------------------------------------------------------------------------
# Ablation variants -- same interface (cnn_tokens, vit_tokens) -> fused tokens
# Used by run_ablations.py to swap fusion strategy while keeping everything
# else (backbones, classification head, training loop) identical.
# ---------------------------------------------------------------------------

class ConcatFusion(nn.Module):
    """Baseline: simple concatenation + linear projection, no attention."""

    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, cnn_tokens, vit_tokens):
        cnn_pooled = cnn_tokens.mean(dim=1)   # (B, dim)
        vit_pooled = vit_tokens.mean(dim=1)   # (B, dim)
        fused = self.proj(torch.cat([cnn_pooled, vit_pooled], dim=-1))  # (B, dim)
        return fused.unsqueeze(1), None       # (B, 1, dim) to match interface


class UngatedCrossAttentionFusion(nn.Module):
    """Cross-attention WITHOUT the gate -- simple addition of both branches.
    This isolates exactly what the gate contributes (your key ablation)."""

    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn_c2v = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.cross_attn_v2c = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm_c2v = nn.LayerNorm(dim)
        self.norm_v2c = nn.LayerNorm(dim)

    def forward(self, cnn_tokens, vit_tokens):
        attn_c2v, _ = self.cross_attn_c2v(cnn_tokens, vit_tokens, vit_tokens)
        attn_c2v = self.norm_c2v(attn_c2v + cnn_tokens)

        attn_v2c, _ = self.cross_attn_v2c(vit_tokens, cnn_tokens, cnn_tokens)
        attn_v2c = self.norm_v2c(attn_v2c + vit_tokens)

        attn_v2c_aligned = GatedCrossAttentionFusion._align_tokens(
            attn_v2c, target_len=attn_c2v.shape[1]
        )
        fused = attn_c2v + attn_v2c_aligned   # simple addition, no gate
        return fused, None


if __name__ == "__main__":
    # smoke test with realistic shapes from backbones.py
    B, N_cnn, N_vit, dim = 2, 49, 196, 256
    cnn_tokens = torch.randn(B, N_cnn, dim)
    vit_tokens = torch.randn(B, N_vit, dim)

    print("Testing GatedCrossAttentionFusion (full CSAF)...")
    csaf = GatedCrossAttentionFusion(dim=dim)
    fused, gate = csaf(cnn_tokens, vit_tokens)
    print(f"  fused shape: {fused.shape}")   # (2, 49, 256)
    print(f"  gate shape:  {gate.shape}")    # (2, 49, 256)
    print(f"  gate value range: [{gate.min().item():.3f}, {gate.max().item():.3f}]")

    print("\nTesting ConcatFusion (ablation baseline)...")
    concat = ConcatFusion(dim=dim)
    fused_c, _ = concat(cnn_tokens, vit_tokens)
    print(f"  fused shape: {fused_c.shape}")   # (2, 1, 256)

    print("\nTesting UngatedCrossAttentionFusion (ablation)...")
    ungated = UngatedCrossAttentionFusion(dim=dim)
    fused_u, _ = ungated(cnn_tokens, vit_tokens)
    print(f"  fused shape: {fused_u.shape}")   # (2, 49, 256)

    print("\nAll fusion variants ran successfully.")
