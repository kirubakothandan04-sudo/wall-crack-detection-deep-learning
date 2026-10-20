import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from csaf_net import build_model, get_device
from sliding_window_detect import get_transform

dev = get_device()
ds = ImageFolder("dataset/test", transform=get_transform())
pos_idx = ds.class_to_idx["Positive"]
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
y = np.array([1 if l == pos_idx else 0 for _, l in ds.samples])

model = build_model(variant="full_csaf").to(dev)
model.load_state_dict(torch.load("checkpoints/full_csaf_best.pth", map_location=dev))
model.eval()

img_mean, std_tok, std_ch, maps = [], [], [], []
first = True
with torch.no_grad():
    for x, _ in loader:
        out = model(x.to(dev))
        gate = out[1]
        if gate is None:
            raise SystemExit("full_csaf returned no gate map.")
        g = gate.detach().cpu().numpy()
        if first:
            print("Gate tensor shape:", g.shape)
            first = False
        if g.ndim == 2:
            g = g[:, None, :]
        img_mean.append(g.mean(axis=(1, 2)))
        std_tok.append(g.mean(axis=2).std(axis=1))   # variation across tokens
        std_ch.append(g.mean(axis=1).std(axis=1))    # variation across channels
        maps.append(g.mean(axis=2))                  # (B, N)

img_mean = np.concatenate(img_mean)
std_tok = np.concatenate(std_tok)
std_ch = np.concatenate(std_ch)
maps = np.concatenate(maps)

print(f"\nTest images: {len(y)} (crack {int(y.sum())}, clean {int((1 - y).sum())})")
print("Gate value near 1 = trusts the CNN-attended branch; near 0 = trusts the ViT-attended branch\n")
print(f"Overall mean gate: {img_mean.mean():.3f}   std across images: {img_mean.std():.3f}")
print(f"Mean gate, crack images: {img_mean[y == 1].mean():.3f}   clean images: {img_mean[y == 0].mean():.3f}")
u = mannwhitneyu(img_mean[y == 1], img_mean[y == 0], alternative="two-sided")
print(f"Mann-Whitney p (crack vs clean): {u.pvalue:.2e}")
print(f"AUC of per-image mean gate as a crack predictor: {roc_auc_score(y, img_mean):.3f}  (0.5 = no information)")
print(f"Variation inside one image: across tokens {std_tok.mean():.3f}, across channels {std_ch.mean():.3f}")

try:
    sd = torch.load("checkpoints/scalar_gate_best.pth", map_location="cpu")
    for k, v in sd.items():
        if "gate_param" in k:
            print(f"\nScalar-gate variant learned one fixed value for everything: {torch.sigmoid(v).item():.3f}")
except Exception as e:
    print("\nCould not read the scalar_gate checkpoint:", e)

N = maps.shape[1]
side = int(round(N ** 0.5))
spatial = side * side == N
fig, axes = plt.subplots(1, 3 if spatial else 1, figsize=(14, 4))
axes = np.atleast_1d(axes)
axes[0].hist(img_mean[y == 0], bins=30, alpha=0.7, label="No crack")
axes[0].hist(img_mean[y == 1], bins=30, alpha=0.7, label="Crack")
axes[0].set_title("Mean gate per image")
axes[0].set_xlabel("Gate (1 = CNN branch, 0 = ViT branch)")
axes[0].legend()
if spatial:
    m0 = maps[y == 0].mean(axis=0).reshape(side, side)
    m1 = maps[y == 1].mean(axis=0).reshape(side, side)
    vmin, vmax = min(m0.min(), m1.min()), max(m0.max(), m1.max())
    for ax, m, name in zip(axes[1:], (m0, m1), ("No crack", "Crack")):
        im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"Average gate map: {name}")
        ax.axis("off")
    fig.colorbar(im, ax=list(axes[1:]), shrink=0.8)
fig.savefig("figures/gate_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved: figures/gate_analysis.png")
