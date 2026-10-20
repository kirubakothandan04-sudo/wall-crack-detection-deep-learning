import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scipy.stats import binomtest
from sklearn.metrics import f1_score, recall_score, precision_score

from csaf_net import build_model, get_device
from sliding_window_detect import get_transform

p = argparse.ArgumentParser()
p.add_argument("--full", default="full_csaf")
p.add_argument("--others", nargs="+",
               default=["ungated_cross_attn", "scalar_gate", "concat", "cnn_only"])
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument("--boots", type=int, default=2000)
a = p.parse_args()

dev = get_device()
ds = ImageFolder("dataset/test", transform=get_transform())
pos_idx = ds.class_to_idx["Positive"]
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
y = np.array([1 if l == pos_idx else 0 for _, l in ds.samples])


def predict(variant):
    m = build_model(variant=variant).to(dev)
    m.load_state_dict(torch.load(f"checkpoints/{variant}_best.pth", map_location=dev))
    m.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            logits, _ = m(x.to(dev))
            out.append(torch.sigmoid(logits).view(-1).cpu().numpy())
    return (np.concatenate(out) > a.threshold).astype(int)


preds = {v: predict(v) for v in [a.full] + a.others}
rng = np.random.default_rng(42)
n = len(y)
idx = [rng.integers(0, n, n) for _ in range(a.boots)]

print(f"Test images: {n}  |  threshold: {a.threshold}\n")
for v, pr in preds.items():
    print(f"{v:20s} F1={f1_score(y, pr)*100:.2f}  "
          f"Recall={recall_score(y, pr)*100:.2f}  Precision={precision_score(y, pr)*100:.2f}")

print("\nPaired comparison: full model vs each other variant")
full = preds[a.full]
for v in a.others:
    o = preds[v]
    b = int(((full == y) & (o != y)).sum())
    c = int(((full != y) & (o == y)).sum())
    pval = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    d_f1 = [f1_score(y[i], full[i]) - f1_score(y[i], o[i]) for i in idx]
    d_rc = [recall_score(y[i], full[i]) - recall_score(y[i], o[i]) for i in idx]
    print(f"\n{a.full} vs {v}")
    print(f"  full right/other wrong: {b}   full wrong/other right: {c}   McNemar p = {pval:.4f}")
    print(f"  F1 diff     {np.mean(d_f1)*100:+.2f} pts  95% CI [{np.percentile(d_f1,2.5)*100:+.2f}, {np.percentile(d_f1,97.5)*100:+.2f}]")
    print(f"  Recall diff {np.mean(d_rc)*100:+.2f} pts  95% CI [{np.percentile(d_rc,2.5)*100:+.2f}, {np.percentile(d_rc,97.5)*100:+.2f}]")
