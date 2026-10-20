import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import average_precision_score, precision_recall_curve

from csaf_net import build_model, get_device
from sliding_window_detect import get_transform

p = argparse.ArgumentParser()
p.add_argument("--full", default="full_csaf")
p.add_argument("--others", nargs="+", default=["ungated_cross_attn", "scalar_gate", "concat"])
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument("--boots", type=int, default=1000)
a = p.parse_args()

dev = get_device()
ds = ImageFolder("dataset/test", transform=get_transform())
pos_idx = ds.class_to_idx["Positive"]
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
y = np.array([1 if l == pos_idx else 0 for _, l in ds.samples])


def probs(variant):
    m = build_model(variant=variant).to(dev)
    m.load_state_dict(torch.load(f"checkpoints/{variant}_best.pth", map_location=dev))
    m.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            logits, _ = m(x.to(dev))
            out.append(torch.sigmoid(logits).view(-1).cpu().numpy())
    return np.concatenate(out)


def prec_at_recall(yb, s, target):
    prec, rec, _ = precision_recall_curve(yb, s)
    ok = prec[rec >= target]
    return float(ok.max()) if ok.size else 0.0


def op_point(yb, s, th):
    pred = s > th
    tp = int((pred & (yb == 1)).sum())
    fp = int((pred & (yb == 0)).sum())
    rec = tp / max(int((yb == 1).sum()), 1)
    prec = tp / max(tp + fp, 1)
    return tp, fp, rec, prec


def diffs(yb, sf, so, th):
    _, _, rec, prec = op_point(yb, sf, th)
    d_ap = average_precision_score(yb, sf) - average_precision_score(yb, so)
    d_prec = prec - prec_at_recall(yb, so, rec)
    return d_ap, d_prec


P = {v: probs(v) for v in [a.full] + a.others}
rng = np.random.default_rng(42)
n = len(y)

print(f"Test images: {n} (crack: {int((y==1).sum())}, clean: {int((y==0).sum())})  threshold: {a.threshold}\n")
for v, s in P.items():
    tp, fp, rec, prec = op_point(y, s, a.threshold)
    print(f"{v:20s} TP={tp} FP={fp} recall={rec*100:.2f} precision={prec*100:.2f}  AP={average_precision_score(y, s)*100:.2f}")

print("\nMatched-operating-point comparison (positive = full model better)")
for v in a.others:
    d_ap, d_prec = diffs(y, P[a.full], P[v], a.threshold)
    bs = [diffs(y[i], P[a.full][i], P[v][i], a.threshold)
          for i in (rng.integers(0, n, n) for _ in range(a.boots))]
    ap_b = [b[0] for b in bs]
    pr_b = [b[1] for b in bs]
    print(f"\n{a.full} vs {v}")
    print(f"  AP diff                      {d_ap*100:+.2f} pts  95% CI [{np.percentile(ap_b,2.5)*100:+.2f}, {np.percentile(ap_b,97.5)*100:+.2f}]")
    print(f"  Precision diff at same recall {d_prec*100:+.2f} pts  95% CI [{np.percentile(pr_b,2.5)*100:+.2f}, {np.percentile(pr_b,97.5)*100:+.2f}]")
