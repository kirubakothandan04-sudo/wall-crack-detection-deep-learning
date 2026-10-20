"""Measure CSAF-Net accuracy on REAL photos (not SDNET2018).

Folder layout (put whole photos here, never used for training):
    real_test/crack/*.jpg    photos that contain a crack
    real_test/clean/*.jpg    photos with NO crack (include windows, frames,
                             shadows, stains, pipes: the hard cases)

Run from the WallCrackDetection folder:
    python eval_real_photos.py                 # default threshold 0.5
    python eval_real_photos.py --thr 0.9       # try stricter thresholds

How it scores: each photo is cut into 224x224 tiles (stride 112). A photo is
called "crack" if at least --min-tiles tiles score above the threshold.
Two numbers are reported: PHOTO level and TILE level (tile level only for
clean photos, where every tile is known to be a non-crack).
Writes real_photo_results.csv with every photo's result.
"""
import argparse, csv, glob, os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from predict_csaf import load_model          # <-- CHANGE if load_model lives elsewhere

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="real_test")
ap.add_argument("--variant", default="full_csaf")
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--min-tiles", type=int, default=1)
ap.add_argument("--stride", type=int, default=112)
a = ap.parse_args()

model, device = load_model(variant=a.variant)
model.eval()
prep = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def logit_of(out):
    if isinstance(out, (tuple, list)):
        for o in out:
            if torch.is_tensor(o) and o.numel() == 1:
                return o.reshape(-1)[0]
        out = out[0]
    return out.reshape(-1)[0]


@torch.no_grad()
def tile_probs(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) < 224:                       # upscale small photos
        s = 224 / min(w, h)
        img = img.resize((int(w * s) + 1, int(h * s) + 1))
        w, h = img.size
    xs = list(range(0, w - 224 + 1, a.stride)) or [0]
    ys = list(range(0, h - 224 + 1, a.stride)) or [0]
    batch, n = [], []
    probs = []
    for y in ys:
        for x in xs:
            batch.append(prep(img.crop((x, y, x + 224, y + 224))))
    for t in batch:                            # one by one: safe on a MacBook Air
        p = torch.sigmoid(logit_of(model(t.unsqueeze(0).to(device)))).item()
        probs.append(p)
    return np.array(probs)


rows = []
for label in ("crack", "clean"):
    for f in sorted(glob.glob(os.path.join(a.root, label, "*"))):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        p = tile_probs(f)
        n_hit = int((p > a.thr).sum())
        pred = "crack" if n_hit >= a.min_tiles else "clean"
        rows.append(dict(file=os.path.basename(f), truth=label, pred=pred,
                         tiles=len(p), tiles_over=n_hit, max_prob=round(float(p.max()), 3)))
        print(f"{label:5s} -> {pred:5s}  tiles>{a.thr}: {n_hit}/{len(p)}  max {p.max():.3f}  {os.path.basename(f)}")

if not rows:
    raise SystemExit(f"No images found under {a.root}/crack and {a.root}/clean")

tp = sum(r["truth"] == "crack" and r["pred"] == "crack" for r in rows)
fn = sum(r["truth"] == "crack" and r["pred"] == "clean" for r in rows)
fp = sum(r["truth"] == "clean" and r["pred"] == "crack" for r in rows)
tn = sum(r["truth"] == "clean" and r["pred"] == "clean" for r in rows)
acc = (tp + tn) / len(rows)
rec = tp / (tp + fn) if tp + fn else float("nan")
prec = tp / (tp + fp) if tp + fp else float("nan")
f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
clean_tiles = sum(r["tiles"] for r in rows if r["truth"] == "clean")
clean_hits = sum(r["tiles_over"] for r in rows if r["truth"] == "clean")

print("\n===== REAL-PHOTO RESULT (threshold %.2f, min tiles %d) =====" % (a.thr, a.min_tiles))
print(f"photos: {len(rows)}  (crack {tp+fn}, clean {fp+tn})")
print(f"TP {tp}  FN {fn}  FP {fp}  TN {tn}")
print(f"accuracy {acc:.3f}  recall {rec:.3f}  precision {prec:.3f}  F1 {f1:.3f}")
if clean_tiles:
    print(f"false-alarm rate on clean tiles: {clean_hits}/{clean_tiles} = {clean_hits/clean_tiles:.3f}")
print("Report this next to the SDNET2018 test result as a SEPARATE number.")

with open("real_photo_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("saved real_photo_results.csv")