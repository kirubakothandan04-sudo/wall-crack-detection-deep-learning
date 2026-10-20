"""
evaluate_wide_view_v2.py
Wide View benchmark v2. Differences from v1:
  - cracks are pasted at random PIXEL positions (not on the 224 grid)
  - crack size is random (--min_size to --max_size) instead of fixed 224
  - the confidence threshold is CHOSEN ON VALIDATION scenes, frozen, and the
    TEST scenes are scored once at the frozen threshold
Every window is scored once and cached, so threshold sweeps are cheap.
Box logic (IoU, NMS) is imported from sliding_window_detect.py.
The wall filter is skipped: at strictness 0 it accepts every window.

Usage (after training has finished):
    export HF_HUB_OFFLINE=1
    python evaluate_wide_view_v2.py --n_val 5 --n_test 5 --out_dir wide_v2_smoke
    python evaluate_wide_view_v2.py
"""

import os
import json
import math
import pickle
import time
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder

from csaf_net import build_model, get_device
from sliding_window_detect import get_transform, iou, non_max_suppression

WIN = 224
ENSEMBLE = ("full_csaf", "cnn_only", "vit_only")
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]


def list_paths(data_dir, split):
    ds = ImageFolder(os.path.join(data_dir, split))
    pos_i, neg_i = ds.class_to_idx["Positive"], ds.class_to_idx["Negative"]
    pos = [p for p, l in ds.samples if l == pos_i]
    neg = [p for p, l in ds.samples if l == neg_i]
    return pos, neg


def window_positions(size, stride):
    """Same window set as sliding_window_detect.py."""
    pos = set()
    for y in range(0, size - WIN + 1, stride):
        for x in range(0, size - WIN + 1, stride):
            pos.add((x, y))
    if (size - WIN) % stride != 0:
        for y in list(range(0, size - WIN + 1, stride)) + [size - WIN]:
            pos.add((size - WIN, y))
    return sorted(pos, key=lambda p: (p[1], p[0]))


def make_canvas(rng, pos_paths, neg_paths, with_crack, args):
    size = args.canvas
    canvas = Image.new("RGB", (size, size))
    for ty in range(0, size, WIN):
        for tx in range(0, size, WIN):
            tile = Image.open(rng.choice(neg_paths)).convert("RGB").resize((WIN, WIN))
            canvas.paste(tile, (tx, ty))
    true_box = None
    if with_crack:
        s = rng.randint(args.min_size, args.max_size)
        crack = Image.open(rng.choice(pos_paths)).convert("RGB").resize((s, s))
        px = rng.randint(0, size - s)
        py = rng.randint(0, size - s)
        canvas.paste(crack, (px, py))
        true_box = (px, py, px + s, py + s)
    return canvas, true_box


def load_models(variants, ckpt_dir, device):
    models = {}
    for v in variants:
        m = build_model(variant=v).to(device)
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{v}_best.pth"),
                                     map_location=device))
        m.eval()
        models[v] = m
    return models


@torch.no_grad()
def score_canvas(canvas, models, positions, transform, device, batch=32):
    X = torch.stack([transform(canvas.crop((x, y, x + WIN, y + WIN)))
                     for x, y in positions])
    out = {v: [] for v in models}
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        for v, m in models.items():
            logits, _ = m(xb)
            out[v].append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
    return {v: np.concatenate(a) for v, a in out.items()}


def build_and_score(split, n, args, models, positions, transform, device):
    pos_paths, neg_paths = list_paths(args.data_dir, split)
    rng = random.Random(args.seed * 1000 + (1 if split == "val" else 2))
    plan = [True] * n + [False] * n
    scenes, t0 = [], time.time()
    for i, with_crack in enumerate(plan, 1):
        canvas, true_box = make_canvas(rng, pos_paths, neg_paths, with_crack, args)
        scores = score_canvas(canvas, models, positions, transform, device, args.batch)
        scenes.append({"true_box": true_box, "scores": scores})
        if i % 10 == 0 or i == len(plan):
            print(f"  [{split}] {i}/{len(plan)} scenes scored ({time.time() - t0:.0f}s)")
    return scenes


def get_scenes(split, n, args, models, positions, transform, device):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"scores_{split}.pkl")
    meta = {"split": split, "n": n, "seed": args.seed, "stride": args.stride,
            "canvas": args.canvas, "min_size": args.min_size,
            "max_size": args.max_size, "variants": sorted(models)}
    if args.reuse and os.path.exists(path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob["meta"] == meta:
            print(f"Reusing cached scores: {path}")
            return blob["scenes"]
    scenes = build_and_score(split, n, args, models, positions, transform, device)
    with open(path, "wb") as f:
        pickle.dump({"meta": meta, "scenes": scenes}, f)
    return scenes


def detections_for(scores, positions, mode, thr, min_agree):
    boxes, sc = [], []
    if mode == "single":
        p = scores["full_csaf"]
        for k, (x, y) in enumerate(positions):
            if p[k] > thr:
                boxes.append((x, y, x + WIN, y + WIN))
                sc.append(float(p[k]))
    else:
        stack = np.stack([scores[v] for v in ENSEMBLE], axis=0)
        for k, (x, y) in enumerate(positions):
            col = stack[:, k]
            agree = col > thr
            if agree.sum() >= min_agree:
                boxes.append((x, y, x + WIN, y + WIN))
                sc.append(float(col[agree].mean()))
    keep = non_max_suppression(boxes, sc, iou_threshold=0.3)
    return [(boxes[i], sc[i]) for i in keep]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def evaluate(scenes, positions, mode, thr, args):
    n_pos = n_neg = hits = 0
    ious_found, clean_boxes = [], []
    off_total = partial_total = 0
    for s in scenes:
        dets = detections_for(s["scores"], positions, mode, thr, args.min_agree)
        if s["true_box"] is None:
            n_neg += 1
            clean_boxes.append(len(dets))
            continue
        n_pos += 1
        best, partial, off = 0.0, 0, 0
        for box, _ in dets:
            v = iou(box, s["true_box"])
            best = max(best, v)
            if v > args.iou:
                continue
            if v > 0:
                partial += 1
            else:
                off += 1
        if best > args.iou:
            hits += 1
            ious_found.append(best)
        off_total += off
        partial_total += partial
    fp_scenes = sum(1 for b in clean_boxes if b > 0)
    precision = hits / (hits + fp_scenes) if (hits + fp_scenes) > 0 else 0.0
    recall = hits / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    lo, hi = wilson(hits, n_pos)
    return {
        "thr": float(thr), "recall": float(recall), "lo": float(lo), "hi": float(hi),
        "fp_clean": float(np.mean(clean_boxes)) if clean_boxes else 0.0,
        "fp_crack": off_total / n_pos if n_pos else 0.0,
        "partial": partial_total / n_pos if n_pos else 0.0,
        "precision": float(precision), "f1": float(f1),
        "mean_iou_found": float(np.mean(ious_found)) if ious_found else 0.0,
        "n_pos": int(n_pos), "n_neg": int(n_neg), "hits": int(hits),
    }


def pick_threshold(rows, rule, fp_budget):
    if rule == "f1":
        return max(rows, key=lambda r: (r["f1"], r["thr"]))
    feasible = [r for r in rows if r["fp_clean"] <= fp_budget]
    if feasible:
        return max(feasible, key=lambda r: (r["recall"], r["thr"]))
    return min(rows, key=lambda r: (r["fp_clean"], -r["thr"]))


def print_table(title, rows, chosen_thr=None):
    print(f"\n{title}")
    print("  thr  recall%  95% CI      FP/clean  FP/crack  partial  F1%")
    for r in rows:
        mark = "  <-- chosen" if chosen_thr is not None and abs(r["thr"] - chosen_thr) < 1e-9 else ""
        print(f"{r['thr']:5.3f}  {r['recall'] * 100:6.1f}  [{r['lo'] * 100:3.0f},{r['hi'] * 100:3.0f}]    "
              f"{r['fp_clean']:6.2f}    {r['fp_crack']:6.2f}   {r['partial']:6.2f}  {r['f1'] * 100:5.1f}{mark}")


def print_test(mode, r):
    print(f"\nTEST ({mode}), frozen threshold {r['thr']:.3f}, "
          f"{r['n_pos']} crack + {r['n_neg']} clean scenes")
    print(f"  Cracks localized:               {r['hits']}/{r['n_pos']} = {r['recall'] * 100:.1f}%"
          f"  (95% CI {r['lo'] * 100:.0f}-{r['hi'] * 100:.0f}%)")
    print(f"  Mean IoU on localized cracks:   {r['mean_iou_found']:.3f}")
    print(f"  False boxes / clean scene:      {r['fp_clean']:.2f}")
    print(f"  Off-crack boxes / crack scene:  {r['fp_crack']:.2f}")
    print(f"  Partial-overlap boxes / crack scene: {r['partial']:.2f}")
    print(f"  Scene precision / F1:           {r['precision'] * 100:.1f}% / {r['f1'] * 100:.1f}%")


def plot_sweeps(sweeps, chosen, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, rows in sweeps.items():
        t = [r["thr"] for r in rows]
        axes[0].plot(t, [r["recall"] * 100 for r in rows], marker="o", label=mode)
        axes[1].plot(t, [r["fp_clean"] for r in rows], marker="o", label=mode)
        axes[0].axvline(chosen[mode], ls="--", alpha=0.4)
        axes[1].axvline(chosen[mode], ls="--", alpha=0.4)
    axes[0].set_title("Validation: cracks localized (%)")
    axes[1].set_title("Validation: false boxes per clean scene")
    for ax in axes:
        ax.set_xlabel("Confidence threshold")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--out_dir", default="wide_v2_results")
    p.add_argument("--n_val", type=int, default=50, help="scenes per class, validation")
    p.add_argument("--n_test", type=int, default=100, help="scenes per class, test")
    p.add_argument("--canvas", type=int, default=900)
    p.add_argument("--stride", type=int, default=56)
    p.add_argument("--min_size", type=int, default=150)
    p.add_argument("--max_size", type=int, default=300)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--min_agree", type=int, default=2)
    p.add_argument("--modes", nargs="+", default=["single", "ensemble"],
                   choices=["single", "ensemble"])
    p.add_argument("--select", choices=["budget", "f1"], default="budget",
                   help="rule for choosing the threshold on validation")
    p.add_argument("--fp_budget", type=float, default=0.2,
                   help="budget rule: max false boxes per clean scene")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--reuse", action="store_true", help="reuse cached window scores")
    args = p.parse_args()

    device = get_device()
    variants = ENSEMBLE if "ensemble" in args.modes else ("full_csaf",)
    models = load_models(variants, args.checkpoint_dir, device)
    positions = window_positions(args.canvas, args.stride)
    transform = get_transform()

    rule = (f"max recall with <= {args.fp_budget} false boxes per clean scene"
            if args.select == "budget" else "max scene-level F1")
    print(f"Device: {device} | windows per scene: {len(positions)} | stride {args.stride}")
    print(f"Crack size {args.min_size}-{args.max_size}px, canvas {args.canvas}px, "
          f"match IoU > {args.iou}")
    print(f"Threshold rule (decided before running): {rule}")

    print("\nScoring VALIDATION scenes...")
    val = get_scenes("val", args.n_val, args, models, positions, transform, device)

    sweeps, chosen = {}, {}
    for mode in args.modes:
        rows = [evaluate(val, positions, mode, t, args) for t in THRESHOLDS]
        best = pick_threshold(rows, args.select, args.fp_budget)
        sweeps[mode], chosen[mode] = rows, best["thr"]
        print_table(f"VALIDATION sweep ({mode})", rows, best["thr"])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "frozen_thresholds.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    plot_sweeps(sweeps, chosen, os.path.join(args.out_dir, "validation_sweep.png"))
    print(f"\nThresholds frozen: {chosen}")

    print("\nScoring TEST scenes (evaluated once, at the frozen thresholds)...")
    test = get_scenes("test", args.n_test, args, models, positions, transform, device)

    summary = {"settings": vars(args), "frozen_thresholds": chosen, "modes": {}}
    for mode in args.modes:
        r = evaluate(test, positions, mode, chosen[mode], args)
        print_test(mode, r)
        summary["modes"][mode] = {"val_sweep": sweeps[mode], "test": r}

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {args.out_dir}/")


if __name__ == "__main__":
    main()