"""
evaluate_wide_view.py
Quantitative evaluation of the Wide View (sliding-window) pipeline on
synthetic canvases built from real SDNET2018 test crops with known
ground truth. Supports single-model and ensemble modes.

Usage:
    python evaluate_wide_view.py --n_positive 30 --n_negative 30
    python evaluate_wide_view.py --n_positive 30 --n_negative 30 --ensemble --threshold 0.9
"""

import os
import argparse
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

from sliding_window_detect import (
    sliding_window_detect, ensemble_sliding_window_detect, iou, get_device,
)

random.seed(42)
np.random.seed(42)

USE_ENSEMBLE = False
MIN_AGREE = 2


def detect(canvas, variant, stride, threshold, use_wall_filter,
           wall_texture_std_threshold, device):
    if USE_ENSEMBLE:
        return ensemble_sliding_window_detect(
            canvas, stride=stride, threshold=threshold, min_agree=MIN_AGREE,
            use_wall_filter=use_wall_filter,
            wall_texture_std_threshold=wall_texture_std_threshold, device=device,
        )
    return sliding_window_detect(
        canvas, variant=variant, stride=stride, threshold=threshold,
        use_wall_filter=use_wall_filter,
        wall_texture_std_threshold=wall_texture_std_threshold, device=device,
    )


def make_positive_canvas(data_dir="dataset", canvas_size=900, tile=224):
    test_ds = ImageFolder(os.path.join(data_dir, "test"))
    class_names = test_ds.classes
    pos_idx = class_names.index("Positive")
    neg_idx = class_names.index("Negative")
    pos_paths = [p for p, l in test_ds.samples if l == pos_idx]
    neg_paths = [p for p, l in test_ds.samples if l == neg_idx]

    canvas = Image.new("RGB", (canvas_size, canvas_size))
    for ty in range(0, canvas_size, tile):
        for tx in range(0, canvas_size, tile):
            neg_img = Image.open(random.choice(neg_paths)).convert("RGB").resize((tile, tile))
            canvas.paste(neg_img, (tx, ty))

    pos_img = Image.open(random.choice(pos_paths)).convert("RGB").resize((tile, tile))
    px = random.randint(0, (canvas_size // tile) - 1) * tile
    py = random.randint(0, (canvas_size // tile) - 1) * tile
    canvas.paste(pos_img, (px, py))

    return canvas, (px, py, px + tile, py + tile), pos_paths, neg_paths


def make_negative_canvas(neg_paths, canvas_size=900, tile=224):
    canvas = Image.new("RGB", (canvas_size, canvas_size))
    for ty in range(0, canvas_size, tile):
        for tx in range(0, canvas_size, tile):
            neg_img = Image.open(random.choice(neg_paths)).convert("RGB").resize((tile, tile))
            canvas.paste(neg_img, (tx, ty))
    return canvas


def evaluate_positive_canvases(n, variant, stride, threshold, iou_threshold,
                                use_wall_filter, wall_texture_std_threshold, device):
    results = []
    neg_paths_cache = None
    for i in range(n):
        canvas, true_box, pos_paths, neg_paths = make_positive_canvas()
        if neg_paths_cache is None:
            neg_paths_cache = neg_paths

        _, detections = detect(
            canvas, variant, stride, threshold,
            use_wall_filter, wall_texture_std_threshold, device,
        )

        best_iou, best_conf = 0.0, None
        false_positives = 0
        near_crack = 0
        for d in detections:
            box_iou = iou(d["box"], true_box)
            if box_iou > iou_threshold:
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_conf = d["confidence"]
            elif box_iou > 0:
                near_crack += 1
            else:
                false_positives += 1

        found = best_iou > iou_threshold
        results.append({
            "found": found,
            "confidence": best_conf if found else None,
            "false_positives": false_positives,
            "near_crack": near_crack,
            "total_detections": len(detections),
        })
        print(f"  [{i+1}/{n}] {'FOUND' if found else 'MISSED'}"
              + (f" (conf={best_conf*100:.1f}%)" if found else "")
              + f", {false_positives} false positive(s), {near_crack} partial-overlap")

    return results, neg_paths_cache


def evaluate_negative_canvases(n, neg_paths, variant, stride, threshold,
                                use_wall_filter, wall_texture_std_threshold, device):
    results = []
    for i in range(n):
        canvas = make_negative_canvas(neg_paths)
        _, detections = detect(
            canvas, variant, stride, threshold,
            use_wall_filter, wall_texture_std_threshold, device,
        )
        results.append({"false_positives": len(detections)})
        print(f"  [{i+1}/{n}] {len(detections)} false positive(s) on pure-negative canvas")

    return results


def plot_summary(pos_results, neg_results, out_path="figures/wide_view_evaluation.png"):
    n_pos = len(pos_results)
    n_found = sum(1 for r in pos_results if r["found"])
    detection_rate = n_found / n_pos if n_pos else 0.0
    avg_conf = np.mean([r["confidence"] for r in pos_results if r["found"]]) if n_found else 0.0
    avg_fp_positive = np.mean([r["false_positives"] for r in pos_results]) if n_pos else 0.0
    avg_fp_negative = np.mean([r["false_positives"] for r in neg_results]) if neg_results else 0.0
    avg_near = np.mean([r["near_crack"] for r in pos_results]) if n_pos else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].bar(["Detected", "Missed"], [n_found, n_pos - n_found],
                color=["#2563EB", "#94A3B8"])
    axes[0].set_title(f"Detection Rate: {detection_rate*100:.1f}%")
    axes[0].set_ylabel("Number of canvases")

    axes[1].bar(["Positive\ncanvases", "Negative\ncanvases"],
                [avg_fp_positive, avg_fp_negative], color=["#F97316", "#EF4444"])
    axes[1].set_title("Avg False Positives per Image")
    axes[1].set_ylabel("Mean extra (non-overlapping) detections")

    confidences = [r["confidence"] for r in pos_results if r["found"]]
    if confidences:
        axes[2].hist(confidences, bins=10, color="#22C55E", edgecolor="white")
        axes[2].set_title(f"Confidence on Correct Detections\n(mean {avg_conf*100:.1f}%)")
        axes[2].set_xlabel("Confidence")
        axes[2].set_ylabel("Count")
    else:
        axes[2].text(0.5, 0.5, "No correct detections", ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    return {
        "detection_rate": detection_rate,
        "avg_confidence_when_found": float(avg_conf),
        "avg_false_positives_per_positive_canvas": float(avg_fp_positive),
        "avg_false_positives_per_negative_canvas": float(avg_fp_negative),
        "avg_near_crack": float(avg_near),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="full_csaf")
    parser.add_argument("--n_positive", type=int, default=20)
    parser.add_argument("--n_negative", type=int, default=20)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--wall_filter", action="store_true", default=True)
    parser.add_argument("--wall_strictness", type=float, default=0.0)
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble (full_csaf + cnn_only + vit_only) instead of one model.")
    parser.add_argument("--min_agree", type=int, default=2,
                        help="Ensemble mode: how many variants must agree.")
    parser.add_argument("--out", type=str, default="figures/wide_view_evaluation.png")
    args = parser.parse_args()

    USE_ENSEMBLE = args.ensemble
    MIN_AGREE = args.min_agree

    device = get_device()
    mode = f"ensemble (min_agree={args.min_agree})" if args.ensemble else f"single ({args.variant})"
    print(f"Using device: {device}")
    print(f"Mode: {mode}  |  threshold: {args.threshold}  |  "
          f"wall_filter: {args.wall_filter} (strictness {args.wall_strictness})\n")

    print(f"Evaluating {args.n_positive} positive (crack-present) canvases...")
    pos_results, neg_paths = evaluate_positive_canvases(
        args.n_positive, args.variant, args.stride, args.threshold, args.iou_threshold,
        args.wall_filter, args.wall_strictness, device,
    )

    print(f"\nEvaluating {args.n_negative} negative (no-crack) canvases...")
    neg_results = evaluate_negative_canvases(
        args.n_negative, neg_paths, args.variant, args.stride, args.threshold,
        args.wall_filter, args.wall_strictness, device,
    )

    summary = plot_summary(pos_results, neg_results, out_path=args.out)

    print("\n" + "=" * 50)
    print("WIDE VIEW EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Mode: {mode}, threshold {args.threshold}")
    print(f"Detection rate (found real crack):        {summary['detection_rate']*100:.1f}%")
    print(f"Avg confidence when found:                {summary['avg_confidence_when_found']*100:.1f}%")
    print(f"Avg false positives / positive canvas:    {summary['avg_false_positives_per_positive_canvas']:.2f}")
    print(f"Avg false positives / negative canvas:    {summary['avg_false_positives_per_negative_canvas']:.2f}")
    print(f"Avg partial-overlap boxes / positive canvas: {summary['avg_near_crack']:.2f}")