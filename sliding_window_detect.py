"""
sliding_window_detect.py
Two-stage crack detection for wide-angle / scene-level images.

CSAF-Net (and every variant) was trained on close-range, pre-cropped
224x224 wall patches (SDNET2018). It was never trained to localize a
crack within a large scene -- feeding it a whole-room photo directly
would squash any crack down to a handful of pixels after the mandatory
224x224 resize, and the model has never seen "room" context at all.

This script adds a simple localization stage in front of the existing,
untouched, already-trained classifier:
    1. Slide a 224x224 window across the input image at a given stride.
    2. Run the existing trained CSAF-Net (or any CNN-branch variant) on
       each window -- no retraining, no new data, same model as everywhere
       else in this project.
    3. Keep windows classified as Positive (crack) above `threshold`.
    4. Merge overlapping positive windows with simple greedy IoU-based
       non-max suppression.
    5. Draw the surviving boxes on the original image.

This is intentionally simple (grid-aligned boxes, not a trained detector)
-- it demonstrates that the *existing* trained classifier can be extended
to scene-level images via a localization wrapper, without requiring a
new bounding-box-annotated dataset (which SDNET2018 does not provide).
A learned object detector (e.g. YOLO) would give tighter boxes but needs
bounding-box annotations this project's dataset does not have -- see the
project's Future Work discussion.

Also included: ensemble_sliding_window_detect(), which runs SEVERAL
already-trained ablation variants (e.g. full_csaf, cnn_only, vit_only) on
each window and only accepts a detection when a minimum number of them
agree. Different architectures tend to be fooled by different things --
a window frame that confuses the fused model may not confuse the
single-backbone models, and vice versa -- so requiring agreement across
architecturally distinct models filters out some of the false positives
a single model would keep, without any retraining or new data.

Usage:
    python sliding_window_detect.py --image path/to/wide_photo.jpg
    python sliding_window_detect.py --image path/to/wide_photo.jpg --ensemble

Standalone run without --image uses a synthetic demo: it pastes a real
crack crop from the test set into a corner of a larger blank "wall",
simulating a wide shot, and shows that the sliding window correctly
localizes it.
"""

import os
import argparse

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from csaf_net import build_model, get_device

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Variants used by the ensemble by default: one full fusion model plus the
# two single-backbone models. These three are architecturally the most
# different from each other (fused vs. CNN-only vs. ViT-only), which gives
# the best chance that a false positive fooling one of them doesn't fool
# all three.
DEFAULT_ENSEMBLE_VARIANTS = ("full_csaf", "cnn_only", "vit_only")


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def is_wall_like(crop_pil, texture_std_threshold=12.0):
    """
    Lightweight heuristic pre-filter: does this crop's BACKGROUND (non-edge
    pixels) look like a real wall surface (concrete/plaster speckle texture),
    or a flat, near-uniform background like open sky or a sheet of paper?

    This is NOT a trained segmentation model -- it's a simple, explainable
    texture check that targets exactly the false-positive pattern the
    classifier is vulnerable to: a thin dark line (crack, wire, text stroke,
    fixture edge) sitting on top of an otherwise flat background. Real wall
    photos have consistent micro-texture (pores, speckling, staining) across
    the ENTIRE patch, not just along the line itself; sky and paper do not.

    Returns True if the crop looks wall-like (should be trusted), False if
    it looks like a flat/non-wall background (should be filtered out
    regardless of the classifier's confidence).
    """
    gray = cv2.cvtColor(np.array(crop_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges slightly so pixels right next to a line (blur/antialiasing)
    # aren't miscounted as "background".
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    non_edge_mask = edges_dilated == 0

    non_edge_pixels = gray[non_edge_mask]
    if non_edge_pixels.size < 200:
        # Almost the whole crop is edges -- can't judge background texture,
        # don't filter it out on this check alone.
        return True

    texture_std = float(non_edge_pixels.std())
    return texture_std >= texture_std_threshold


def iou(box_a, box_b):
    """box = (x1, y1, x2, y2)"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def non_max_suppression(boxes, scores, iou_threshold=0.3):
    """Greedy NMS. boxes: list of (x1,y1,x2,y2), scores: list of confidences."""
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = np.array([
            j for j in rest
            if iou(boxes[i], boxes[j]) < iou_threshold
        ])
    return keep


def sliding_window_detect(image, variant="full_csaf", checkpoint_dir="checkpoints",
                           window_size=224, stride=112, threshold=0.6,
                           nms_iou_threshold=0.3, device=None,
                           use_wall_filter=True, wall_texture_std_threshold=0.0):
    """
    image: PIL.Image (RGB), any size, ideally larger than window_size
    use_wall_filter: if True (default), a positive classifier prediction is
        only kept if the crop's background also passes the is_wall_like()
        texture heuristic -- filters out confident-but-wrong hits on sky,
        paper, and other flat non-wall backgrounds (see is_wall_like docstring).
    Returns: (annotated_image_np, detections) where detections is a list
    of dicts {box: (x1,y1,x2,y2), confidence: float, wall_like: bool}
    """
    if device is None:
        device = get_device()

    ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pth")
    model = build_model(variant=variant).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    transform = get_transform()
    W, H = image.size

    def _passes_filter(crop):
        if not use_wall_filter:
            return True
        return is_wall_like(crop, texture_std_threshold=wall_texture_std_threshold)

    # If the image is smaller than one window, just classify it whole
    # (falls back to normal single-crop behavior).
    if W <= window_size or H <= window_size:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(img_tensor)
            conf = torch.sigmoid(logits).item()
        boxes, scores = [], []
        if conf > threshold and _passes_filter(image):
            boxes = [(0, 0, W, H)]
            scores = [conf]
    else:
        boxes, scores = [], []
        for y in range(0, H - window_size + 1, stride):
            for x in range(0, W - window_size + 1, stride):
                crop = image.crop((x, y, x + window_size, y + window_size))
                img_tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(img_tensor)
                    conf = torch.sigmoid(logits).item()
                if conf > threshold and _passes_filter(crop):
                    boxes.append((x, y, x + window_size, y + window_size))
                    scores.append(conf)

        # cover the right/bottom edges if they don't divide evenly
        if (W - window_size) % stride != 0:
            for y in list(range(0, H - window_size + 1, stride)) + [H - window_size]:
                x = W - window_size
                crop = image.crop((x, y, x + window_size, y + window_size))
                img_tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(img_tensor)
                    conf = torch.sigmoid(logits).item()
                if conf > threshold and _passes_filter(crop):
                    boxes.append((x, y, x + window_size, y + window_size))
                    scores.append(conf)

    keep_idx = non_max_suppression(boxes, scores, iou_threshold=nms_iou_threshold)
    detections = [{"box": boxes[i], "confidence": scores[i]} for i in keep_idx]

    annotated = np.array(image.convert("RGB")).copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = f"{det['confidence']*100:.0f}%"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return annotated, detections


def ensemble_sliding_window_detect(image, variants=DEFAULT_ENSEMBLE_VARIANTS,
                                    checkpoint_dir="checkpoints", window_size=224, stride=112,
                                    threshold=0.6, min_agree=2, nms_iou_threshold=0.3, device=None,
                                    use_wall_filter=True, wall_texture_std_threshold=0.0):
    """
    Same sliding-window localization as sliding_window_detect(), but each
    window is classified by SEVERAL already-trained variants instead of
    one, and is only kept as a detection when at least `min_agree` of them
    independently predict "crack" above `threshold`.

    Rationale: different architectures make different mistakes. A window
    frame or text stroke that fools the fused full_csaf model (which
    leans on both branches) may not fool cnn_only or vit_only in the same
    way, since each sees the input through a different inductive bias.
    Requiring agreement is a form of ensemble filtering -- it trades
    recall (a real crack all three models miss is now dropped) for
    precision (false positives idiosyncratic to one model are dropped),
    without any retraining or new data -- purely reusing models already
    trained for the ablation study.

    variants: which trained variants to poll. Defaults to
        (full_csaf, cnn_only, vit_only) -- the three most architecturally
        different models available.
    min_agree: how many of len(variants) must predict positive for a
        window to be kept. Defaults to 2 (majority of 3).

    Returns: (annotated_image_np, detections) where each detection dict
    has "box", "confidence" (mean confidence across AGREEING variants),
    and "votes" (dict variant -> confidence for every polled variant, for
    transparency/debugging).
    """
    if device is None:
        device = get_device()

    models = {}
    for v in variants:
        ckpt_path = os.path.join(checkpoint_dir, f"{v}_best.pth")
        m = build_model(variant=v).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        models[v] = m

    transform = get_transform()
    W, H = image.size

    def _passes_filter(crop):
        if not use_wall_filter:
            return True
        return is_wall_like(crop, texture_std_threshold=wall_texture_std_threshold)

    def _predict_all(crop):
        img_tensor = transform(crop).unsqueeze(0).to(device)
        votes = {}
        with torch.no_grad():
            for v, m in models.items():
                logits, _ = m(img_tensor)
                votes[v] = torch.sigmoid(logits).item()
        return votes

    def _window_positions():
        if W <= window_size or H <= window_size:
            yield 0, 0
            return
        for y in range(0, H - window_size + 1, stride):
            for x in range(0, W - window_size + 1, stride):
                yield x, y
        if (W - window_size) % stride != 0:
            for y in list(range(0, H - window_size + 1, stride)) + [H - window_size]:
                yield W - window_size, y

    boxes, scores, all_votes = [], [], []
    seen_positions = set()
    for x, y in _window_positions():
        if (x, y) in seen_positions:
            continue
        seen_positions.add((x, y))

        if W <= window_size or H <= window_size:
            crop = image
            box = (0, 0, W, H)
        else:
            crop = image.crop((x, y, x + window_size, y + window_size))
            box = (x, y, x + window_size, y + window_size)

        votes = _predict_all(crop)
        agree_count = sum(1 for conf in votes.values() if conf > threshold)

        if agree_count >= min_agree and _passes_filter(crop):
            agreeing_confs = [conf for conf in votes.values() if conf > threshold]
            boxes.append(box)
            scores.append(float(np.mean(agreeing_confs)))
            all_votes.append(votes)

    keep_idx = non_max_suppression(boxes, scores, iou_threshold=nms_iou_threshold)
    detections = [
        {"box": boxes[i], "confidence": scores[i], "votes": all_votes[i]}
        for i in keep_idx
    ]

    annotated = np.array(image.convert("RGB")).copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
        label = f"{det['confidence']*100:.0f}%"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return annotated, detections


def make_synthetic_wide_demo(data_dir="dataset", canvas_size=900, save_path=None):
    """
    Builds a synthetic 'wide shot' by pasting a real Positive (crack) test
    crop into a random position on a large canvas filled with real
    Negative (no-crack) crops -- simulates a wide-angle wall photo where
    only a small region actually contains a crack.
    """
    from torchvision.datasets import ImageFolder
    import random

    test_ds = ImageFolder(os.path.join(data_dir, "test"))
    class_names = test_ds.classes
    pos_idx = class_names.index("Positive")
    neg_idx = class_names.index("Negative")

    pos_paths = [p for p, l in test_ds.samples if l == pos_idx]
    neg_paths = [p for p, l in test_ds.samples if l == neg_idx]

    canvas = Image.new("RGB", (canvas_size, canvas_size))
    tile = 224
    for ty in range(0, canvas_size, tile):
        for tx in range(0, canvas_size, tile):
            neg_img = Image.open(random.choice(neg_paths)).convert("RGB").resize((tile, tile))
            canvas.paste(neg_img, (tx, ty))

    pos_img = Image.open(random.choice(pos_paths)).convert("RGB").resize((tile, tile))
    px = random.randint(0, (canvas_size // tile) - 1) * tile
    py = random.randint(0, (canvas_size // tile) - 1) * tile
    canvas.paste(pos_img, (px, py))

    if save_path:
        canvas.save(save_path)
        print(f"Synthetic wide demo saved to {save_path} (crack placed at ({px}, {py}))")

    return canvas, (px, py, px + tile, py + tile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a wide-angle image. If omitted, a synthetic demo is generated.")
    parser.add_argument("--variant", type=str, default="full_csaf")
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--out", type=str, default="figures/sliding_window_demo.png")
    parser.add_argument("--ensemble", action="store_true",
                         help="Use ensemble_sliding_window_detect with DEFAULT_ENSEMBLE_VARIANTS "
                              "instead of a single variant.")
    parser.add_argument("--min_agree", type=int, default=2,
                         help="Ensemble mode only: how many variants must agree.")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("RGB")
        true_box = None
    else:
        print("No --image given -- generating a synthetic wide-shot demo instead.")
        image, true_box = make_synthetic_wide_demo(save_path="figures/synthetic_wide_input.png")

    if args.ensemble:
        print(f"Running ENSEMBLE sliding-window detection with variants={DEFAULT_ENSEMBLE_VARIANTS}, "
              f"min_agree={args.min_agree}...")
        annotated, detections = ensemble_sliding_window_detect(
            image, stride=args.stride, threshold=args.threshold, min_agree=args.min_agree
        )
    else:
        print("Running sliding-window detection...")
        annotated, detections = sliding_window_detect(
            image, variant=args.variant, stride=args.stride, threshold=args.threshold
        )

    print(f"Found {len(detections)} crack region(s):")
    for d in detections:
        line = f"  box={d['box']}  confidence={d['confidence']*100:.1f}%"
        if "votes" in d:
            vote_str = ", ".join(f"{v}={c*100:.0f}%" for v, c in d["votes"].items())
            line += f"  votes=({vote_str})"
        print(line)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(annotated)
    if true_box:
        x1, y1, x2, y2 = true_box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor="lime", facecolor="none",
                                  linestyle="--", label="Ground-truth crack location")
        ax.add_patch(rect)
        ax.legend(loc="upper right")
    title_suffix = f"ensemble, min_agree={args.min_agree}" if args.ensemble else args.variant
    ax.set_title(f"Sliding-Window Crack Detection ({title_suffix})")
    ax.axis("off")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")