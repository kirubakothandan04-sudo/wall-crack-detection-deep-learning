import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms

from predict_csaf import (
    load_model,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from sliding_window_detect import sliding_window_detect


# ============================================================
# Configuration
# ============================================================

SEED = 42

N_POSITIVE = 20
N_NEGATIVE = 20

CANVAS_SIZE = 900
TILE_SIZE = 224

THRESHOLD = 0.99
IOU_THRESHOLD = 0.30
STRIDE = 112

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Dataset
# ============================================================

dataset = ImageFolder("dataset/test")

class_names = dataset.classes

pos_idx = class_names.index("Positive")
neg_idx = class_names.index("Negative")

pos_paths = [
    p for p, y in dataset.samples
    if y == pos_idx
]

neg_paths = [
    p for p, y in dataset.samples
    if y == neg_idx
]

print("Classes:", class_names)
print("Positive images:", len(pos_paths))
print("Negative images:", len(neg_paths))


# ============================================================
# Synthetic wide-view canvas generation
# ============================================================

def make_positive_canvas():
    """
    Creates a 900x900 wide scene containing:
      - mostly negative/no-crack 224x224 tiles
      - exactly one positive/crack 224x224 tile

    Returns:
      canvas
      ground-truth crack bounding box
    """

    canvas = Image.new(
        "RGB",
        (CANVAS_SIZE, CANVAS_SIZE)
    )

    # Fill canvas with negative tiles
    for ty in range(0, CANVAS_SIZE, TILE_SIZE):

        for tx in range(0, CANVAS_SIZE, TILE_SIZE):

            img = Image.open(
                random.choice(neg_paths)
            ).convert("RGB")

            img = img.resize(
                (TILE_SIZE, TILE_SIZE)
            )

            canvas.paste(
                img,
                (tx, ty)
            )

    # Select one real crack image
    crack_path = random.choice(pos_paths)

    crack_img = Image.open(
        crack_path
    ).convert("RGB")

    crack_img = crack_img.resize(
        (TILE_SIZE, TILE_SIZE)
    )

    # Random tile location
    grid_x = random.randint(
        0,
        CANVAS_SIZE // TILE_SIZE - 1
    )

    grid_y = random.randint(
        0,
        CANVAS_SIZE // TILE_SIZE - 1
    )

    x = grid_x * TILE_SIZE
    y = grid_y * TILE_SIZE

    # Paste crack
    canvas.paste(
        crack_img,
        (x, y)
    )

    true_box = (
        x,
        y,
        x + TILE_SIZE,
        y + TILE_SIZE
    )

    return canvas, true_box


def make_negative_canvas():
    """
    Creates a 900x900 scene containing only negative/no-crack tiles.
    """

    canvas = Image.new(
        "RGB",
        (CANVAS_SIZE, CANVAS_SIZE)
    )

    for ty in range(0, CANVAS_SIZE, TILE_SIZE):

        for tx in range(0, CANVAS_SIZE, TILE_SIZE):

            img = Image.open(
                random.choice(neg_paths)
            ).convert("RGB")

            img = img.resize(
                (TILE_SIZE, TILE_SIZE)
            )

            canvas.paste(
                img,
                (tx, ty)
            )

    return canvas


# ============================================================
# Intersection over Union
# ============================================================

def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1)
        *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        *
        max(0, by2 - by1)
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Whole-image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),
])


# ============================================================
# Whole-image prediction
# ============================================================

def whole_image_predict(
    model,
    device,
    image
):

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits, _ = model(tensor)

        probability = (
            torch.sigmoid(logits)
            .item()
        )

    return probability


# ============================================================
# Load CSAF-Net
# ============================================================

print("\nLoading CSAF-Net...")

model, device = load_model(
    variant="full_csaf"
)

print("Device:", device)


# ============================================================
# Generate SAME evaluation canvases
# ============================================================

print(
    "\nGenerating fixed evaluation set..."
)

positive_cases = []
negative_cases = []


for _ in range(N_POSITIVE):

    canvas, true_box = (
        make_positive_canvas()
    )

    positive_cases.append(
        (canvas, true_box)
    )


for _ in range(N_NEGATIVE):

    canvas = make_negative_canvas()

    negative_cases.append(
        canvas
    )


print(
    f"Generated "
    f"{len(positive_cases)} positive + "
    f"{len(negative_cases)} negative canvases"
)


# ============================================================
# Storage for results
# ============================================================

# Whole-image classification
whole_pos_predictions = []
whole_neg_predictions = []

# Wide-view classification
wide_pos_any_detection = []
wide_neg_any_detection = []

# Wide-view localization
wide_pos_found = []

# IoU values
wide_pos_ious = []


# ============================================================
# Evaluation header
# ============================================================

print(
    "\n======================================"
)

print(
    "WHOLE IMAGE vs WIDE VIEW"
)

print(
    "======================================"
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Stride:",
    STRIDE
)

print(
    "IoU threshold:",
    IOU_THRESHOLD
)


# ============================================================
# Positive canvases
# ============================================================

print(
    "\nPositive canvases..."
)


for i, (canvas, true_box) in enumerate(
    positive_cases,
    start=1
):

    # --------------------------------------------------------
    # Whole-image baseline
    # --------------------------------------------------------

    whole_prob = whole_image_predict(
        model,
        device,
        canvas
    )

    whole_pos_predictions.append(
        whole_prob >= THRESHOLD
    )

    # --------------------------------------------------------
    # Wide-view detection
    # IMPORTANT:
    # sliding_window_detect returns:
    #
    #     annotated_image, detections
    # --------------------------------------------------------

    _, detections = (
        sliding_window_detect(
            canvas,
            variant="full_csaf",
            stride=STRIDE,
            threshold=THRESHOLD,
            nms_iou_threshold=IOU_THRESHOLD,
        )
    )

    # Any detection at all?
    wide_pos_any_detection.append(
        len(detections) > 0
    )

    # --------------------------------------------------------
    # Find best IoU with ground truth
    # --------------------------------------------------------

    best_iou = 0.0

    for det in detections:

        # Detection format:
        #
        # {
        #     "box": (x1, y1, x2, y2),
        #     "confidence": score
        # }

        box = det["box"]

        current_iou = box_iou(
            box,
            true_box
        )

        best_iou = max(
            best_iou,
            current_iou
        )

    wide_pos_ious.append(
        best_iou
    )

    # Successful localization?
    found = (
        best_iou >= IOU_THRESHOLD
    )

    wide_pos_found.append(
        found
    )

    print(
        f"[POS {i:02d}] "
        f"Whole={whole_prob:.3f} | "
        f"Wide detections={len(detections)} | "
        f"Best IoU={best_iou:.3f} | "
        f"Found={found}"
    )


# ============================================================
# Negative canvases
# ============================================================

print(
    "\nNegative canvases..."
)


for i, canvas in enumerate(
    negative_cases,
    start=1
):

    # --------------------------------------------------------
    # Whole-image baseline
    # --------------------------------------------------------

    whole_prob = whole_image_predict(
        model,
        device,
        canvas
    )

    whole_neg_predictions.append(
        whole_prob >= THRESHOLD
    )

    # --------------------------------------------------------
    # Wide-view
    # --------------------------------------------------------

    _, detections = (
        sliding_window_detect(
            canvas,
            variant="full_csaf",
            stride=STRIDE,
            threshold=THRESHOLD,
            nms_iou_threshold=IOU_THRESHOLD,
        )
    )

    wide_neg_any_detection.append(
        len(detections) > 0
    )

    print(
        f"[NEG {i:02d}] "
        f"Whole={whole_prob:.3f} | "
        f"Wide detections={len(detections)}"
    )


# ============================================================
# Binary metrics
# ============================================================

def binary_metrics(
    positive_predictions,
    negative_predictions
):

    tp = sum(
        positive_predictions
    )

    fn = (
        len(positive_predictions)
        -
        tp
    )

    fp = sum(
        negative_predictions
    )

    tn = (
        len(negative_predictions)
        -
        fp
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        /
        (precision + recall)
        if (precision + recall)
        else 0.0
    )

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn)
    )

    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Calculate metrics
# ============================================================

whole_metrics = binary_metrics(
    whole_pos_predictions,
    whole_neg_predictions
)


wide_classification_metrics = (
    binary_metrics(
        wide_pos_any_detection,
        wide_neg_any_detection
    )
)


wide_localization_metrics = (
    binary_metrics(
        wide_pos_found,
        wide_neg_any_detection
    )
)


# ============================================================
# Final results
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "FINAL COMPARISON"
)

print(
    "=" * 70
)


# ------------------------------------------------------------
# Whole image
# ------------------------------------------------------------

print(
    "\nWHOLE-IMAGE BASELINE"
)

for k, v in whole_metrics.items():

    if isinstance(v, float):

        print(
            f"{k:12s}: {v:.4f}"
        )

    else:

        print(
            f"{k:12s}: {v}"
        )


# ------------------------------------------------------------
# Wide view: any detection
# ------------------------------------------------------------

print(
    "\nWIDE VIEW — ANY DETECTION"
)

for k, v in (
    wide_classification_metrics.items()
):

    if isinstance(v, float):

        print(
            f"{k:12s}: {v:.4f}"
        )

    else:

        print(
            f"{k:12s}: {v}"
        )


# ------------------------------------------------------------
# Wide view: localization
# ------------------------------------------------------------

print(
    "\nWIDE VIEW — LOCALIZED DETECTION (IoU)"
)

for k, v in (
    wide_localization_metrics.items()
):

    if isinstance(v, float):

        print(
            f"{k:12s}: {v:.4f}"
        )

    else:

        print(
            f"{k:12s}: {v}"
        )


# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

average_iou = np.mean(
    wide_pos_ious
)

print(
    "\nAverage best IoU on positive canvases:",
    f"{average_iou:.4f}"
)


# ============================================================
# Save results
# ============================================================

result_path = os.path.join(
    OUT_DIR,
    "wide_view_comparison.txt"
)


with open(
    result_path,
    "w"
) as f:

    f.write(
        "Whole Image vs Wide View\n"
    )

    f.write(
        f"Seed: {SEED}\n"
        f"Positive canvases: {N_POSITIVE}\n"
        f"Negative canvases: {N_NEGATIVE}\n"
        f"Canvas size: {CANVAS_SIZE}\n"
        f"Tile size: {TILE_SIZE}\n"
        f"Threshold: {THRESHOLD}\n"
        f"Stride: {STRIDE}\n"
        f"IoU threshold: {IOU_THRESHOLD}\n\n"
    )

    f.write(
        "WHOLE IMAGE\n"
    )

    f.write(
        str(whole_metrics)
    )

    f.write(
        "\n\n"
    )

    f.write(
        "WIDE VIEW ANY DETECTION\n"
    )

    f.write(
        str(wide_classification_metrics)
    )

    f.write(
        "\n\n"
    )

    f.write(
        "WIDE VIEW LOCALIZATION\n"
    )

    f.write(
        str(wide_localization_metrics)
    )

    f.write(
        "\n\n"
    )

    f.write(
        f"Average best IoU: "
        f"{average_iou:.4f}\n"
    )


print(
    "\nSaved:",
    result_path
)
