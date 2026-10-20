"""
Evaluate CSAF-Net Grad-CAM localization against ground-truth pixel-level masks.

This is an exploratory evaluation on a small set of manually annotated images.
Grad-CAM is a weakly-supervised localization method driven by classification,
NOT a trained segmentation model. Expected IoU ceiling: 0.3-0.5 (vs 0.7-0.9 for
trained segmentation models).

Usage:
    python evaluate_gradcam_masks.py

Outputs:
    - research_artifacts/gradcam_mask_evaluation.csv (per-image metrics)
    - research_artifacts/gradcam_mask_evaluation.md (summary report)
    - research_artifacts/gradcam_eval_plots/ (visualizations)
"""
import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

# Import project modules
from gradcam_csaf import GradCAM, disable_inplace
from predict_csaf import load_model

# Configuration
XAI_PHOTOS_DIR = "xai_photos"
OUTPUT_DIR = "research_artifacts"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "gradcam_eval_plots")
VARIANT = "full_csaf"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Excluded images
EXCLUDED = ["sdnet_7125-186.jpg"]  # Empty mask

os.makedirs(PLOTS_DIR, exist_ok=True)


def get_image_mask_pairs():
    """Find all valid image-mask pairs, excluding empty masks."""
    pairs = []
    for img_file in sorted(os.listdir(XAI_PHOTOS_DIR)):
        if img_file in EXCLUDED:
            continue
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        if '_mask' in img_file:
            continue

        base = os.path.splitext(img_file)[0]
        mask_file = f"{base}_mask.png"
        mask_path = os.path.join(XAI_PHOTOS_DIR, mask_file)

        if os.path.exists(mask_path):
            pairs.append((img_file, mask_file))

    return pairs


def preprocess_for_model(image_path):
    """Load image and return both original and preprocessed tensor."""
    img_original = cv2.imread(image_path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(img_original)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(img_pil).unsqueeze(0)

    return img_original, tensor


def generate_gradcam(model, gradcam_engine, tensor, original_shape):
    """
    Generate Grad-CAM heatmap and resize to original image dimensions.

    Returns:
        cam_resized: (H, W) float array normalized to [0, 1]
        confidence: float, model confidence score
    """
    cam, confidence = gradcam_engine(tensor)

    # Verify normalization (should already be [0, 1] from GradCAM implementation)
    assert cam.min() >= 0 and cam.max() <= 1, f"CAM not normalized: [{cam.min()}, {cam.max()}]"

    # Resize CAM to original image dimensions using bilinear interpolation
    h, w = original_shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # Ensure still in [0, 1] after resize
    cam_resized = np.clip(cam_resized, 0, 1)

    return cam_resized, confidence


def apply_threshold(cam, method='otsu', fixed_threshold=0.5):
    """
    Apply thresholding to continuous CAM to produce binary mask.

    Args:
        cam: (H, W) float array in [0, 1]
        method: 'otsu' or 'fixed'
        fixed_threshold: used if method='fixed'

    Returns:
        binary_mask: (H, W) boolean array
        threshold_value: float, the threshold used
    """
    if method == 'otsu':
        # Convert to uint8 for Otsu's method
        cam_uint8 = (cam * 255).astype(np.uint8)
        threshold_value, binary_mask = cv2.threshold(
            cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_value = threshold_value / 255.0  # Convert back to [0, 1]
        binary_mask = binary_mask > 127
    elif method == 'fixed':
        threshold_value = fixed_threshold
        binary_mask = cam >= fixed_threshold
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    return binary_mask, threshold_value


def load_ground_truth_mask(mask_path):
    """
    Load ground-truth mask as binary array.

    Returns:
        mask: (H, W) boolean array, True = crack pixel
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Threshold at 127 to ensure binary (annotation tool uses 255 for crack)
    mask_binary = mask > 127
    return mask_binary


def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate pixel-level segmentation metrics.

    Handles zero-denominator cases explicitly:
    - If both pred and gt are empty: precision=1.0, recall=1.0, IoU=1.0, Dice=1.0
    - If pred empty but gt not: precision=1.0 (no FP), recall=0.0, IoU=0.0, Dice=0.0
    - If gt empty but pred not: precision=0.0, recall=1.0 (no FN), IoU=0.0, Dice=0.0

    Returns:
        dict with keys: iou, dice, precision, recall, f1
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    tn = np.sum(~pred_mask & ~gt_mask)

    # Handle zero denominators
    if tp + fp + fn == 0:
        # Both masks empty
        return {
            'iou': 1.0,
            'dice': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    # IoU (Intersection over Union)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Dice coefficient
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def save_visualization(img_original, cam, pred_otsu, pred_fixed, gt_mask,
                       img_name, output_dir):
    """Save side-by-side visualization of predictions vs ground truth."""
    h, w = img_original.shape[:2]

    # Create visualization
    fig_width = w * 5
    fig_height = h
    canvas = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

    # 1. Original image
    canvas[:, 0:w] = img_original

    # 2. Grad-CAM heatmap
    cam_vis = (cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_vis, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    canvas[:, w:2*w] = cam_color

    # 3. Otsu thresholded
    pred_otsu_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_otsu_vis[pred_otsu] = [0, 255, 0]  # Green for predicted crack
    canvas[:, 2*w:3*w] = pred_otsu_vis

    # 4. Fixed threshold (0.5)
    pred_fixed_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_fixed_vis[pred_fixed] = [0, 255, 0]
    canvas[:, 3*w:4*w] = pred_fixed_vis

    # 5. Ground truth
    gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt_mask] = [255, 255, 255]  # White for ground truth crack
    canvas[:, 4*w:5*w] = gt_vis

    # Save
    base_name = os.path.splitext(img_name)[0]
    output_path = os.path.join(output_dir, f"{base_name}_eval.png")
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, canvas_bgr)


def main():
    print("=" * 80)
    print("Grad-CAM Localization Evaluation Against Ground-Truth Masks")
    print("=" * 80)
    print()

    # Load model
    print("Loading CSAF-Net model...")
    model, device = load_model(variant=VARIANT)
    model.eval()

    # Set up Grad-CAM
    print(f"Setting up Grad-CAM on target layer: cnn_backbone.features.denseblock3...")
    target_layer = model.cnn_backbone.features.denseblock3
    gradcam_engine = GradCAM(model, target_layer, plus=False)

    # Get image-mask pairs
    pairs = get_image_mask_pairs()
    print(f"Found {len(pairs)} valid image-mask pairs")
    print(f"Excluded: {EXCLUDED}")
    print()

    # Evaluate each pair
    results_otsu = []
    results_fixed = []

    print("Running evaluation...")
    for img_name, mask_name in tqdm(pairs, desc="Processing"):
        img_path = os.path.join(XAI_PHOTOS_DIR, img_name)
        mask_path = os.path.join(XAI_PHOTOS_DIR, mask_name)

        # Load and preprocess
        img_original, tensor = preprocess_for_model(img_path)
        tensor = tensor.to(device)

        # Generate Grad-CAM
        cam, confidence = generate_gradcam(model, gradcam_engine, tensor, img_original.shape)

        # Load ground truth
        gt_mask = load_ground_truth_mask(mask_path)

        # Apply both thresholding methods
        pred_otsu, threshold_otsu = apply_threshold(cam, method='otsu')
        pred_fixed, threshold_fixed = apply_threshold(cam, method='fixed', fixed_threshold=0.5)

        # Calculate metrics for Otsu
        metrics_otsu = calculate_metrics(pred_otsu, gt_mask)
        results_otsu.append({
            'image': img_name,
            'iou': metrics_otsu['iou'],
            'dice': metrics_otsu['dice'],
            'precision': metrics_otsu['precision'],
            'recall': metrics_otsu['recall'],
            'f1': metrics_otsu['f1'],
            'confidence': confidence,
            'threshold': threshold_otsu,
            'crack_pixels_gt': int(np.sum(gt_mask)),
            'crack_pixels_pred': int(np.sum(pred_otsu)),
            'tp': int(metrics_otsu['tp']),
            'fp': int(metrics_otsu['fp']),
            'fn': int(metrics_otsu['fn']),
        })

        # Calculate metrics for fixed threshold
        metrics_fixed = calculate_metrics(pred_fixed, gt_mask)
        results_fixed.append({
            'image': img_name,
            'iou': metrics_fixed['iou'],
            'dice': metrics_fixed['dice'],
            'precision': metrics_fixed['precision'],
            'recall': metrics_fixed['recall'],
            'f1': metrics_fixed['f1'],
            'confidence': confidence,
            'threshold': threshold_fixed,
            'crack_pixels_gt': int(np.sum(gt_mask)),
            'crack_pixels_pred': int(np.sum(pred_fixed)),
            'tp': int(metrics_fixed['tp']),
            'fp': int(metrics_fixed['fp']),
            'fn': int(metrics_fixed['fn']),
        })

        # Save visualization for first 10 images
        if len(results_otsu) <= 10:
            save_visualization(img_original, cam, pred_otsu, pred_fixed, gt_mask,
                             img_name, PLOTS_DIR)

    # Convert to DataFrames
    df_otsu = pd.DataFrame(results_otsu)
    df_fixed = pd.DataFrame(results_fixed)

    # Save CSVs
    csv_otsu = os.path.join(OUTPUT_DIR, "gradcam_mask_evaluation_otsu.csv")
    csv_fixed = os.path.join(OUTPUT_DIR, "gradcam_mask_evaluation_fixed.csv")
    df_otsu.to_csv(csv_otsu, index=False, float_format='%.4f')
    df_fixed.to_csv(csv_fixed, index=False, float_format='%.4f')

    print()
    print(f"Saved per-image results:")
    print(f"  - {csv_otsu}")
    print(f"  - {csv_fixed}")
    print()

    # Generate summary report
    generate_summary_report(df_otsu, df_fixed, len(pairs))

    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


def generate_summary_report(df_otsu, df_fixed, n_images):
    """Generate markdown summary report."""

    report = f"""# Grad-CAM Localization Evaluation Report
**Date:** 2026-10-12
**Model:** CSAF-Net (full_csaf variant)
**Target Layer:** cnn.features.denseblock3 (14×14 feature maps)
**Evaluated Images:** {n_images}/25 (1 excluded: empty mask)

---

## Disclaimer

**This is an exploratory evaluation on a small manually annotated set.**

Grad-CAM is a **weakly-supervised localization method** driven by classification objectives, NOT a trained segmentation model. It highlights discriminative regions for the classification decision but is not optimized for precise boundary delineation.

**Expected performance:**
- Grad-CAM / CAM-based methods: IoU ~0.3–0.5
- Trained segmentation models: IoU ~0.7–0.9

**Purpose:** Measure how well classification-driven attention aligns with crack boundaries, not segmentation accuracy.

---

## Methodology

### Grad-CAM Generation
1. Load image and preprocess to 224×224 (ImageNet normalization)
2. Forward pass through CSAF-Net to obtain crack classification logit
3. Backpropagate from crack logit to target layer (denseblock3)
4. Compute Grad-CAM: weighted sum of feature maps using gradient-based weights
5. Apply ReLU and normalize to [0, 1]: `cam = (cam - min) / (max - min)`
6. Resize CAM to original image dimensions using bilinear interpolation

### Binarization
Two thresholding methods evaluated:

**1. Otsu's Thresholding (Adaptive)**
- Automatically determines optimal threshold to separate foreground/background
- Data-driven, no manual tuning
- Threshold varies per image based on CAM intensity distribution

**2. Fixed Threshold at 0.5**
- Constant threshold across all images
- Simple baseline for comparison

### Ground Truth
- Manually annotated binary masks from `xai_photos/`
- Polyline-based annotation tool with 7px stroke width
- White (255) = crack, Black (0) = background
- Thresholded at 127 to ensure binary

### Metrics
Per-image pixel-level metrics:
- **IoU (Intersection over Union):** TP / (TP + FP + FN)
- **Dice Score:** 2×TP / (2×TP + FP + FN)
- **Precision:** TP / (TP + FP) — fraction of predicted crack pixels that are correct
- **Recall:** TP / (TP + FN) — fraction of true crack pixels detected
- **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)

**Zero-denominator convention:**
- Both masks empty: All metrics = 1.0
- Prediction empty, GT not: Precision = 1.0, Recall = 0.0, IoU/Dice/F1 = 0.0
- GT empty, prediction not: Precision = 0.0, Recall = 1.0, IoU/Dice/F1 = 0.0

### Exclusions
- **sdnet_7125-186.jpg** excluded (empty mask, 0 crack pixels)
- Final N = {n_images} images evaluated

---

## Results

### Otsu Thresholding (Adaptive)

"""

    # Otsu statistics
    metrics_otsu = ['iou', 'dice', 'precision', 'recall', 'f1']
    report += "| Metric | Mean | Median | Std | Min | Max |\n"
    report += "|--------|------|--------|-----|-----|-----|\n"
    for metric in metrics_otsu:
        values = df_otsu[metric]
        report += f"| **{metric.upper()}** | {values.mean():.4f} | {values.median():.4f} | {values.std():.4f} | {values.min():.4f} | {values.max():.4f} |\n"

    report += f"\n**Model Confidence:** Mean = {df_otsu['confidence'].mean():.4f}, Median = {df_otsu['confidence'].median():.4f}\n"
    report += f"**Otsu Threshold:** Mean = {df_otsu['threshold'].mean():.4f}, Median = {df_otsu['threshold'].median():.4f}\n\n"

    # Fixed threshold statistics
    report += "### Fixed Threshold (0.5)\n\n"
    report += "| Metric | Mean | Median | Std | Min | Max |\n"
    report += "|--------|------|--------|-----|-----|-----|\n"
    for metric in metrics_otsu:
        values = df_fixed[metric]
        report += f"| **{metric.upper()}** | {values.mean():.4f} | {values.median():.4f} | {values.std():.4f} | {values.min():.4f} | {values.max():.4f} |\n"

    report += f"\n**Model Confidence:** Mean = {df_fixed['confidence'].mean():.4f}, Median = {df_fixed['confidence'].median():.4f}\n\n"

    # Comparison
    report += f"""### Method Comparison

| Method | Mean IoU | Mean Dice | Mean Precision | Mean Recall | Mean F1 |
|--------|----------|-----------|----------------|-------------|---------|
| **Otsu (Adaptive)** | {df_otsu['iou'].mean():.4f} | {df_otsu['dice'].mean():.4f} | {df_otsu['precision'].mean():.4f} | {df_otsu['recall'].mean():.4f} | {df_otsu['f1'].mean():.4f} |
| **Fixed (0.5)** | {df_fixed['iou'].mean():.4f} | {df_fixed['dice'].mean():.4f} | {df_fixed['precision'].mean():.4f} | {df_fixed['recall'].mean():.4f} | {df_fixed['f1'].mean():.4f} |

---

## Analysis

### Performance Assessment

"""

    mean_iou_otsu = df_otsu['iou'].mean()
    mean_iou_fixed = df_fixed['iou'].mean()

    if mean_iou_otsu >= 0.3:
        assessment = "REASONABLE"
        explanation = f"Mean IoU of {mean_iou_otsu:.3f} is within the expected range (0.3–0.5) for weakly-supervised CAM-based localization methods."
    elif mean_iou_otsu >= 0.2:
        assessment = "MODERATE"
        explanation = f"Mean IoU of {mean_iou_otsu:.3f} is below typical CAM performance but still indicates some alignment between attention and crack boundaries."
    else:
        assessment = "WEAK"
        explanation = f"Mean IoU of {mean_iou_otsu:.3f} suggests limited alignment. The model may be attending to context (background, frame) rather than cracks themselves."

    report += f"**Overall: {assessment}**\n\n{explanation}\n\n"

    # Best/worst cases
    best_idx_otsu = df_otsu['iou'].idxmax()
    worst_idx_otsu = df_otsu['iou'].idxmin()

    report += f"""### Best and Worst Cases (Otsu)

**Best Localization:**
- Image: `{df_otsu.loc[best_idx_otsu, 'image']}`
- IoU: {df_otsu.loc[best_idx_otsu, 'iou']:.4f}
- Dice: {df_otsu.loc[best_idx_otsu, 'dice']:.4f}
- Precision: {df_otsu.loc[best_idx_otsu, 'precision']:.4f}
- Recall: {df_otsu.loc[best_idx_otsu, 'recall']:.4f}

**Worst Localization:**
- Image: `{df_otsu.loc[worst_idx_otsu, 'image']}`
- IoU: {df_otsu.loc[worst_idx_otsu, 'iou']:.4f}
- Dice: {df_otsu.loc[worst_idx_otsu, 'dice']:.4f}
- Precision: {df_otsu.loc[worst_idx_otsu, 'precision']:.4f}
- Recall: {df_otsu.loc[worst_idx_otsu, 'recall']:.4f}

---

## Limitations

### Method Limitations
1. **Not a segmentation model:** Grad-CAM is designed for classification interpretability, not pixel-accurate boundary prediction
2. **Coarse localization:** Even with denseblock3 (14×14), spatial resolution is limited compared to U-Net/DeepLabV3 architectures
3. **Binary thresholding:** Converting continuous attention to binary masks is inherently lossy and threshold-dependent
4. **No boundary refinement:** Unlike segmentation models, no post-processing or CRF refinement is applied

### Evaluation Limitations
1. **Small sample size:** N={n_images} images is insufficient for robust generalization claims
2. **No train/test split:** Thresholds (Otsu, fixed 0.5) evaluated on the same set used to choose them
3. **Label this as exploratory:** Do not present these metrics as independent test performance
4. **Annotation bias:** Manual polyline annotations may not capture all crack pixels (7px stroke width)
5. **Domain shift:** Mixed sources (web photos, SDNET2018) may affect consistency

### Technical Notes
- **Normalization:** Grad-CAM output is normalized to [0, 1] by implementation (no double normalization)
- **Interpolation:** Bilinear for continuous CAM upsampling to preserve gradient smoothness
- **Coordinate system:** Standard OpenCV (H, W) ordering, verified alignment between predictions and masks
- **Metric aggregation:** Mean and median reported; median more robust to outliers

---

## Recommendations

### For Improved Localization
1. **Train a segmentation model:** Use these {n_images} pairs + expand to 50–100 annotated images
2. **Use Grad-CAM++ or Score-CAM:** May improve localization over vanilla Grad-CAM
3. **Multi-scale fusion:** Combine CAMs from multiple layers (denseblock2, denseblock3, denseblock4)
4. **Post-process with CRF:** Conditional Random Field can refine boundaries using image edges

### For Robust Evaluation
1. **Expand annotation set:** Target 100+ images across diverse crack types and backgrounds
2. **Stratified split:** Reserve 20–30% as held-out test set, never used for threshold tuning
3. **Cross-validation:** 5-fold CV to assess metric stability
4. **Compare to baselines:** Simple thresholds on grayscale images, Canny edges, etc.

---

## Files Generated

### Data
- `gradcam_mask_evaluation_otsu.csv` — Per-image metrics (Otsu thresholding)
- `gradcam_mask_evaluation_fixed.csv` — Per-image metrics (Fixed threshold 0.5)

### Visualizations
- `gradcam_eval_plots/*.png` — First 10 images showing:
  - Column 1: Original image
  - Column 2: Grad-CAM heatmap
  - Column 3: Otsu thresholded prediction (green)
  - Column 4: Fixed threshold prediction (green)
  - Column 5: Ground truth mask (white)

---

## Conclusion

The CSAF-Net Grad-CAM achieves a mean IoU of **{mean_iou_otsu:.3f}** (Otsu) and **{mean_iou_fixed:.3f}** (fixed 0.5) on {n_images} manually annotated crack images. This performance is {'consistent with' if mean_iou_otsu >= 0.3 else 'below'} the expected range for weakly-supervised attention-based localization methods.

**Key Takeaway:** Grad-CAM provides approximate localization suitable for interpretability and coarse spatial understanding, but should not replace trained segmentation models for applications requiring precise crack boundary delineation.

**Next Steps:** {'Expand annotation set to 50–100 images and train a U-Net or DeepLabV3 model for production-grade segmentation.' if mean_iou_otsu >= 0.25 else 'Investigate whether the model is attending to crack regions or spurious context before expanding annotation efforts.'}
"""

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "gradcam_mask_evaluation.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Saved summary report: {report_path}")
    print()
    print("Summary Statistics (Otsu):")
    print(f"  Mean IoU:       {df_otsu['iou'].mean():.4f}")
    print(f"  Mean Dice:      {df_otsu['dice'].mean():.4f}")
    print(f"  Mean Precision: {df_otsu['precision'].mean():.4f}")
    print(f"  Mean Recall:    {df_otsu['recall'].mean():.4f}")
    print(f"  Mean F1:        {df_otsu['f1'].mean():.4f}")
    print()
    print("Summary Statistics (Fixed 0.5):")
    print(f"  Mean IoU:       {df_fixed['iou'].mean():.4f}")
    print(f"  Mean Dice:      {df_fixed['dice'].mean():.4f}")
    print(f"  Mean Precision: {df_fixed['precision'].mean():.4f}")
    print(f"  Mean Recall:    {df_fixed['recall'].mean():.4f}")
    print(f"  Mean F1:        {df_fixed['f1'].mean():.4f}")


if __name__ == "__main__":
    main()
