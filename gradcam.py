"""
gradcam.py
Grad-CAM visualization for CSAF-Net's DenseNet-121 branch.

Shows WHERE in the image the model's CNN branch is looking when it makes
a prediction -- a spatial complement to the gate visualization (which
shows HOW MUCH the model trusts the CNN branch, but not WHERE within it).

Only applies to variants with a DenseNet-121 branch: cnn_only, concat,
ungated_cross_attn, scalar_gate, full_csaf. Does not apply to vit_only
(no CNN branch to visualize).

Usage:
    from gradcam import GradCAM, get_target_layer

    model, device = load_model(variant="full_csaf")
    target_layer = get_target_layer(model, "full_csaf")
    cam_engine = GradCAM(model, target_layer)
    cam, confidence = cam_engine.generate(image_tensor, device)
    overlay = overlay_heatmap(cam, original_pil_image)

Standalone:
    python gradcam.py
    -> saves figures/gradcam_examples.png (grid of originals + overlays)
"""

import os
import random

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from csaf_net import build_model, get_device

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Variants with a DenseNet-121 branch that Grad-CAM can be applied to.
CNN_BRANCH_VARIANTS = ["cnn_only", "concat", "ungated_cross_attn", "scalar_gate", "full_csaf"]


def get_target_layer(model, variant):
    """
    Returns the DenseNet-121 conv stack to hook Grad-CAM onto, for a given
    built model + variant name. Raises if the variant has no CNN branch.

    This is the FINAL feature map (7x7 spatial resolution at 224x224 input)
    -- good semantic signal, coarse localization. Use get_fine_target_layer
    for a higher-resolution (but slightly less semantic) alternative, e.g.
    for the combined Grad-CAM + Canny segmentation, where spatial precision
    matters more than semantic strength.
    """
    if variant == "cnn_only":
        return model.backbone.features
    elif variant in CNN_BRANCH_VARIANTS:
        return model.cnn_backbone.features
    else:
        raise ValueError(
            f"Grad-CAM needs a DenseNet-121 branch; variant '{variant}' has none "
            f"(e.g. vit_only). Choose from: {CNN_BRANCH_VARIANTS}"
        )


def get_fine_target_layer(model, variant):
    """
    Returns an EARLIER DenseNet-121 block (denseblock3, before the final
    downsampling transition) -- 14x14 spatial resolution at 224x224 input,
    double the final layer's 7x7. Each cell covers a ~16x16 pixel region
    instead of ~32x32, so a thin crack and a nearby window frame are less
    likely to fall inside the same cell and get conflated. Semantic
    strength is slightly weaker this early in the network, which is why
    this is used only where precise localization matters (segmentation),
    not for the primary Grad-CAM confidence display.
    """
    if variant == "cnn_only":
        return model.backbone.features.denseblock3
    elif variant in CNN_BRANCH_VARIANTS:
        return model.cnn_backbone.features.denseblock3
    else:
        raise ValueError(
            f"Grad-CAM needs a DenseNet-121 branch; variant '{variant}' has none "
            f"(e.g. vit_only). Choose from: {CNN_BRANCH_VARIANTS}"
        )


class GradCAM:
    """
    Standard Grad-CAM: hooks the forward activation and backward gradient
    of a target conv layer, then weights each channel's activation map by
    the global-average-pooled gradient for the target score.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, device):
        """
        image_tensor: (1, 3, 224, 224), NOT yet on device, NOT normalized
        for grad tracking -- this method handles moving to device and
        enabling gradients on the input itself (needed since the backbone
        is frozen: without requires_grad on the input, no gradient would
        reach the activation at all).

        Returns:
            cam: (h, w) numpy array in [0, 1], h/w matching the conv
                 feature map's spatial size (e.g. 7x7 for DenseNet-121
                 at 224x224 input)
            confidence: float, sigmoid(logit) for this image
        """
        self.model.eval()
        image_tensor = image_tensor.clone().detach().to(device).requires_grad_(True)

        logits, _ = self.model(image_tensor)
        score = logits.sum()   # single-image, single-logit binary task

        self.model.zero_grad()
        score.backward()

        gradients = self.gradients          # (1, C, h, w)
        activations = self.activations      # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        confidence = torch.sigmoid(logits).item()
        return cam, confidence


def overlay_heatmap(cam, original_pil_image, alpha=0.45, image_size=IMAGE_SIZE):
    """
    Resizes the low-res CAM up to image_size and blends it as a JET
    colormap over the (resized) original image.

    Returns an (H, W, 3) uint8 RGB numpy array, ready for plt.imshow
    or PIL.Image.fromarray.
    """
    heatmap = cv2.resize(cam, (image_size, image_size))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig_resized = np.array(original_pil_image.resize((image_size, image_size)).convert("RGB"))
    overlay = (alpha * heatmap_color + (1 - alpha) * orig_resized).astype(np.uint8)
    return overlay


def combined_crack_mask(cam, original_pil_image, image_size=IMAGE_SIZE,
                         cam_percentile=80, canny_low=50, canny_high=150,
                         dilate_iterations=1, mask_color=(0, 255, 0)):
    """
    Heuristic pixel-level crack mask combining TWO independently computed
    signals -- Grad-CAM (where the CNN branch attends) and Canny edges
    (where actual sharp intensity discontinuities are) -- rather than a
    trained segmentation model, which would require pixel-mask annotations
    this project's dataset (SDNET2018) does not provide.

    Rationale: Canny alone fires on every hard edge in a scene (window
    frames, brick coursing, architectural lines), which is far too noisy
    on a full facade. Grad-CAM alone is coarse (a 7x7 grid upsampled),
    giving only a rough region, not a precise outline. Intersecting them
    -- "only count edge pixels that also fall inside the model's attended
    region" -- keeps edges the model actually cares about and discards
    edges elsewhere in the frame, giving a materially tighter, more
    trustworthy outline than either signal alone.

    Returns:
        overlay: (image_size, image_size, 3) uint8 RGB array, the original
            image with surviving mask pixels highlighted in mask_color
        mask: (image_size, image_size) bool array, the final combined mask
    """
    # 1. Upsample the low-res CAM and threshold to a "high attention" region.
    cam_resized = cv2.resize(cam, (image_size, image_size))
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()
    threshold_val = np.percentile(cam_resized, cam_percentile)
    attended_mask = cam_resized >= threshold_val

    # 2. Compute Canny edges on the same crop at the same resolution.
    img_resized = np.array(original_pil_image.resize((image_size, image_size)).convert("RGB"))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if dilate_iterations > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilate_iterations)
    edge_mask = edges > 0

    # 3. Intersection: only keep edges that fall inside the attended region.
    combined_mask = edge_mask & attended_mask

    overlay = img_resized.copy()
    overlay[combined_mask] = mask_color

    return overlay, combined_mask


def mask_linearity_score(mask, min_contour_area=15):
    """
    Measures whether a binary mask (e.g. from combined_crack_mask) looks
    like a thin, elongated LINE (consistent with a real crack) or a
    blocky, roughly square/rectangular REGION (consistent with a window
    frame, stone block, or other architectural feature).

    For each connected component in the mask, fits a minimum-area
    rotated rectangle and computes its aspect ratio (long side / short
    side). A thin crack line produces a high aspect ratio (long and
    narrow); a window frame's outline or a blocky texture patch produces
    a low aspect ratio (closer to square).

    Returns the MAXIMUM aspect ratio found across components -- i.e. "is
    there at least one clearly line-like structure in this mask" -- since
    a real crack only needs to be present once, while non-crack clutter
    (e.g. a window frame outline) can coexist in the same mask.

    This is a simple geometric heuristic, not a learned classifier of
    "crack-shaped vs not" -- it estimates elongation from pixel geometry,
    nothing more.
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_aspect = 0.0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        short_side = max(min(w, h), 1e-6)
        long_side = max(w, h)
        aspect = long_side / short_side
        max_aspect = max(max_aspect, aspect)

    return max_aspect


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def plot_gradcam_examples(variant="full_csaf", data_dir="dataset", checkpoint_dir="checkpoints",
                           out_path="figures/gradcam_examples.png", num_samples=6):
    """
    Picks a mix of correctly-classified Positive (crack) and Negative
    (no crack) test images, runs Grad-CAM on the CNN branch for each,
    and saves a grid of original | heatmap overlay pairs.
    """
    device = get_device()

    ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  Skipping Grad-CAM -- checkpoint not found at {ckpt_path}")
        return

    model = build_model(variant=variant).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    target_layer = get_target_layer(model, variant)
    cam_engine = GradCAM(model, target_layer)

    full_ds = ImageFolder(os.path.join(data_dir, "test"), transform=get_test_transform())
    class_names = full_ds.classes
    pos_idx = class_names.index("Positive")
    neg_idx = class_names.index("Negative")

    indices = list(range(len(full_ds)))
    random.shuffle(indices)

    half = num_samples // 2
    picked = []   # list of (tensor, raw_pil_image, label)
    n_pos, n_neg = 0, 0

    for idx in indices:
        img_tensor, lab = full_ds[idx]        # img_tensor: (3, 224, 224), normalized
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 224, 224)
        sample_path, _ = full_ds.samples[idx]

        with torch.no_grad():
            logits, _ = model(img_tensor.to(device))
            pred = int(torch.sigmoid(logits).item() > 0.5)

        if lab != pred:
            continue

        raw_img = Image.open(sample_path).convert("RGB")

        if lab == pos_idx and n_pos < half:
            picked.append((img_tensor, raw_img, lab))
            n_pos += 1
        elif lab == neg_idx and n_neg < (num_samples - half):
            picked.append((img_tensor, raw_img, lab))
            n_neg += 1

        if n_pos >= half and n_neg >= (num_samples - half):
            break

    if not picked:
        print("  Skipping Grad-CAM -- no correctly-classified samples found")
        return

    fig, axes = plt.subplots(2, len(picked), figsize=(3 * len(picked), 6.5))
    if len(picked) == 1:
        axes = axes.reshape(2, 1)

    for i, (img_tensor, raw_img, lab) in enumerate(picked):
        cam, confidence = cam_engine.generate(img_tensor, device)
        overlay = overlay_heatmap(cam, raw_img)

        axes[0, i].imshow(raw_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        true_label = class_names[lab]
        axes[0, i].set_title(f"True: {true_label}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        pred_label = "Positive" if confidence > 0.5 else "Negative"
        axes[1, i].set_title(f"Grad-CAM (Pred: {pred_label}, {confidence*100:.1f}%)", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(f"Grad-CAM -- CNN Branch Attention -- {variant}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Generating Grad-CAM examples for full_csaf...")
    plot_gradcam_examples(variant="full_csaf")