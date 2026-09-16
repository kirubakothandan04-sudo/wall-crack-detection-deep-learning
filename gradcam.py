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