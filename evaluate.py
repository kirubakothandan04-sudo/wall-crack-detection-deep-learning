"""
evaluate.py
Generates paper-ready figures:
  1. Confusion matrices for all 5 trained variants (grid comparison)
  2. Bar chart comparing accuracy/F1 across variants (the ablation table, visualized)
  3. Gate map visualization for full_csaf -- shows which images/regions the
     gate weights toward CNN (local) vs ViT (global) features
  4. Training curves (accuracy + loss over epochs) for full_csaf
  5. Trainable parameter count comparison across all variants
  6. Sample predictions grid -- correct AND incorrect, with confidence scores
  7. ROC curves + AUC for all 5 variants (overlaid on one axis)

Usage:
    python evaluate.py

Requires:
    - checkpoints/<variant>_best.pth for each variant (from train.py)
    - logs/<variant>_log.json for each variant (from train.py)
    - dataset/test/ folder

Outputs (saved to figures/):
    confusion_matrices.png
    ablation_comparison.png
    gate_visualization.png
    training_curves.png
    parameter_comparison.png
    sample_predictions.png
    roc_curves.png
"""

import os
import json
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc

from csaf_net import build_model, get_device

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VARIANTS = ["cnn_only", "vit_only", "concat", "ungated_cross_attn", "full_csaf"]
VARIANT_LABELS = {
    "cnn_only": "CNN Only",
    "vit_only": "ViT Only",
    "concat": "Concat Fusion",
    "ungated_cross_attn": "Cross-Attn (No Gate)",
    "full_csaf": "CSAF-Net (Full, Gated)",
}


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# batch_size bumped 8 -> 32: fewer, larger batches run noticeably faster on
# MPS/CPU than many small ones, since each batch has fixed overhead
# regardless of size. Drop this back down if you hit memory pressure.
def get_test_loader(data_dir="dataset", batch_size=32):
    test_ds = ImageFolder(os.path.join(data_dir, "test"), transform=get_test_transform())
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return loader, test_ds


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total = len(loader)
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            logits, _ = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            print(f"    batch {i}/{total}", end="\r")
    print()  # newline after progress
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrices(device, data_dir="dataset", checkpoint_dir="checkpoints",
                             out_path="figures/confusion_matrices.png"):
    loader, test_ds = get_test_loader(data_dir)
    class_names = test_ds.classes   # ['Negative', 'Positive']
    print(f"  Test set size: {len(test_ds)} images, batch_size={loader.batch_size}")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for ax, variant in zip(axes, VARIANTS):
        ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {variant} -- checkpoint not found at {ckpt_path}")
            ax.axis("off")
            continue

        print(f"  Evaluating {variant}...")
        model = build_model(variant=variant).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        labels, preds, probs = get_predictions(model, loader, device)
        cm = confusion_matrix(labels, preds)

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(VARIANT_LABELS[variant], fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted")
        if variant == VARIANTS[0]:
            ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_ablation_comparison(log_dir="logs", out_path="figures/ablation_comparison.png"):
    results = {}
    for variant in VARIANTS:
        log_path = os.path.join(log_dir, f"{variant}_log.json")
        if not os.path.exists(log_path):
            print(f"  Skipping {variant} -- log not found at {log_path}")
            continue
        with open(log_path) as f:
            data = json.load(f)
        results[variant] = data["test"]

    variants_present = [v for v in VARIANTS if v in results]
    metrics = ["accuracy", "f1", "precision", "recall"]
    x = np.arange(len(variants_present))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, metric in enumerate(metrics):
        values = [results[v][metric] for v in variants_present]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants_present], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation Study: CSAF-Net Component Contributions")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_gate_visualization(device, data_dir="dataset", checkpoint_dir="checkpoints",
                             out_path="figures/gate_visualization.png", num_samples=6):
    """
    Shows a handful of test images alongside their per-token gate values
    (averaged over channels), reshaped to a grid to show spatial gate behavior.
    Gate close to 1 -> CNN-attended branch dominates (local detail).
    Gate close to 0 -> ViT-attended branch dominates (global context).

    Deliberately picks a MIX of correctly-classified Positive (crack) and
    Negative (no crack) samples -- rather than just the first fixed batch --
    so the figure actually demonstrates the gate behaving differently across
    image types instead of showing 6 near-identical negatives every run.
    """
    ckpt_path = os.path.join(checkpoint_dir, "full_csaf_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  Skipping gate visualization -- checkpoint not found at {ckpt_path}")
        return

    model = build_model(variant="full_csaf").to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Use a bigger, shuffled loader to search across the test set for a good mix
    full_ds = ImageFolder(os.path.join(data_dir, "test"), transform=get_test_transform())
    search_loader = DataLoader(full_ds, batch_size=32, shuffle=True)
    class_names = full_ds.classes   # ['Negative', 'Positive']
    pos_idx = class_names.index("Positive")
    neg_idx = class_names.index("Negative")

    half = num_samples // 2
    picked_images, picked_labels = [], []
    n_pos, n_neg = 0, 0

    with torch.no_grad():
        for images_b, labels_b in search_loader:
            images_b_dev = images_b.to(device)
            logits, _ = model(images_b_dev)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu()

            for img, lab, pred in zip(images_b, labels_b, preds):
                correct = (lab.item() == pred.item())
                if not correct:
                    continue
                if lab.item() == pos_idx and n_pos < half:
                    picked_images.append(img)
                    picked_labels.append(lab)
                    n_pos += 1
                elif lab.item() == neg_idx and n_neg < (num_samples - half):
                    picked_images.append(img)
                    picked_labels.append(lab)
                    n_neg += 1

            if n_pos >= half and n_neg >= (num_samples - half):
                break

    if len(picked_images) < num_samples:
        print(f"  Warning: only found {len(picked_images)} correctly-classified "
              f"samples (wanted {num_samples}) -- figure will show fewer.")
        num_samples = len(picked_images)

    images = torch.stack(picked_images[:num_samples]).to(device)
    labels = torch.stack(picked_labels[:num_samples])
    test_ds = full_ds  # reuse for .classes below

    with torch.no_grad():
        logits, gate = model(images)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()

    # gate: (B, N_cnn, dim) -- average over channel dim to get per-token scalar
    gate_scalar = gate.mean(dim=-1).cpu().numpy()   # (B, N_cnn)
    grid_size = int(np.sqrt(gate_scalar.shape[1]))   # e.g. 49 -> 7x7

    # unnormalize images for display
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)

    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))

    for i in range(num_samples):
        img = images[i].cpu().numpy() * std + mean
        img = np.clip(img.transpose(1, 2, 0), 0, 1)

        axes[0, i].imshow(img)
        true_label = test_ds.classes[labels[i]]
        pred_label = test_ds.classes[preds[i]]
        axes[0, i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
        axes[0, i].axis("off")

        gate_map = gate_scalar[i].reshape(grid_size, grid_size)
        im = axes[1, i].imshow(gate_map, cmap="RdBu_r", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Gate map")

    fig.suptitle("Gate Behavior: Red = CNN-dominant (local detail), Blue = ViT-dominant (global context)",
                  fontsize=10)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_training_curves(variant="full_csaf", log_dir="logs",
                          out_path="figures/training_curves.png"):
    """
    Line plots of train/val accuracy and train/val loss across epochs,
    for a single variant (defaults to full_csaf). Standard figure for
    showing the model converged and didn't badly overfit.
    """
    log_path = os.path.join(log_dir, f"{variant}_log.json")
    if not os.path.exists(log_path):
        print(f"  Skipping training curves -- log not found at {log_path}")
        return

    with open(log_path) as f:
        data = json.load(f)
    history = data["history"]

    epochs = [h["epoch"] for h in history]
    train_acc = [h["train"]["accuracy"] for h in history]
    val_acc = [h["val"]["accuracy"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    ax1.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{VARIANT_LABELS.get(variant, variant)} -- Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_loss, marker="o", color="tab:red", label="Train Loss")
    ax2.plot(epochs, val_loss, marker="o", color="tab:orange", label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{VARIANT_LABELS.get(variant, variant)} -- Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_parameter_comparison(out_path="figures/parameter_comparison.png"):
    """
    Bar chart of trainable parameter counts across all 5 variants.
    Backbones are frozen by default (see backbones.py), so this mostly
    reflects the size of each fusion module + classification head --
    useful for showing CSAF-Net's gate adds negligible parameters for
    its accuracy/F1/recall gain over the ungated version.
    """
    counts = {}
    for variant in VARIANTS:
        try:
            model = build_model(variant=variant)
        except Exception as e:
            print(f"  Skipping {variant} -- could not build model: {e}")
            continue
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        counts[variant] = {"trainable": trainable, "total": total}
        del model

    variants_present = [v for v in VARIANTS if v in counts]
    trainable_vals = [counts[v]["trainable"] / 1e6 for v in variants_present]   # millions
    total_vals = [counts[v]["total"] / 1e6 for v in variants_present]

    x = np.arange(len(variants_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, total_vals, width, label="Total Parameters", color="lightgray")
    ax.bar(x + width / 2, trainable_vals, width, label="Trainable Parameters", color="tab:blue")

    for i, v in enumerate(trainable_vals):
        ax.text(x[i] + width / 2, v + max(trainable_vals) * 0.02, f"{v:.2f}M",
                 ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants_present], rotation=15, ha="right")
    ax.set_ylabel("Parameters (Millions)")
    ax.set_title("Model Size Comparison Across Variants")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_sample_predictions(device, data_dir="dataset", checkpoint_dir="checkpoints",
                             variant="full_csaf", out_path="figures/sample_predictions.png",
                             num_correct=4, num_incorrect=2):
    """
    Grid of real test images with predicted label + confidence score,
    including a couple of misclassified examples for an honest picture
    of where the model still struggles.
    """
    ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  Skipping sample predictions -- checkpoint not found at {ckpt_path}")
        return

    model = build_model(variant=variant).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    full_ds = ImageFolder(os.path.join(data_dir, "test"), transform=get_test_transform())
    search_loader = DataLoader(full_ds, batch_size=32, shuffle=True)
    class_names = full_ds.classes

    correct_samples, incorrect_samples = [], []

    with torch.no_grad():
        for images_b, labels_b in search_loader:
            images_dev = images_b.to(device)
            logits, _ = model(images_dev)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > 0.5).long()

            for img, lab, pred, prob in zip(images_b, labels_b, preds, probs):
                entry = (img, lab.item(), pred.item(), prob.item())
                if lab.item() == pred.item() and len(correct_samples) < num_correct:
                    correct_samples.append(entry)
                elif lab.item() != pred.item() and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(entry)

            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break

    samples = correct_samples + incorrect_samples
    if not samples:
        print("  Skipping sample predictions -- no samples found")
        return

    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)

    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, (img, lab, pred, prob) in zip(axes, samples):
        img_np = img.numpy() * std + mean
        img_np = np.clip(img_np.transpose(1, 2, 0), 0, 1)
        ax.imshow(img_np)

        true_label = class_names[lab]
        pred_label = class_names[pred]
        confidence = prob if pred == 1 else (1 - prob)
        correct = (lab == pred)

        title_color = "green" if correct else "red"
        status = "Correct" if correct else "Wrong"
        ax.set_title(f"{status}\nTrue: {true_label} | Pred: {pred_label}\nConf: {confidence*100:.1f}%",
                     fontsize=9, color=title_color)
        ax.axis("off")

    fig.suptitle(f"Sample Predictions -- {VARIANT_LABELS.get(variant, variant)}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_roc_curves(device, data_dir="dataset", checkpoint_dir="checkpoints",
                     out_path="figures/roc_curves.png"):
    """
    ROC curve + AUC for all 5 variants, overlaid on one axis.
    A more threshold-independent view of separability than the confusion
    matrix / accuracy-at-0.5 alone.
    """
    loader, test_ds = get_test_loader(data_dir)

    fig, ax = plt.subplots(figsize=(7.5, 7))

    for variant in VARIANTS:
        ckpt_path = os.path.join(checkpoint_dir, f"{variant}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {variant} -- checkpoint not found at {ckpt_path}")
            continue

        print(f"  Evaluating {variant} for ROC...")
        model = build_model(variant=variant).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        labels, preds, probs = get_predictions(model, loader, device)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, linewidth=2,
                label=f"{VARIANT_LABELS[variant]} (AUC = {roc_auc:.3f})")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- All Variants")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}\n")

    print("Generating confusion matrices for all variants...")
    plot_confusion_matrices(device)

    print("\nGenerating ablation comparison chart...")
    plot_ablation_comparison()

    print("\nGenerating gate visualization for full_csaf...")
    plot_gate_visualization(device)

    print("\nGenerating training curves for full_csaf...")
    plot_training_curves(variant="full_csaf")

    print("\nGenerating parameter count comparison...")
    plot_parameter_comparison()

    print("\nGenerating sample predictions grid...")
    plot_sample_predictions(device, variant="full_csaf")

    print("\nGenerating ROC curves for all variants...")
    plot_roc_curves(device)

    print("\nAll figures saved to figures/")