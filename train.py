"""
train.py
Generic training script for CSAF-Net and all ablation variants.
Same script trains any variant -- just pass --variant.

Usage:
    python train.py --variant full_csaf --epochs 5
    python train.py --variant cnn_only --epochs 5
    python train.py --variant vit_only --epochs 5
    python train.py --variant concat --epochs 5
    python train.py --variant ungated_cross_attn --epochs 5
    python train.py --variant scalar_gate --epochs 5

Expects dataset/ folder from prepare_dataset.py:
    dataset/train/Positive, dataset/train/Negative
    dataset/val/Positive,   dataset/val/Negative
    dataset/test/Positive,  dataset/test/Negative
"""

import os
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from csaf_net import build_model, get_device


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(data_dir="dataset", batch_size=8, num_workers=0):
    train_ds = ImageFolder(os.path.join(data_dir, "train"), transform=get_transforms(train=True))
    val_ds = ImageFolder(os.path.join(data_dir, "val"), transform=get_transforms(train=False))
    test_ds = ImageFolder(os.path.join(data_dir, "test"), transform=get_transforms(train=False))

    # sanity check label mapping is consistent (Negative=0, Positive=1 expected)
    print(f"Class mapping: {train_ds.class_to_idx}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            if train:
                optimizer.zero_grad()

            logits, _ = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train_model(variant, epochs=5, batch_size=8, lr=1e-4, data_dir="dataset",
                 checkpoint_dir="checkpoints", log_dir="logs"):
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training variant: {variant}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size)

    model = build_model(variant=variant).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history = []
    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(checkpoint_dir, f"{variant}_best.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s)")
        print(f"  Train - loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}  f1: {train_metrics['f1']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}  f1: {val_metrics['f1']:.4f}")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best model saved (val f1: {best_val_f1:.4f})")

    # final test evaluation using best checkpoint
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal Test Metrics ({variant}):")
    print(f"  accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  f1:        {test_metrics['f1']:.4f}")
    print(f"  precision: {test_metrics['precision']:.4f}")
    print(f"  recall:    {test_metrics['recall']:.4f}")

    log_path = os.path.join(log_dir, f"{variant}_log.json")
    with open(log_path, "w") as f:
        json.dump({"history": history, "test": test_metrics}, f, indent=2)
    print(f"\nLog saved to {log_path}")
    print(f"Checkpoint saved to {best_checkpoint_path}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="full_csaf",
                         choices=["cnn_only", "vit_only", "concat", "ungated_cross_attn",
                                  "scalar_gate", "full_csaf"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="dataset")
    args = parser.parse_args()

    train_model(
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
    )