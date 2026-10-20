"""finetune_real.py
Fine-tune an already-trained CSAF-Net variant on SDNET2018 + your REAL patches.

What changes versus train.py (everything else is the same model and head):
  * starts from checkpoints/<variant>_best.pth (does not start from scratch)
  * unfreezes the last DenseNet block (denseblock4 + norm5) and the last 2 ViT
    layers, with a small learning rate (1e-5); fusion + head use 1e-4
  * stronger augmentation (random crop, flips, rotation, colour jitter, blur)
  * real patches are repeated so they are not drowned out by SDNET2018
  * BatchNorm kept in eval mode (batch size 8 is too small to update it safely)
  * best epoch chosen on real-val F1 and SDNET val F1 together, so the model
    does not forget SDNET2018

Writes checkpoints_real/<variant>_best.pth and logs_real/<variant>_log.json.
Your ORIGINAL checkpoints/ are never touched.

Run (after make_real_dataset.py):
    python finetune_real.py --epochs 6
"""
import os, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from csaf_net import build_model, get_device

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="full_csaf",
                choices=["concat", "ungated_cross_attn", "scalar_gate", "full_csaf"])
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr-head", type=float, default=1e-4)
ap.add_argument("--lr-backbone", type=float, default=1e-5)
ap.add_argument("--sdnet-n", type=int, default=1200, help="SDNET2018 train images used (keeps epochs short)")
ap.add_argument("--real-repeat", type=int, default=3)
ap.add_argument("--data-dir", default="dataset")
ap.add_argument("--real-dir", default="patches_real")
ap.add_argument("--base-ckpt-dir", default="checkpoints")
ap.add_argument("--out-dir", default="checkpoints_real")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

device = get_device()
print("Device:", device)

# ---- data ----
sd_train = ImageFolder(os.path.join(a.data_dir, "train"), transform=train_tf)
real_train = ImageFolder(os.path.join(a.real_dir, "train"), transform=train_tf)
real_val = ImageFolder(os.path.join(a.real_dir, "val"), transform=eval_tf)
sd_val = ImageFolder(os.path.join(a.data_dir, "val"), transform=eval_tf)
sd_test = ImageFolder(os.path.join(a.data_dir, "test"), transform=eval_tf)
for ds in (sd_train, real_train, real_val, sd_val, sd_test):
    assert ds.class_to_idx == {"Negative": 0, "Positive": 1}, ds.class_to_idx

idx = torch.randperm(len(sd_train), generator=torch.Generator().manual_seed(a.seed))[:a.sdnet_n].tolist()
train_ds = ConcatDataset([Subset(sd_train, idx)] + [real_train] * a.real_repeat)
vidx = torch.randperm(len(sd_val), generator=torch.Generator().manual_seed(1))[:300].tolist()
print(f"Train: {a.sdnet_n} SDNET + {len(real_train)} real x{a.real_repeat} = {len(train_ds)} images/epoch")
print(f"Val:   {len(real_val)} real patches, {len(vidx)} SDNET images")

train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
real_val_loader = DataLoader(real_val, batch_size=16, shuffle=False, num_workers=0)
sd_val_loader = DataLoader(Subset(sd_val, vidx), batch_size=16, shuffle=False, num_workers=0)
sd_test_loader = DataLoader(sd_test, batch_size=16, shuffle=False, num_workers=0)

# ---- model: load original weights, unfreeze last blocks ----
model = build_model(variant=a.variant).to(device)
base = os.path.join(a.base_ckpt_dir, f"{a.variant}_best.pth")
model.load_state_dict(torch.load(base, map_location=device))
print("Loaded", base)

for p in model.parameters():
    p.requires_grad = False
backbone_params = []
for mod in (model.cnn_backbone.features.denseblock4, model.cnn_backbone.features.norm5,
            model.vit_backbone.vit.encoder.layer[-2], model.vit_backbone.vit.encoder.layer[-1],
            model.vit_backbone.vit.layernorm):
    for p in mod.parameters():
        p.requires_grad = True
        backbone_params.append(p)
bb_ids = {id(p) for p in backbone_params}
other = [p for n, p in model.named_parameters()
         if (n.startswith(("fusion", "head", "cnn_backbone.proj", "cnn_backbone.norm",
                           "vit_backbone.proj", "vit_backbone.norm")))]
for p in other:
    p.requires_grad = True
other = [p for p in other if id(p) not in bb_ids]
print(f"Trainable: {sum(p.numel() for p in backbone_params)/1e6:.1f}M backbone, "
      f"{sum(p.numel() for p in other)/1e6:.2f}M fusion/head")

opt = torch.optim.AdamW([{"params": backbone_params, "lr": a.lr_backbone},
                         {"params": other, "lr": a.lr_head}], weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
crit = nn.BCEWithLogitsLoss()


def set_train():
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


@torch.no_grad()
def evaluate(loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits, _ = model(x.to(device))
        ps.extend((torch.sigmoid(logits) > 0.5).long().cpu().tolist())
        ys.extend(y.tolist())
    return dict(accuracy=accuracy_score(ys, ps), f1=f1_score(ys, ps, zero_division=0),
                precision=precision_score(ys, ps, zero_division=0),
                recall=recall_score(ys, ps, zero_division=0))


os.makedirs(a.out_dir, exist_ok=True); os.makedirs("logs_real", exist_ok=True)
ckpt = os.path.join(a.out_dir, f"{a.variant}_best.pth")
best, history = -1.0, []
before_real, before_sd = evaluate(real_val_loader), evaluate(sd_val_loader)
print(f"\nBEFORE fine-tune  real-val F1 {before_real['f1']:.3f} (acc {before_real['accuracy']:.3f})   "
      f"SDNET-val F1 {before_sd['f1']:.3f}")

for ep in range(1, a.epochs + 1):
    t0 = time.time(); set_train(); tot = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device, dtype=torch.float32)
        opt.zero_grad()
        logits, _ = model(x)
        loss = crit(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        tot += loss.item() * x.size(0)
    sched.step()
    r, s = evaluate(real_val_loader), evaluate(sd_val_loader)
    score = 0.5 * r["f1"] + 0.5 * s["f1"]
    print(f"Epoch {ep}/{a.epochs} ({time.time()-t0:.0f}s) loss {tot/len(train_ds):.4f} | "
          f"real-val F1 {r['f1']:.3f} acc {r['accuracy']:.3f} | SDNET-val F1 {s['f1']:.3f}")
    history.append(dict(epoch=ep, loss=tot / len(train_ds), real_val=r, sdnet_val=s))
    if score > best:
        best = score
        torch.save(model.state_dict(), ckpt)
        print("  -> saved best (combined val F1 %.3f)" % score)

model.load_state_dict(torch.load(ckpt, map_location=device))
test = evaluate(sd_test_loader)
print("\nSDNET2018 test after fine-tuning (check it did not collapse):",
      {k: round(v, 4) for k, v in test.items()})
with open(os.path.join("logs_real", f"{a.variant}_log.json"), "w") as f:
    json.dump(dict(args=vars(a), before_real_val=before_real, before_sdnet_val=before_sd,
                   history=history, sdnet_test_after=test), f, indent=2)
print("Saved", ckpt, "and logs_real/%s_log.json" % a.variant)
print("Now measure on held-out real photos:\n  python eval_real_photos.py --root real_split/test --ckpt-dir checkpoints_real")