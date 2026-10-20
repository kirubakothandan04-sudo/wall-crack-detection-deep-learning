import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from csaf_net import build_model, get_device
from sliding_window_detect import get_transform

dev = get_device()
ds = ImageFolder("dataset/test", transform=get_transform())
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
model = build_model(variant="full_csaf").to(dev)
model.load_state_dict(torch.load("checkpoints/full_csaf_best.pth", map_location=dev))
model.eval()

G = []
with torch.no_grad():
    for x, _ in loader:
        G.append(model(x.to(dev))[1].detach().cpu().numpy())
G = np.concatenate(G)            # (images, tokens, channels)

total = G.var()
chan = G.mean(axis=(0, 1)).var()   # variance explained by channel identity
img = G.mean(axis=(1, 2)).var()    # by which image it is
tok = G.mean(axis=(0, 2)).var()    # by which spatial token
print(f"Gate shape {G.shape}, total variance {total:.5f}")
print(f"Share of variance from channel identity: {chan/total*100:.1f}%")
print(f"Share from image identity:               {img/total*100:.1f}%")
print(f"Share from spatial token position:       {tok/total*100:.1f}%")
print(f"Std across tokens within a channel, averaged: {G.std(axis=1).mean():.4f}")
print(f"Std across images, per token and channel, averaged: {G.std(axis=0).mean():.4f}")
