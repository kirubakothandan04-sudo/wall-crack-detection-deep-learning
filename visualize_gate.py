import torch
import matplotlib.pyplot as plt
from csaf_net import build_model, get_device
from train import get_dataloaders

device = get_device()
model = build_model(variant="full_csaf").to(device)
model.load_state_dict(torch.load("checkpoints/full_csaf_best.pth", map_location=device))
model.eval()

_, _, test_loader = get_dataloaders(batch_size=8)

gate_means = []
labels_all = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        _, gate = model(images)
        per_image_mean = gate.mean(dim=(1, 2))
        gate_means.extend(per_image_mean.cpu().numpy())
        labels_all.extend(labels.numpy())

plt.figure(figsize=(6,4))
plt.hist([g for g,l in zip(gate_means, labels_all) if l==0], bins=20, alpha=0.6, label="No Crack")
plt.hist([g for g,l in zip(gate_means, labels_all) if l==1], bins=20, alpha=0.6, label="Crack")
plt.xlabel("Mean gate value (closer to 1 = favors CNN branch)")
plt.ylabel("Count")
plt.legend()
plt.title("Gate activation distribution: Crack vs No-Crack")
plt.savefig("figures/gate_distribution.png", dpi=150)
print("Saved to figures/gate_distribution.png")
