# probe_cues.py
import os, sys, glob
import torch
from PIL import Image
from csaf_net import build_model, get_device
from sliding_window_detect import get_transform

dev, tf = get_device(), get_transform()
models = {}
for v in ["full_csaf", "cnn_only", "vit_only"]:
    m = build_model(variant=v).to(dev)
    m.load_state_dict(torch.load(f"checkpoints/{v}_best.pth", map_location=dev))
    m.eval()
    models[v] = m

for f in (sys.argv[1:] or ["probe/window", "probe/crack", "probe/wall"]):
    paths = sorted(p for p in glob.glob(os.path.join(f, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        print(f"\n{f}: no images")
        continue
    print(f"\n{f} ({len(paths)} crops)")
    for v, m in models.items():
        ps = []
        with torch.no_grad():
            for p in paths:
                x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
                logits, _ = m(x)
                ps.append(torch.sigmoid(logits).item())
        print(f"  {v:10s} mean P(crack) = {sum(ps) / len(ps):.2f}   > 0.5: {sum(p > 0.5 for p in ps)}/{len(ps)}")