import glob, torch
from PIL import Image
from csaf_net import build_model, get_device
from sliding_window_detect import get_transform

dev = get_device()
m = build_model(variant="full_csaf").to(dev)
m.load_state_dict(torch.load("checkpoints/full_csaf_best.pth", map_location=dev))
m.eval()
t = get_transform()
paths = glob.glob("dataset/test/Positive/*")[:100]
confs = []
for p in paths:
    x = t(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        confs.append(torch.sigmoid(m(x)[0]).item())
for th in (0.5, 0.93, 0.99):
    print(th, sum(c > th for c in confs) / len(confs))
