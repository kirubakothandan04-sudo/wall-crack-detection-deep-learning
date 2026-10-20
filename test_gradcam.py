"""Compare YOUR gradcam.py with the new gradcam_csaf.py on one image,
then run an occlusion check.

Put these in the WallCrackDetection folder:  gradcam.py (yours),
gradcam_csaf.py (new), test_gradcam.py (this file).

Run:   python test_gradcam.py crack.jpg
Output: gradcam_compare.png  (original | your gradcam.py | gradcam_csaf.py)

Lines marked  # <-- CHANGE  may need matching to your project.
"""
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import gradcam as old                           # your existing file
import gradcam_csaf as new                      # the new file
from predict_csaf import load_model             # <-- CHANGE: where load_model lives

VARIANT = "full_csaf"
path = sys.argv[1] if len(sys.argv) > 1 else "crack.jpg"

model, device = load_model(variant=VARIANT)     # <-- CHANGE if the call differs
model.eval()

prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img = Image.open(path).convert("RGB")
x = prep(img).unsqueeze(0).to(device)
img224 = img.resize((224, 224))


def to_pil(o):
    return o if isinstance(o, Image.Image) else Image.fromarray(np.asarray(o))


def crack_prob(t):
    with torch.no_grad():
        out = model(t)                          # <-- CHANGE if forward needs more args
        return torch.sigmoid(new.pick_logit(out)).item()


def find_layer(m, name):
    for n, mod in m.named_modules():
        if n.endswith(name):
            return n, mod
    return None, None


# ---- 1a. Your existing Grad-CAM ----
old_engine = old.GradCAM(model, old.get_target_layer(model, VARIANT))
cam_old, conf_old = old_engine.generate(x, device)
img_old = to_pil(old.overlay_heatmap(cam_old, img224)).resize((224, 224))

# ---- 1b. New Grad-CAM (finer layer, crack logit, in-place ReLU off) ----
lname, layer = find_layer(model, "denseblock3")
if layer is None:
    lname, layer = find_layer(model, "denseblock4")
print("New Grad-CAM target layer:", lname)
new_engine = new.GradCAM(model, layer, plus=True)      # plus=True is Grad-CAM++
cam_new, prob_new = new_engine(x)
img_new = Image.fromarray(new.overlay_cam(np.array(img224), cam_new))

canvas = Image.new("RGB", (224 * 3, 224))
for i, im in enumerate([img224, img_old, img_new]):
    canvas.paste(im, (224 * i, 0))
canvas.save("gradcam_compare.png")
print(f"Confidence: old {conf_old:.3f}, new {prob_new:.3f}  -> saved gradcam_compare.png")

# ---- 2. Occlusion test (uses the raw logit: the probability is saturated near 1.0) ----
def crack_logit(t_):
    with torch.no_grad():
        return new.pick_logit(model(t_)).item()

STEP = 28
base = crack_logit(x)
print(f"\nBaseline crack logit: {base:.3f}  (probability {crack_prob(x):.3f})")
grid = np.zeros((224 // STEP, 224 // STEP))
for gy, y0 in enumerate(range(0, 224, STEP)):
    for gx, x0 in enumerate(range(0, 224, STEP)):
        t_ = x.clone()
        t_[:, :, y0:y0 + STEP, x0:x0 + STEP] = 0.0     # 0 = mean colour after normalisation
        grid[gy, gx] = base - crack_logit(t_)

# save the occlusion map next to the Grad-CAM for comparison
g = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)
occ = new.overlay_cam(np.array(img224), g, alpha=0.6, floor=0.2)
Image.fromarray(occ).save("occlusion_map.png")

order = np.dstack(np.unravel_index(np.argsort(-grid.ravel()), grid.shape))[0][:5]
print("Top 5 squares (28x28) where covering them lowers the logit most:")
for gy, gx in order:
    print(f"  x={gx*STEP:3d}-{gx*STEP+STEP:3d}, y={gy*STEP:3d}-{gy*STEP+STEP:3d}  logit drop {grid[gy, gx]:+.3f}")
print("\nSaved occlusion_map.png. Compare it with gradcam_compare.png.")
print("Both should be bright on the crack. Bright on the frame/corners -> context, report as a limitation.")