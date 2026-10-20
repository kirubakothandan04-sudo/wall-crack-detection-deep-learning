"""
explain_compare.py
Compares several ways of showing where CSAF-Net (full model) finds a crack and,
when you provide a hand-drawn crack mask, measures how well each method points
at the crack.

Usage:
    export HF_HUB_OFFLINE=1
    python explain_compare.py photo1.jpg photo2.jpg ...
A mask named <photo>_mask.png (white = crack) next to a photo is used
automatically (make it with annotate_cracks.py). Panels go to explain_out/.
"""
import os
import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csaf_net import build_model, get_device
from gradcam import overlay_heatmap, combined_crack_mask
from sliding_window_detect import get_transform

SIZE = 224


def norm(h):
    h = np.maximum(h, 0)
    return h / h.max() if h.max() > 0 else h


def upsample(cam):
    return cv2.resize(cam.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def layer_cam(model, layer, x, kind):
    """kind='grad' -> Grad-CAM (channel weights); kind='layer' -> LayerCAM (per-location weights)."""
    store = {}

    def fwd(_module, _inp, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = layer.register_forward_hook(fwd)
    xg = x.clone().detach().requires_grad_(True)
    try:
        logit = model(xg)[0].sum()
        model.zero_grad()
        logit.backward()
    finally:
        handle.remove()
    a, g = store["a"].detach(), store["g"].detach()
    if kind == "grad":
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1))
    else:
        cam = F.relu((F.relu(g) * a).sum(dim=1))
    return cam[0].cpu().numpy()


def smoothgrad(model, x, n=16, sigma=0.15):
    """Gradient of the final logit w.r.t. the input pixels (CNN + ViT + gate), averaged over noisy copies."""
    spread = float(x.max() - x.min())
    acc = np.zeros((SIZE, SIZE), dtype=np.float64)
    for _ in range(n):
        xn = (x + torch.randn_like(x) * sigma * spread).detach().requires_grad_(True)
        logit = model(xn)[0].sum()
        (g,) = torch.autograd.grad(logit, xn)
        acc += g.abs().amax(dim=1)[0].detach().cpu().numpy()
    acc = cv2.GaussianBlur((acc / n).astype(np.float32), (0, 0), 3)
    return norm(acc)


@torch.no_grad()
def occlusion(model, x, win=32, stride=16, bs=32):
    """Drop in the final logit when a square is replaced by the dataset-mean colour (full model)."""
    base = model(x)[0].item()
    pos = [(i, j) for i in range(0, SIZE - win + 1, stride) for j in range(0, SIZE - win + 1, stride)]
    heat = np.zeros((SIZE, SIZE))
    cnt = np.zeros((SIZE, SIZE))
    for s in range(0, len(pos), bs):
        chunk = pos[s:s + bs]
        xb = x.repeat(len(chunk), 1, 1, 1)
        for k, (i, j) in enumerate(chunk):
            xb[k, :, i:i + win, j:j + win] = 0.0
        drops = (base - model(xb)[0]).cpu().numpy()
        for (i, j), d in zip(chunk, drops):
            heat[i:i + win, j:j + win] += d
            cnt[i:i + win, j:j + win] += 1
    return norm(heat / np.maximum(cnt, 1))


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 20


def metrics(heat, mask, r):
    """hit: strongest point lies within r px of the drawn crack.
    enrichment: share of heat inside that band divided by the band's share of the image (1.0 = chance)."""
    if heat.max() <= 0:
        return False, 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near = cv2.dilate(mask.astype(np.uint8), k) > 0
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    share = heat[near].sum() / max(heat.sum(), 1e-9)
    return bool(near[y, x]), float(share / max(near.mean(), 1e-9))


def seg_scores(pred, gt, r):
    """Boundary-style scores with a tolerance band of r px:
    precision = share of predicted mask pixels within r px of the drawn crack,
    recall    = share of drawn-crack pixels within r px of the predicted mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    gt_band = cv2.dilate(gt.astype(np.uint8), k) > 0
    pr_band = cv2.dilate(pred.astype(np.uint8), k) > 0
    prec = float((pred & gt_band).sum() / max(pred.sum(), 1))
    rec = float((gt & pr_band).sum() / max(gt.sum(), 1))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--r", type=int, default=16, help="band (px at 224x224) around the drawn crack that counts as on the crack")
    ap.add_argument("--out", default="explain_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    dev = get_device()
    model = build_model(variant="full_csaf").to(dev)
    model.load_state_dict(torch.load("checkpoints/full_csaf_best.pth", map_location=dev))
    model.eval()
    tf = get_transform()
    feats = model.cnn_backbone.features
    layers = {
        "Grad-CAM 7x7 (current)": (feats, "grad"),
        "Grad-CAM 14x14": (feats.denseblock3, "grad"),
        "LayerCAM 14x14": (feats.denseblock3, "layer"),
        "LayerCAM 28x28": (feats.denseblock2, "layer"),
    }
    scores = {}
    seg_all = {}

    for path in a.images:
        pil = Image.open(path).convert("RGB")
        x = tf(pil).unsqueeze(0).to(dev)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)[0]).item()
        maps = {n: norm(upsample(layer_cam(model, l, x, k))) for n, (l, k) in layers.items()}
        maps["SmoothGrad (full model)"] = smoothgrad(model, x)
        maps["Occlusion (full model)"] = occlusion(model, x)

        mp = os.path.splitext(path)[0] + "_mask.png"
        mask = load_mask(mp) if os.path.exists(mp) else None
        line = f"{os.path.basename(path)}  P(crack)={prob:.3f}"
        if mask is not None:
            for n, h in maps.items():
                hit, enr = metrics(h, mask, a.r)
                scores.setdefault(n, []).append((hit, enr))
            line += "  (mask found)"
        print(line)

        if mask is not None:
            seg_sources = {
                "Canny only (no attention)": None,
                "Canny + Grad-CAM 14x14 (current app)": maps["Grad-CAM 14x14"],
                "Canny + Grad-CAM 7x7": maps["Grad-CAM 7x7 (current)"],
                "Canny + SmoothGrad": maps["SmoothGrad (full model)"],
                "Canny + Occlusion": maps["Occlusion (full model)"],
            }
            seg_masks = {}
            for n, h in seg_sources.items():
                if h is None:
                    _, pm = combined_crack_mask(np.ones((SIZE, SIZE), dtype=np.float32), pil, cam_percentile=0)
                else:
                    _, pm = combined_crack_mask(h, pil, cam_percentile=80)
                seg_masks[n] = pm
                seg_all.setdefault(n, []).append(seg_scores(pm, mask, a.r))
            o2 = np.array(pil.resize((SIZE, SIZE)))
            fig2, ax2 = plt.subplots(1, len(seg_masks) + 2, figsize=(2.6 * (len(seg_masks) + 2), 3.3))
            ax2[0].imshow(o2)
            ax2[0].set_title("Original", fontsize=8)
            g = o2.copy()
            g[mask] = (0, 255, 0)
            ax2[1].imshow(g)
            ax2[1].set_title("Drawn crack", fontsize=8)
            for ax, (n, pm) in zip(ax2[2:], seg_masks.items()):
                v = o2.copy()
                v[pm] = (255, 0, 0)
                ax.imshow(v)
                ax.set_title(n, fontsize=7)
            for ax in ax2:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(a.out, os.path.splitext(os.path.basename(path))[0] + "_seg.png"), dpi=130, bbox_inches="tight")
            plt.close()

        orig = np.array(pil.resize((SIZE, SIZE)))
        cols = ["Original"] + list(maps) + (["Drawn crack"] if mask is not None else [])
        fig, axs = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 3.3))
        axs[0].imshow(orig)
        for ax, (n, h) in zip(axs[1:], maps.items()):
            ax.imshow(overlay_heatmap(h, pil))
        if mask is not None:
            ov = orig.copy()
            ov[mask] = (0, 255, 0)
            axs[-1].imshow(ov)
        for ax, c in zip(axs, cols):
            ax.set_title(c, fontsize=8)
            ax.axis("off")
        fig.suptitle(line, fontsize=10)
        plt.tight_layout()
        outp = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0] + "_explain.png")
        plt.savefig(outp, dpi=130, bbox_inches="tight")
        plt.close()

    if scores:
        n_img = len(next(iter(scores.values())))
        print(f"\nOn {n_img} annotated image(s), band r={a.r}px:")
        print(f"{'method':26s} strongest point on crack   enrichment (1.0 = chance)")
        for n, v in scores.items():
            hits = sum(h for h, _ in v)
            enr = float(np.mean([e for _, e in v]))
            print(f"{n:26s} {hits}/{n_img} ({100 * hits / n_img:.0f}%)            {enr:.2f}")

    if seg_all:
        n_img = len(next(iter(seg_all.values())))
        print(f"\nSegmentation (Canny edges kept inside the attended region) vs drawn crack, {n_img} image(s), band r={a.r}px:")
        print(f"{'mask source':38s} precision   recall   F1")
        for n, v in seg_all.items():
            arr = np.array(v)
            print(f"{n:38s} {arr[:, 0].mean():.2f}        {arr[:, 1].mean():.2f}     {arr[:, 2].mean():.2f}")
    print(f"\nPanels saved in {a.out}/")


if __name__ == "__main__":
    main()