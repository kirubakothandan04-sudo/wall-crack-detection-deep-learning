"""Grad-CAM for a single-logit crack classifier (CSAF-Net).

Usage (adapt the layer and forward call to your model):

    cam_tool = GradCAM(model, target_layer=model.cnn.features.denseblock3)
    cam, prob = cam_tool(x)                 # x: 1x3x224x224, ImageNet-normalised
    overlay = overlay_cam(rgb_uint8, cam)   # rgb_uint8: HxWx3 original image

Checks that matter:
 1. The target is the CRACK logit (raw output before sigmoid), not the probability.
 2. x.requires_grad_(True): the backbones are frozen, so without this no gradient
    reaches the feature map and the result is empty or wrong.
 3. In-place ReLU (DenseNet default) breaks hooks: call disable_inplace(model).
 4. Run model.eval() and do NOT wrap the call in torch.no_grad().
 5. Use a finer layer (denseblock3, 14x14) rather than only the last 7x7 map.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def pick_logit(out):
    """Models may return (logit, gate, ...). Return the single crack logit."""
    if isinstance(out, (tuple, list)):
        for o in out:
            if torch.is_tensor(o) and o.numel() == 1:
                return o.reshape(-1)[0]
        out = out[0]
    return out.reshape(-1)[0]


def disable_inplace(model):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
            m.inplace = False


class GradCAM:
    def __init__(self, model, target_layer, plus=False):
        self.model, self.plus = model.eval(), plus
        disable_inplace(model)
        self.acts = self.grads = None
        target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:   # skipped when the model runs under no_grad
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, **kwargs):
        x = x.clone().requires_grad_(True)
        self.model.zero_grad()
        logit = pick_logit(self.model(x, **kwargs))   # crack logit
        logit.backward()
        A, G = self.acts.detach()[0], self.grads.detach()[0]   # C x h x w
        if self.plus:                                        # Grad-CAM++
            g2, g3 = G ** 2, G ** 3
            denom = 2 * g2 + A.sum((1, 2), keepdim=True) * g3
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(G)).sum((1, 2))
        else:
            w = G.mean((1, 2))
        cam = F.relu((w[:, None, None] * A).sum(0))
        cam = cam - cam.min()
        cam = (cam / cam.max()).cpu().numpy() if cam.max() > 0 else cam.cpu().numpy()
        return cam, torch.sigmoid(logit).item()


def overlay_cam(rgb, cam, alpha=0.5, floor=0.35):
    """Blend heat only where the map is strong, so the image stays readable."""
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    mask = (cam > floor)[..., None] * alpha * cam[..., None]
    return (rgb * (1 - mask) + heat * mask).astype(np.uint8)


@torch.no_grad()
def occlusion_check(model, x, box, fill=0.0, **kwargs):
    """Does confidence depend on a region? box = (x0, y0, x1, y1) in 224 px.
    Run once on the crack and once on the window frame and compare."""
    model.eval()
    base = torch.sigmoid(pick_logit(model(x, **kwargs))).item()
    x2 = x.clone()
    x0, y0, x1, y1 = box
    x2[:, :, y0:y1, x0:x1] = fill
    after = torch.sigmoid(pick_logit(model(x2, **kwargs))).item()
    return base, after, base - after