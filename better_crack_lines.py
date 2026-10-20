"""better_crack_lines.py
Drop-in replacement for analyze_crack() in predict_csaf.py.

Same return value as before:  (crack_type, edges_img, annotated_img)
  crack_type     "Horizontal Crack" / "Vertical Crack" / "Diagonal Crack" / "Fine Crack"
  edges_img      2-D uint8 mask of the kept crack-like pieces
  annotated_img  RGB copy of the photo with those pieces drawn in red

Why it differs from Canny + Hough: concrete grain makes thousands of tiny
edges, and Hough joins them into "lines". This version looks for DARK THIN
ridges (black-hat filter), then keeps only connected pieces that are long,
thin and not a solid bar. Thick straight bars (window frames, door edges)
are rejected by the fill test.

It is still a heuristic aid, not a validated crack segmenter.
"""
import numpy as np
import cv2
from PIL import Image

WORK_MAX_SIDE = 640      # analyse at most this big (speed); result is scaled back


def _odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def crack_mask(rgb, k_frac=0.04, sigma=2.5, min_len_frac=0.12,
               min_elong=3.0, max_fill=0.65):
    """Return a uint8 mask (0/255) of crack-like pieces for an RGB uint8 image."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    k = _odd(max(9, min(h, w) * k_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    thr = bh.mean() + sigma * bh.std()
    m = (bh > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros((h, w), np.uint8)
    min_len = min_len_frac * min(h, w)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        ys, xs = np.where(lab == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (cx, cy), (rw, rh), _ang = cv2.minAreaRect(pts)
        long_s, short_s = max(rw, rh), max(min(rw, rh), 1.0)
        fill = area / (long_s * short_s)
        if long_s >= min_len and long_s / short_s >= min_elong and fill <= max_fill:
            keep[lab == i] = 255
    return keep


def _orientation_deg(mask):
    """Dominant orientation of the kept pixels, 0 = horizontal, 90 = vertical."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return abs(np.degrees(np.arctan2(vy, vx)))


def analyze_crack(image):
    """image: PIL image or RGB numpy array. Returns (crack_type, edges, annotated)."""
    rgb = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
    H, W = rgb.shape[:2]

    s = min(1.0, WORK_MAX_SIDE / max(H, W))
    work = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    mask_w = crack_mask(work)
    mask = cv2.resize(mask_w, (W, H), interpolation=cv2.INTER_NEAREST) if s < 1 else mask_w

    annotated = rgb.copy()
    thick = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    annotated[thick > 0] = (255, 0, 0)

    ang = _orientation_deg(mask)
    if ang is None:
        crack_type = "Fine Crack"
    elif ang < 20:
        crack_type = "Horizontal Crack"
    elif ang > 70:
        crack_type = "Vertical Crack"
    else:
        crack_type = "Diagonal Crack"
    return crack_type, mask, annotated