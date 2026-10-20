"""make_real_dataset.py
Turn your real photos into a train / val / test split, BY PHOTO.

Input (whole photos you collected):
    real_data/crack/*.jpg    photos with a crack. Frame the crack near the
                             MIDDLE of the photo (the crops are taken near the centre).
    real_data/clean/*.jpg    photos with NO crack. Include windows, frames,
                             pipes, shadows, stains, paint lines, hands.

Output:
    real_split/{train,val,test}/{crack,clean}/   whole photos (copies)
        -> use real_split/test with eval_real_photos.py  (never trained on)
    patches_real/{train,val}/{Positive,Negative}/  224x224 patches
        -> used by finetune_real.py

A photo goes to exactly one split, so patches of one photo never leak
between train and test. Run:
    python make_real_dataset.py
"""
import os, glob, shutil, random, argparse
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="real_data")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--per-crack", type=int, default=12, help="patches per crack photo")
ap.add_argument("--per-clean", type=int, default=20, help="patches per clean photo")
a = ap.parse_args()
random.seed(a.seed)

EXT = (".jpg", ".jpeg", ".png")
SPLIT = (0.60, 0.15, 0.25)          # train / val / test, by photo


def list_imgs(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(EXT))


def split_photos(files):
    files = files[:]
    random.shuffle(files)
    n = len(files)
    if n == 0:
        return [], [], []
    n_val = max(1, round(n * SPLIT[1])) if n >= 4 else 0
    n_test = max(1, round(n * SPLIT[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def crop_square(img, side, cx, cy):
    w, h = img.size
    side = int(max(32, min(side, w, h)))
    x0 = int(min(max(cx - side / 2, 0), w - side))
    y0 = int(min(max(cy - side / 2, 0), h - side))
    return img.crop((x0, y0, x0 + side, y0 + side)).resize((224, 224), Image.LANCZOS)


def patches_crack(img, n):
    w, h = img.size
    out = []
    for _ in range(n):
        side = random.uniform(0.45, 0.95) * min(w, h)
        cx = w / 2 + random.uniform(-0.12, 0.12) * w      # near the centre
        cy = h / 2 + random.uniform(-0.12, 0.12) * h
        out.append(crop_square(img, side, cx, cy))
    return out


def patches_clean(img, n):
    w, h = img.size
    out = []
    for _ in range(n):
        side = random.uniform(0.30, 1.0) * min(w, h)       # anywhere, mixed sizes
        cx = random.uniform(side / 2, max(side / 2, w - side / 2))
        cy = random.uniform(side / 2, max(side / 2, h - side / 2))
        out.append(crop_square(img, side, cx, cy))
    return out


for d in ("real_split", "patches_real"):
    if os.path.isdir(d):
        shutil.rmtree(d)

counts = {}
for label, per, maker, cls in (("crack", a.per_crack, patches_crack, "Positive"),
                               ("clean", a.per_clean, patches_clean, "Negative")):
    files = list_imgs(os.path.join(a.src, label))
    parts = dict(zip(("train", "val", "test"), split_photos(files)))
    for name, fl in parts.items():
        os.makedirs(os.path.join("real_split", name, label), exist_ok=True)
        counts[(label, name)] = len(fl)
        n_patch = 0
        for f in fl:
            shutil.copy(f, os.path.join("real_split", name, label, os.path.basename(f)))
            if name == "test":
                continue
            img = Image.open(f).convert("RGB")
            dst = os.path.join("patches_real", name, cls)
            os.makedirs(dst, exist_ok=True)
            stem = os.path.splitext(os.path.basename(f))[0]
            for k, p in enumerate(maker(img, per)):
                p.save(os.path.join(dst, f"{stem}_{k:02d}.jpg"), quality=92)
                n_patch += 1
        if name != "test":
            counts[(label, name, "patches")] = n_patch

print("Photos per split (whole photos, split by photo):")
for label in ("crack", "clean"):
    print(f"  {label:5s}  train {counts[(label,'train')]:3d}   val {counts[(label,'val')]:3d}   test {counts[(label,'test')]:3d}")
print("Patches written to patches_real/:")
for label in ("crack", "clean"):
    print(f"  {label:5s}  train {counts[(label,'train','patches')]:4d}   val {counts[(label,'val','patches')]:4d}")
low = [k for k, v in counts.items() if len(k) == 2 and v < 5]
if low:
    print("\nWARNING: fewer than 5 photos in", low, "- collect more; results will be very noisy.")
print("\nNext: python finetune_real.py     then:  python eval_real_photos.py --root real_split/test")