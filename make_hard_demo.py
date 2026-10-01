import os, random, argparse
from PIL import Image
from torchvision.datasets import ImageFolder

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=7)
p.add_argument("--w", type=int, default=2000)
p.add_argument("--h", type=int, default=1200)
p.add_argument("--crack_size", type=int, default=180, help="smaller = harder")
p.add_argument("--out", type=str, default="figures/hard_wide_demo.png")
a = p.parse_args()

random.seed(a.seed)
ds = ImageFolder(os.path.join("dataset", "test"))
pos = [s for s, l in ds.samples if l == ds.classes.index("Positive")]
neg = [s for s, l in ds.samples if l == ds.classes.index("Negative")]

tile = 224
canvas = Image.new("RGB", (a.w, a.h))
for y in range(0, a.h, tile):
    for x in range(0, a.w, tile):
        canvas.paste(Image.open(random.choice(neg)).convert("RGB").resize((tile, tile)), (x, y))

cs = a.crack_size
crack = Image.open(random.choice(pos)).convert("RGB").resize((cs, cs))
px = random.randint(50, a.w - cs - 50)   # not aligned to the 224 grid
py = random.randint(50, a.h - cs - 50)
canvas.paste(crack, (px, py))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
canvas.save(a.out)
print(f"Saved {a.out}")
print(f"Crack is hidden at x={px}-{px+cs}, y={py}-{py+cs} (only you know this)")