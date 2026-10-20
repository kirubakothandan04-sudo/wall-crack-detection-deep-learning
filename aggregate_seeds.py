import os, json, glob
import numpy as np

root = os.path.expanduser("~/csaf_seeds")
variants = ["full_csaf", "ungated_cross_attn", "scalar_gate"]
metrics = ["accuracy", "f1", "precision", "recall"]

for v in variants:
    paths = [f"logs/{v}_log.json"] + sorted(glob.glob(os.path.join(root, "logs", "seed*", f"{v}_log.json")))
    runs = [json.load(open(p))["test"] for p in paths if os.path.exists(p)]
    n = len(runs)
    print(f"\n{v}  (n = {n} runs)")
    for m in metrics:
        vals = np.array([r[m] for r in runs]) * 100
        sd = vals.std(ddof=1) if n > 1 else 0.0
        print(f"  {m:10s} {vals.mean():6.2f} +/- {sd:5.2f}   runs: " + ", ".join(f"{x:.2f}" for x in vals))
