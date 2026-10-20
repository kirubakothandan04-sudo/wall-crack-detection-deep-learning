import os, random, argparse
import numpy as np
import torch
from train import train_model

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
p.add_argument("--variants", nargs="+",
               default=["full_csaf", "ungated_cross_attn", "scalar_gate"])
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--out_root", default=os.path.expanduser("~/csaf_seeds"))
a = p.parse_args()

for seed in a.seeds:
    for v in a.variants:
        log_dir = os.path.join(a.out_root, "logs", f"seed{seed}")
        ckpt_dir = os.path.join(a.out_root, "checkpoints", f"seed{seed}")
        if os.path.exists(os.path.join(log_dir, f"{v}_log.json")):
            print(f"Skipping seed {seed} {v}: already done")
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\n##### seed {seed} | variant {v} #####")
        train_model(variant=v, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                    checkpoint_dir=ckpt_dir, log_dir=log_dir)
