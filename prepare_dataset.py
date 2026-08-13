"""
prepare_dataset.py
Prepares SDNET2018 wall-crack subset for CSAF-Net.

Usage:
    python prepare_dataset.py --source /path/to/SDNET2018 --output dataset

Expects source folder structure (from SDNET2018 official download):
    SDNET2018/
        W/
            CW/   <- cracked wall images
            UW/   <- uncracked wall images

Produces:
    dataset/
        train/Positive, train/Negative
        val/Positive,   val/Negative
        test/Positive,  test/Negative
"""

import os
import shutil
import random
import argparse

random.seed(42)

PER_CLASS_LIMIT = 2000
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def collect_images(folder):
    valid_ext = (".jpg", ".jpeg", ".png")
    return [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]


def split_list(items, ratios):
    random.shuffle(items)
    n = len(items)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def copy_split(files, src_folder, dst_root, split_name, class_name):
    dst_folder = os.path.join(dst_root, split_name, class_name)
    os.makedirs(dst_folder, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to extracted SDNET2018 folder")
    parser.add_argument("--output", default="dataset", help="Output dataset folder")
    parser.add_argument("--limit", type=int, default=PER_CLASS_LIMIT, help="Max images per class")
    args = parser.parse_args()

    cracked_dir = os.path.join(args.source, "W", "CW")
    uncracked_dir = os.path.join(args.source, "W", "UW")

    if not os.path.isdir(cracked_dir) or not os.path.isdir(uncracked_dir):
        raise FileNotFoundError(
            f"Expected folders not found:\n  {cracked_dir}\n  {uncracked_dir}\n"
            "Check your SDNET2018 extraction path/structure."
        )

    print("Scanning source folders...")
    cracked_files = collect_images(cracked_dir)
    uncracked_files = collect_images(uncracked_dir)
    print(f"  Found {len(cracked_files)} cracked wall images")
    print(f"  Found {len(uncracked_files)} uncracked wall images")

    random.shuffle(cracked_files)
    random.shuffle(uncracked_files)
    cracked_files = cracked_files[:args.limit]
    uncracked_files = uncracked_files[:args.limit]
    print(f"\nUsing {len(cracked_files)} cracked / {len(uncracked_files)} uncracked (limit={args.limit})")

    cracked_splits = split_list(cracked_files, SPLIT_RATIOS)
    uncracked_splits = split_list(uncracked_files, SPLIT_RATIOS)

    if os.path.exists(args.output):
        print(f"\nRemoving existing '{args.output}' folder...")
        shutil.rmtree(args.output)

    print("\nCopying files...")
    for split_name in ["train", "val", "test"]:
        copy_split(cracked_splits[split_name], cracked_dir, args.output, split_name, "Positive")
        copy_split(uncracked_splits[split_name], uncracked_dir, args.output, split_name, "Negative")
        print(f"  {split_name}: {len(cracked_splits[split_name])} Positive, "
              f"{len(uncracked_splits[split_name])} Negative")

    print(f"\nDone. Dataset ready at ./{args.output}/")
    print("Structure:")
    print(f"  {args.output}/train/Positive, {args.output}/train/Negative")
    print(f"  {args.output}/val/Positive,   {args.output}/val/Negative")
    print(f"  {args.output}/test/Positive,  {args.output}/test/Negative")


if __name__ == "__main__":
    main()
