#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quickly verify YOLO dataset readability by Ultralytics.

- Checks if data.yaml can be loaded
- Lists number of images under train/val/test
- Verifies each image has a corresponding label (except for empty negatives)
- Prints several example paths for sanity check

Usage:
  python check_yolo_dataset.py -d /path/to/fold0/data.yaml
"""

import argparse
from pathlib import Path
import yaml
import os

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def collect_images(img_dir: Path):
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

def check_pair(imgs, img_root: Path, labels_root: Path):
    """imgs を img_root からの相対にして、labels_root に .txt を張る"""
    missing_labels = []
    empty_labels = 0
    for ip in imgs:
        # 常に img_root を基準に相対パスを計算
        rel = ip.relative_to(img_root)
        lp = labels_root / rel.with_suffix(".txt")
        if not lp.exists():
            missing_labels.append(ip)
        else:
            if lp.stat().st_size == 0:
                empty_labels += 1
    return missing_labels, empty_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Path to data.yaml (e.g., fold0/data.yaml)")
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[ERR] {data_yaml} not found.")
        return

    print(f"[INFO] Checking dataset: {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    for key in ["train", "val", "test"]:
        if key not in y:
            print(f"[WARN] '{key}' not found in yaml.")
            continue
        img_dir = Path(y[key]) / "images" if not (Path(y[key]).name == "images") else Path(y[key])
        lab_dir = img_dir.parent / "labels"
        imgs = collect_images(img_dir)
        print(f"\n[{key.upper()}]  images={len(imgs)}  dir={img_dir}")
        if not imgs:
            continue
        missing, empty = check_pair(imgs, img_dir, lab_dir)
        print(f"  missing_labels={len(missing)}  empty_labels={empty}")
        if imgs:
            print(f"  example: {imgs[0].name}")
        if missing[:5]:
            print(f"  first missing example: {missing[0]}")

    print("\n✅ If counts look correct and missing_labels=0, YOLO can read this dataset normally.")

if __name__ == "__main__":
    main()
