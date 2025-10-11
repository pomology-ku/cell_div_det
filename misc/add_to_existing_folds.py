#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append additional data into *existing* foldK/train/{images,labels}.
- Positive: CVAT YOLO-OBB export (images/train + labels/train)
- Negative: images-only dirs with confirmed empty scenes (labels created as empty)

After appending, updates each foldK/train.txt.

Usage examples:
  # add positives (CVAT export)
  python add_to_existing_folds.py -r /dat/v251003 -a /dat/v251003_additional -p _add -y

  # add negatives (images only)
  python add_to_existing_folds.py -r /dat/v251003 -n /dat/neg_tiles -p _add -y

  # mix: multiple sources
  python add_to_existing_folds.py -r /dat/add \
      -a /dat/add_pos1 -a /dat/add_pos2 \
      -n /dat/neg1 -n /dat/neg2 -p _add -y
"""

import argparse, os, sys, shutil
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def link_or_copy(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def collect_images_recursive(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def derive_label_for_image(img_path: Path, img_root: Path, lab_root: Path) -> Path:
    rel = img_path.relative_to(img_root).with_suffix(".txt")
    return lab_root / rel

def strip_leading_train(rel: Path) -> Path:
    parts = list(rel.parts)
    if parts and parts[0].lower() == "train":
        parts = parts[1:]
    return Path(*parts) if parts else Path(rel.name)

def write_list_txt(base_dir: Path, file_path: Path, image_paths_abs: list[Path]):
    rels = [str(p.relative_to(base_dir)).replace(os.sep, "/") for p in image_paths_abs]
    file_path.write_text("\n".join(rels) + "\n", encoding="utf-8")

def add_positive_source(add_root: Path, fold_dirs, prefix: str, tag: str, use_symlink: bool, skip: bool, grand_total: list[int]):
    img_src = add_root / "images" / "train"
    lab_src = add_root / "labels" / "train"
    if not img_src.exists() or not lab_src.exists():
        print(f"[ERR] {add_root} must contain images/train and labels/train", file=sys.stderr)
        sys.exit(1)
    imgs = collect_images_recursive(img_src)
    print(f"[INFO] POS '{add_root}' ({len(imgs)} imgs) -> subdir '{prefix}/{tag}'")
    for fold in fold_dirs:
        tr_img = fold/"train"/"images"/prefix/tag
        tr_lab = fold/"train"/"labels"/prefix/tag
        added = 0
        for ip in imgs:
            rel = strip_leading_train(ip.relative_to(img_src))
            dst_img = tr_img / rel
            lp = derive_label_for_image(ip, img_src, lab_src)
            rel_lab = strip_leading_train(lp.relative_to(lab_src))
            dst_lab = tr_lab / rel_lab

            if skip and dst_img.exists() and dst_lab.exists():
                continue

            link_or_copy(ip, dst_img, use_symlink)
            if lp.exists() and lp.stat().st_size > 0:
                link_or_copy(lp, dst_lab, use_symlink)
            else:
                dst_lab.parent.mkdir(parents=True, exist_ok=True)
                if not dst_lab.exists():
                    dst_lab.write_text("", encoding="utf-8")  # empty = negative/no objects
            added += 1
            grand_total[0] += 1

        all_train_imgs = collect_images_recursive(fold/"train"/"images")
        write_list_txt(fold, fold/"train.txt", all_train_imgs)
        print(f"  {fold.name}: +{added} (train.txt -> {len(all_train_imgs)})")

def add_negative_source(neg_root: Path, fold_dirs, prefix: str, tag: str, use_symlink: bool, skip: bool, grand_total: list[int]):
    if not neg_root.exists():
        print(f"[ERR] Negative root not found: {neg_root}", file=sys.stderr)
        sys.exit(1)
    imgs = collect_images_recursive(neg_root)
    print(f"[INFO] NEG '{neg_root}' ({len(imgs)} imgs) -> subdir '{prefix}/{tag}' (labels will be empty)")
    for fold in fold_dirs:
        tr_img = fold/"train"/"images"/prefix/tag
        tr_lab = fold/"train"/"labels"/prefix/tag
        added = 0
        for ip in imgs:
            # keep structure under neg_root; if the top dir is "train", drop it
            rel = strip_leading_train(ip.relative_to(neg_root))
            dst_img = tr_img / rel
            dst_lab = tr_lab / rel.with_suffix(".txt")
            if skip and dst_img.exists() and dst_lab.exists():
                continue
            link_or_copy(ip, dst_img, use_symlink)
            dst_lab.parent.mkdir(parents=True, exist_ok=True)
            if not dst_lab.exists():
                dst_lab.write_text("", encoding="utf-8")  # empty label = negative
            added += 1
            grand_total[0] += 1

        all_train_imgs = collect_images_recursive(fold/"train"/"images")
        write_list_txt(fold, fold/"train.txt", all_train_imgs)
        print(f"  {fold.name}: +{added} NEG (train.txt -> {len(all_train_imgs)})")

def main():
    ap = argparse.ArgumentParser(description="Append positive/negative data to existing folds' train sets.")
    ap.add_argument("-r", "--root", required=True, help="Root that contains foldK/ (K=0..).")
    ap.add_argument("-a", "--add", action="append",
                    help="Positive (CVAT YOLO-OBB) dataset root (has images/train and labels/train). Repeatable.")
    ap.add_argument("-n", "--neg", action="append",
                    help="Negative images-only root (no labels). Repeatable; recurses.")
    ap.add_argument("-p", "--prefix", default="_add",
                    help="Subdir under train/{images,labels} to place additions (default: _add).")
    ap.add_argument("-t", "--tag", default=None,
                    help="Tag name under <prefix>/ for positives. Default: each positive dir name.")
    ap.add_argument("-g", "--neg_tag", default=None,
                    help="Tag name under <prefix>/ for negatives. Default: each neg dir name.")
    ap.add_argument("-y", "--symlink", action="store_true",
                    help="Symlink instead of copy (fallback to copy on failure).")
    ap.add_argument("-k", "--skip", action="store_true",
                    help="Skip if destination image+label already exist.")
    ap.add_argument("-f", "--folds", default=None,
                    help="Comma-separated folds to update (e.g., 0,2,4). Default: all fold* under root.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.folds:
        fold_ids = [s.strip() for s in args.folds.split(",") if s.strip() != ""]
        fold_dirs = [root / f"fold{fid}" for fid in fold_ids]
    else:
        fold_dirs = sorted([p for p in root.glob("fold*") if p.is_dir()])
    if not fold_dirs:
        print("[ERR] No fold* directories found under root.", file=sys.stderr)
        sys.exit(1)

    grand_total = [0]

    # positives
    if args.add:
        for a in [Path(x).resolve() for x in args.add]:
            tag = args.tag if args.tag else a.name
            add_positive_source(a, fold_dirs, args.prefix, tag, args.symlink, args.skip, grand_total)

    # negatives
    if args.neg:
        for nroot in [Path(x).resolve() for x in args.neg]:
            ntag = args.neg_tag if args.neg_tag else nroot.name
            add_negative_source(nroot, fold_dirs, args.prefix, ntag, args.symlink, args.skip, grand_total)

    print(f"[DONE] Total files added across folds: {grand_total[0]}")

if __name__ == "__main__":
    main()

