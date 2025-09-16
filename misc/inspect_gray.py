#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像の次元とビット深度を確認し、3chが同一なら1chに無劣化変換するツール
- 使い方（確認のみ）: python inspect_gray.py /path/to/images --glob "*.tif"
- 使い方（1chへ変換）: python inspect_gray.py /path/to/images --out /path/to/out --glob "*.tif"
"""
import argparse, sys
from pathlib import Path
import cv2
import numpy as np

def load_image(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read: {p}")
    return img

def channels_equal(img: np.ndarray) -> bool:
    if img.ndim == 2:  # already single-channel
        return True
    if img.ndim == 3 and img.shape[2] == 1:
        return True
    if img.ndim == 3 and img.shape[2] == 3:
        c0, c1, c2 = img[:,:,0], img[:,:,1], img[:,:,2]
        return np.array_equal(c0, c1) and np.array_equal(c0, c2)
    return False

def to_single_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:,:,0]
    if img.ndim == 3 and img.shape[2] == 3:
        # 3chが同一ならそのまま1chへ。異なる場合はOpenCVのグレースケール変換。
        if channels_equal(img):
            return img[:,:,0]
        # BGR→Gray（cv2はBGR並び）
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unexpected image shape: {img.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="画像ディレクトリ or 単一ファイル")
    ap.add_argument("--glob", default="*.tif", help="パターン（例: *.tif, *.png）")
    ap.add_argument("--out", type=Path, default=None, help="1chに変換して保存する出力ディレクトリ（未指定なら保存しない）")
    args = ap.parse_args()

    paths = []
    if args.root.is_file():
        paths = [args.root]
    else:
        paths = sorted(args.root.rglob(args.glob))

    if not paths:
        print("No files matched.")
        return

    n_total = n_1ch = n_3ch_same = n_3ch_diff = 0

    for p in paths:
        try:
            img = load_image(p)
        except Exception as e:
            print(f"[NG] {p}: {e}")
            continue

        n_total += 1
        shape = img.shape
        dtype = img.dtype
        info = f"{p.name}: shape={shape}, dtype={dtype}"

        if img.ndim == 2:
            n_1ch += 1
            print("[1ch] " + info)
        elif img.ndim == 3 and img.shape[2] == 1:
            n_1ch += 1
            print("[1ch(3D)] " + info)
        elif img.ndim == 3 and img.shape[2] == 3:
            if channels_equal(img):
                n_3ch_same += 1
                print("[3ch=identical] " + info)
            else:
                n_3ch_diff += 1
                print("[3ch=different] " + info)
        else:
            print("[WARN] Unexpected dims " + info)

        # 1ch保存オプション
        if args.out:
            out_dir = args.out
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (p.stem + ".tif")
            gray = to_single_channel(img)
            # 無劣化保存（ビット深度は元dtypeを保持）
            ok = cv2.imwrite(str(out_path), gray)
            if not ok:
                print(f"[SAVE-ERR] {out_path}")

    print("\n=== Summary ===")
    print(f"Total: {n_total}")
    print(f"Single-channel: {n_1ch}")
    print(f"3ch (identical channels): {n_3ch_same}")
    print(f"3ch (different channels): {n_3ch_diff}")

if __name__ == "__main__":
    main()
