#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBB(4点) -> HBB のYOLOラベル変換
- 入力: 各行 = cls x1 y1 x2 y2 x3 y3 x4 y4  （すべて正規化[0,1]）
- 出力: 各行 = cls cx cy w h               （すべて正規化[0,1]）
- 5要素(= OBBの xc yc w h angle) が来た場合は、そのままAABBに直して出力します。
"""

import argparse, sys
from pathlib import Path

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def poly4_to_aabb_norm(coords):
    # coords = [x1, y1, x2, y2, x3, y3, x4, y4]
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = (xmax - xmin)
    h = (ymax - ymin)
    # 0-1にクリップ
    return [clamp01(cx), clamp01(cy), clamp01(w), clamp01(h)]

def obb5_to_aabb_norm(xc, yc, w, h, angle):
    # すでに正規化済みの xc,yc,w,h を受け取り、AABBへ（回転は無視）
    # 回転を無視するとAABBはそのまま w,h（※角度による外接矩形ではなく、回転前の軸平行箱に相当）
    # 一般には「外接AABB」を取りたければ4点復元→poly4_to_aabb_norm を使う必要がある。
    return [clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)]

def convert_file(in_path: Path, out_path: Path, min_wh=1e-6):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                cls = int(float(parts[0]))
            except Exception:
                # クラスIDが整数でない場合の保険
                cls = int(round(float(parts[0])))
            nums = list(map(float, parts[1:]))

            if len(nums) == 8:
                cx, cy, w, h = poly4_to_aabb_norm(nums)
            elif len(nums) == 5:
                # xc yc w h angle
                cx, cy, w, h = obb5_to_aabb_norm(*nums)
            else:
                # 想定外の行はスキップ
                continue

            # 面積が極小ならスキップ（オプション）
            if w < min_wh or h < min_wh:
                continue

            lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with out_path.open("w", encoding="utf-8") as g:
        g.write("\n".join(lines_out) + ("\n" if lines_out else ""))

def main():
    ap = argparse.ArgumentParser(description="Convert OBB(4点or5要素) labels to YOLO HBB")
    ap.add_argument("-i", "--in_dir", required=True, help="入力ラベルディレクトリ（例: dat/v250830/labels/train）")
    ap.add_argument("-o", "--out_dir", required=True, help="出力ラベルディレクトリ（例: out_hbb/labels/train）")
    ap.add_argument("--min_wh", type=float, default=1e-6, help="最小幅/高さ（正規化）。これ未満は除外")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    txts = sorted(in_dir.rglob("*.txt"))
    if not txts:
        print(f"No .txt found under {in_dir}", file=sys.stderr)
        sys.exit(1)

    for p in txts:
        rel = p.relative_to(in_dir)
        out_p = out_dir / rel
        convert_file(p, out_p, min_wh=args.min_wh)

    print(f"Done. Wrote HBB labels to: {out_dir}")

if __name__ == "__main__":
    main()
