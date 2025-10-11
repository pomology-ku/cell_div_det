#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-OBB poly ラベル (.txt) の座標を [0,1] 範囲にクリップして再保存するツール

- フォルダ配下のすべての .txt を再帰的に探索
- 各行の <cls> x1 y1 x2 y2 x3 y3 x4 y4 [conf] を対象
- 負値や1超値はクリップ（0～1）
- 面積が極小なポリゴンはスキップ
- 修正があった場合は .bak にバックアップを残す

使い方:
    python clip_yolo_obb_labels.py -l train/labels
"""

import argparse
from pathlib import Path
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(description="Clip YOLO-OBB (poly) label coordinates to [0,1].")
    ap.add_argument("-l", "--labels", required=True, type=Path, help="labels ディレクトリ（再帰的探索）")
    ap.add_argument("--area-min", type=float, default=1e-6, help="この面積未満のpolyは除外")
    ap.add_argument("--backup", action="store_true", help=".bak バックアップを残す")
    return ap.parse_args()

def poly_area(P):
    return 0.5 * abs(np.dot(P[:,0], np.roll(P[:,1], -1)) - np.dot(P[:,1], np.roll(P[:,0], -1)))

def sanitize_poly_line(parts, area_min=1e-6):
    """cls x1 y1 ... y4 [conf] をクリップして返す。除外対象は None。"""
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:9]))
    conf = parts[9] if len(parts) >= 10 else None

    P = np.array(coords, dtype=float).reshape(4,2)
    # 0..1 クリップ
    P[:,0] = np.clip(P[:,0], 0.0, 1.0)
    P[:,1] = np.clip(P[:,1], 0.0, 1.0)
    # 面積チェック
    if poly_area(P) < area_min:
        return None
    # 反時計回りに統一（任意）
    cross = np.cross(P[1]-P[0], P[2]-P[1])
    if cross < 0:
        P = P[::-1]
    flat = [str(cls)] + [f"{v:.6f}" for v in P.reshape(-1).tolist()]
    if conf is not None:
        flat.append(conf)
    return " ".join(flat)

def clip_dir(labels_dir: Path, area_min: float=1e-6, backup: bool=False):
    n_total = n_fixed = n_dropped = 0
    for txt in labels_dir.rglob("*.txt"):
        lines = txt.read_text(encoding="utf-8").splitlines()
        out_lines = []
        changed = False
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 9:  # obbや他形式は無視
                out_lines.append(line)
                continue
            n_total += 1
            before = parts[1:9]
            fixed = sanitize_poly_line(parts, area_min)
            if fixed is None:
                n_dropped += 1
                changed = True
                continue
            after = fixed.split()[1:9]
            if any(abs(float(a)-float(b)) > 1e-9 for a,b in zip(before, after)):
                changed = True
                n_fixed += 1
            out_lines.append(fixed)
        if changed:
            if backup:
                txt.rename(txt.with_suffix(".txt.bak"))
            txt.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[DONE] total={n_total}, fixed={n_fixed}, dropped={n_dropped}")

def main():
    args = parse_args()
    if not args.labels.exists():
        raise SystemExit(f"[ERROR] not found: {args.labels}")
    clip_dir(args.labels, args.area_min, args.backup)

if __name__ == "__main__":
    main()
