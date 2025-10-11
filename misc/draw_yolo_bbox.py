#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO OBB 可視化ツール（回転矩形 or 4点ポリゴン）
- ラベル形式A: <cls> <xc> <yc> <w> <h> <angle_rad> [conf]   # 正規化(0..1), 角度ラジアン（+は反時計回り）
- ラベル形式B: <cls> x1 y1 x2 y2 x3 y3 x4 y4 [conf]          # 正規化(0..1)想定（画素でも可: 自動で0..1範囲外はクリップ）

使い方例:
  単一画像:
    python draw_yolo_obb.py -i sample.jpg -l sample.txt -o out
  ディレクトリ一括:
    python draw_yolo_obb.py -i imgs/ -l labels/ -o out

オプション:
  -i, --input         入力画像（ファイル or ディレクトリ）
  -l, --labels        ラベル（ファイル or ディレクトリ）
  -o, --out           出力ディレクトリ（未作成なら自動作成）
  -t, --thickness     線の太さ（px, 既定:2）
  -m, --min-area      描画する最小面積（px^2, 既定:0=制限なし）
  -k, --skip-missing  対応ラベルが無い画像をスキップ（指定なしならラベル無しでも画像だけ保存）
  -x, --exts          読む拡張子カンマ区切り（既定:"jpg,jpeg,png,bmp,tif,tiff,webp"）
  -c, --classnames    クラス名テキスト（1行1クラス, 任意）
  --draw-center       OBB/ポリゴンの中心点を描く
  --format            'auto'|'obb'|'poly' 解析固定（既定:auto）
"""

import argparse
from pathlib import Path
import cv2
import sys
import math
import numpy as np

IMG_EXTS_DEFAULT = ["jpg","jpeg","png","bmp","tif","tiff","webp"]

def parse_args():
    ap = argparse.ArgumentParser(description="Draw YOLO-format OBB (rotated) bboxes/polygons on images.")
    ap.add_argument("-i", "--input", required=True, type=Path, help="Image file or directory")
    ap.add_argument("-l", "--labels", required=True, type=Path, help="Label file or directory")
    ap.add_argument("-o", "--out", required=True, type=Path, help="Output directory")
    ap.add_argument("-t", "--thickness", type=int, default=2, help="Line thickness (px)")
    ap.add_argument("-m", "--min-area", type=int, default=0, help="Minimum polygon area to draw (px^2)")
    ap.add_argument("-k", "--skip-missing", action="store_true", help="Skip images without label file")
    ap.add_argument("-x", "--exts", type=str, default=",".join(IMG_EXTS_DEFAULT),
                    help="Comma-separated image extensions (no dot)")
    ap.add_argument("-c", "--classnames", type=Path, default=None,
                    help="Optional classes.txt (one class name per line)")
    ap.add_argument("--draw-center", action="store_true", help="Draw center point")
    ap.add_argument("--format", choices=["auto","obb","poly"], default="auto",
                    help="Force parse format: 'auto' (default), 'obb' (xc,yc,w,h,angle), or 'poly' (x1..y4)")
    return ap.parse_args()

def load_classnames(path: Path|None):
    if not path:
        return None
    if not path.exists():
        print(f"[WARN] classnames file not found: {path}", file=sys.stderr)
        return None
    with path.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip() != ""]
    return names

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def poly_area(points_xy: np.ndarray) -> float:
    """points_xy: (N,2) in pixel coords"""
    x = points_xy[:,0]
    y = points_xy[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def rotate_rect_to_poly(cx, cy, bw, bh, angle_rad):
    """
    回転矩形(中心(cx,cy), 幅bw, 高さbh, 角度angle_rad[CCW]) -> 4点ポリゴン (x,y) np.ndarray shape=(4,2)
    四隅の順序: 左上→右上→右下→左下 になるように CCW順（一般的には可視化に大差なし）
    """
    # 矩形のローカル座標（中心原点）：(±bw/2, ±bh/2)
    dx = bw / 2.0
    dy = bh / 2.0
    rect_local = np.array([[-dx, -dy],
                           [ +dx, -dy],
                           [ +dx, +dy],
                           [ -dx, +dy]], dtype=np.float64)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)
    poly = rect_local @ R.T
    poly[:,0] += cx
    poly[:,1] += cy
    return poly

def clamp_poly_to_image(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    """各点を画像内に丸めて収める（厳密なクリッピングではなく点のクランプ）。"""
    poly[:,0] = np.clip(poly[:,0], 0, W-1)
    poly[:,1] = np.clip(poly[:,1], 0, H-1)
    return poly

def parse_line_auto(parts, W, H):
    """
    parts: list[str] (1行をsplit)
    返り値: ("obb" or "poly", cls, poly_pts(4,2), center_xy(2), conf or None)
    形式は自動推定:
      - len>=6 かつ 2..5番目が浮動小数のとき → OBB
      - len>=9 のとき → 4点ポリゴン
    """
    if len(parts) >= 9:
        # 4点ポリゴン: <cls> x1 y1 x2 y2 x3 y3 x4 y4 [conf]
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:9]))
        conf = float(parts[9]) if len(parts) >= 10 else None
        # 正規化→px
        pts = np.array(coords, dtype=np.float64).reshape(4,2)
        # 0..1想定だが一部ツールはpxのこともあるので、0..2の範囲なら正規化としてW,Hを掛ける（極端な値はそのまま扱う）
        if (pts >= 0).all() and (pts <= 1.0).all():
            pts[:,0] *= W
            pts[:,1] *= H
        pts = clamp_poly_to_image(pts, W, H)
        cx = float(np.mean(pts[:,0]))
        cy = float(np.mean(pts[:,1]))
        return "poly", cls, pts, (cx, cy), conf

    if len(parts) >= 6:
        # OBB: <cls> <xc> <yc> <w> <h> <angle_rad> [conf]
        cls = int(float(parts[0]))
        xc, yc, w, h, ang = map(float, parts[1:6])
        conf = float(parts[6]) if len(parts) >= 7 else None
        cx = xc * W
        cy = yc * H
        bw = w * W
        bh = h * H
        pts = rotate_rect_to_poly(cx, cy, bw, bh, ang)
        pts = clamp_poly_to_image(pts, W, H)
        return "obb", cls, pts, (cx, cy), conf

    raise ValueError("Malformed line (need OBB>=6 fields or POLY>=9 fields)")

def parse_line_obb(parts, W, H):
    cls = int(float(parts[0]))
    if len(parts) < 6:
        raise ValueError("OBB requires >=6 fields")
    xc, yc, w, h, ang = map(float, parts[1:6])
    conf = float(parts[6]) if len(parts) >= 7 else None
    cx = xc * W
    cy = yc * H
    bw = w * W
    bh = h * H
    pts = rotate_rect_to_poly(cx, cy, bw, bh, ang)
    pts = clamp_poly_to_image(pts, W, H)
    return "obb", cls, pts, (cx, cy), conf

def parse_line_poly(parts, W, H):
    cls = int(float(parts[0]))
    if len(parts) < 9:
        raise ValueError("POLY requires >=9 fields")
    coords = list(map(float, parts[1:9]))
    conf = float(parts[9]) if len(parts) >= 10 else None
    pts = np.array(coords, dtype=np.float64).reshape(4,2)
    if (pts >= 0).all() and (pts <= 1.0).all():
        pts[:,0] *= W
        pts[:,1] *= H
    pts = clamp_poly_to_image(pts, W, H)
    cx = float(np.mean(pts[:,0]))
    cy = float(np.mean(pts[:,1]))
    return "poly", cls, pts, (cx, cy), conf

def read_labels_file(txt_path: Path, W: int, H: int, force_fmt: str = "auto"):
    """テキストをパースし、[(fmt, cls, pts(4,2), (cx,cy), conf), ...] を返す。存在無し/空は []."""
    if not txt_path.exists():
        return []
    out = []
    with txt_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                if force_fmt == "auto":
                    rec = parse_line_auto(parts, W, H)
                elif force_fmt == "obb":
                    rec = parse_line_obb(parts, W, H)
                else:
                    rec = parse_line_poly(parts, W, H)
                out.append(rec)
            except Exception as e:
                print(f"[WARN] parse error {txt_path}:{ln}: {e} :: {s}", file=sys.stderr)
                continue
    return out

def color_for_class(cls: int):
    """クラスごとの色（BGR）。OpenCVはBGR。"""
    palette = [
        (36,255,12), (0,204,255), (255,178,102), (204,0,255), (0,153,255),
        (102,255,255), (255,102,178), (178,255,102), (255,204,0), (102,178,255)
    ]
    return palette[cls % len(palette)]

def draw_poly(img, pts: np.ndarray, color, thickness: int):
    """pts shape=(4,2)"""
    pts_i = pts.astype(np.int32)
    cv2.polylines(img, [pts_i], isClosed=True, color=color, thickness=thickness)

def draw_one_image(img_path: Path, label_path: Path, out_dir: Path, thickness: int,
                   min_area: int, classnames, draw_center: bool, force_fmt: str):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] cannot read image: {img_path}", file=sys.stderr)
        return False
    H, W = img.shape[:2]

    anns = read_labels_file(label_path, W, H, force_fmt)
    for fmt, cls, pts, (cx, cy), conf in anns:
        area = poly_area(pts)
        if min_area > 0 and area < min_area:
            continue

        color = color_for_class(cls)
        draw_poly(img, pts, color, thickness)

        # テキスト（クラス名/ID + conf）
        if classnames and 0 <= cls < len(classnames):
            cls_txt = classnames[cls]
        else:
            cls_txt = str(cls)
        text = f"{cls_txt}" if conf is None else f"{cls_txt} {conf:.2f}"

        # ラベル位置は上辺に近い頂点の上に
        top_idx = int(np.argmin(pts[:,1]))
        tx, ty = int(pts[top_idx,0]), int(pts[top_idx,1])
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        box_pt1 = (max(0, tx), max(0, ty - th - baseline - 6))
        box_pt2 = (min(W-1, tx + tw + 6), min(H-1, ty))
        cv2.rectangle(img, box_pt1, box_pt2, color, -1)
        cv2.putText(img, text, (box_pt1[0]+3, box_pt2[1]-baseline-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        if draw_center:
            cv2.circle(img, (int(round(cx)), int(round(cy))), max(1, thickness+1), color, -1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        print(f"[WARN] failed to write: {out_path}", file=sys.stderr)
        return False
    return True

def main():
    args = parse_args()
    classnames = load_classnames(args.classnames)
    exts = ["." + e.lower().strip() for e in args.exts.split(",") if e.strip()]

    # 入力: ファイル or ディレクトリ
    if args.input.is_file():
        # ラベル側: ディレクトリなら同stem.txt、ファイルならそのまま
        if args.labels.is_dir():
            label_path = args.labels / (args.input.stem + ".txt")
        else:
            label_path = args.labels
        if not label_path.exists():
            if args.skip_missing:
                print(f"[INFO] skip (label not found): {label_path}", file=sys.stderr)
                return 0
            else:
                print(f"[INFO] label not found, drawing none: {label_path}", file=sys.stderr)
        draw_one_image(args.input, label_path, args.out, args.thickness,
                       args.min_area, classnames, args.draw_center, args.format)
        return 0

    # ディレクトリ一括
    if not args.input.is_dir():
        print(f"[ERROR] input not found: {args.input}", file=sys.stderr)
        return 1

    if args.labels.is_file():
        print(f"[ERROR] labels should be a directory when input is a directory.", file=sys.stderr)
        return 1

    img_paths = [p for p in args.input.rglob("*") if p.suffix.lower() in exts]
    img_paths.sort()
    if not img_paths:
        print(f"[WARN] no images found under: {args.input}", file=sys.stderr)

    n_ok = 0
    for img_path in img_paths:
        rel = img_path.relative_to(args.input)
        out_dir = args.out / rel.parent

        label_path = args.labels / (img_path.stem + ".txt")
        if not label_path.exists():
            if args.skip_missing:
                print(f"[INFO] skip (label not found): {label_path}", file=sys.stderr)
                continue
            else:
                print(f"[INFO] label not found, drawing none: {label_path}", file=sys.stderr)

        ok = draw_one_image(img_path, label_path, out_dir, args.thickness,
                            args.min_area, classnames, args.draw_center, args.format)
        if ok:
            n_ok += 1

    print(f"[DONE] wrote {n_ok} images to: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

