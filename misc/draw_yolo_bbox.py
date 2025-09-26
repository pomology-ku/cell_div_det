#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO bbox 可視化ツール（AABB）
- ラベル形式: <cls> <xc> <yc> <w> <h> [conf]  (正規化, 0..1)
- 画像: 単一ファイル or ディレクトリを指定可能
- ラベル: 単一ファイル or ディレクトリを指定可能（画像と同名 .txt を探索）

使い方例:
  単一画像:
    python draw_yolo_bboxes.py -i sample.jpg -l sample.txt -o out
  ディレクトリ一括:
    python draw_yolo_bboxes.py -i imgs/ -l labels/ -o out

オプション:
  -i, --input         入力画像（ファイル or ディレクトリ）
  -l, --labels        ラベル（ファイル or ディレクトリ）
  -o, --out           出力ディレクトリ（未作成なら自動作成）
  -t, --thickness     線の太さ（px, 既定:2）
  -m, --min-area      描画する最小面積（px^2, 既定:0=制限なし）
  -k, --skip-missing  対応ラベルが無い画像をスキップ（指定なしならラベル無しでも画像だけ保存）
  -x, --exts          読む拡張子カンマ区切り（既定:"jpg,jpeg,png,bmp,tif,tiff,webp"）
  -c, --classnames    クラス名テキスト（1行1クラス, 任意）
"""

import argparse
from pathlib import Path
import cv2
import sys

IMG_EXTS_DEFAULT = ["jpg","jpeg","png","bmp","tif","tiff","webp"]

def parse_args():
    ap = argparse.ArgumentParser(description="Draw YOLO-format AABB bboxes on images.")
    ap.add_argument("-i", "--input", required=True, type=Path, help="Image file or directory")
    ap.add_argument("-l", "--labels", required=True, type=Path, help="Label file or directory")
    ap.add_argument("-o", "--out", required=True, type=Path, help="Output directory")
    ap.add_argument("-t", "--thickness", type=int, default=2, help="Line thickness (px)")
    ap.add_argument("-m", "--min-area", type=int, default=0, help="Minimum bbox area to draw (px^2)")
    ap.add_argument("-k", "--skip-missing", action="store_true", help="Skip images without label file")
    ap.add_argument("-x", "--exts", type=str, default=",".join(IMG_EXTS_DEFAULT),
                    help="Comma-separated image extensions (no dot)")
    ap.add_argument("-c", "--classnames", type=Path, default=None,
                    help="Optional classes.txt (one class name per line)")
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

def yolo_to_xyxy(xc, yc, w, h, W, H):
    """正規化YOLO -> ピクセルxyxy"""
    x = xc * W
    y = yc * H
    bw = w * W
    bh = h * H
    x1 = int(round(x - bw/2))
    y1 = int(round(y - bh/2))
    x2 = int(round(x + bw/2))
    y2 = int(round(y + bh/2))
    # 画像内にクリップ
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))
    return x1, y1, x2, y2

def read_labels_file(txt_path: Path):
    """1行: <cls> <xc> <yc> <w> <h> [conf] をパース。空/存在無しなら空配列。"""
    if not txt_path.exists():
        return []
    anns = []
    with txt_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] malformed line (len<{5}) {txt_path}:{ln}: {line}", file=sys.stderr)
                continue
            try:
                cls = int(float(parts[0]))  # 0,1,2...（int想定だが安全に）
                xc, yc, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else None
                anns.append((cls, xc, yc, w, h, conf))
            except Exception as e:
                print(f"[WARN] parse error {txt_path}:{ln}: {e} :: {line}", file=sys.stderr)
                continue
    return anns

def color_for_class(cls: int):
    """クラスごとの色（BGR）。OpenCVはBGR。"""
    # 簡易に適当に散らす
    palette = [
        (36,255,12), (0,204,255), (255,178,102), (204,0,255), (0,153,255),
        (102,255,255), (255,102,178), (178,255,102), (255,204,0), (102,178,255)
    ]
    return palette[cls % len(palette)]

def draw_one_image(img_path: Path, label_path: Path, out_dir: Path, thickness: int,
                   min_area: int, classnames):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] cannot read image: {img_path}", file=sys.stderr)
        return False
    H, W = img.shape[:2]
    anns = read_labels_file(label_path)

    for cls, xc, yc, w, h, conf in anns:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if min_area > 0 and area < min_area:
            continue
        color = color_for_class(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # テキスト: クラス名/ID と conf（もしあれば）
        if classnames and 0 <= cls < len(classnames):
            cls_txt = classnames[cls]
        else:
            cls_txt = str(cls)
        if conf is not None:
            text = f"{cls_txt} {conf:.2f}"
        else:
            text = f"{cls_txt}"

        # テキスト背景付きで見やすく
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        tx, ty = x1, max(0, y1 - th - 4)
        cv2.rectangle(img, (tx, ty), (tx + tw + 6, ty + th + baseline + 6), color, -1)
        cv2.putText(img, text, (tx + 3, ty + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        # 中心点（薄め）
        cx, cy = int(round(xc*W)), int(round(yc*H))
        cv2.circle(img, (cx, cy), max(1, thickness+1), color, -1)

    # 保存
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

    # 入力がファイルかディレクトリかで分岐
    if args.input.is_file():
        # 画像がファイル → ラベルもファイル指定が望ましい
        if args.labels.is_dir():
            # 同名.txt を探す
            label_path = args.labels / (args.input.stem + ".txt")
        else:
            label_path = args.labels
        if not label_path.exists():
            if args.skip_missing:
                print(f"[INFO] skip (label not found): {label_path}", file=sys.stderr)
                return 0
            else:
                print(f"[INFO] label not found, drawing none: {label_path}", file=sys.stderr)
        draw_one_image(args.input, label_path, args.out, args.thickness, args.min_area, classnames)
        return 0

    # ディレクトリ一括
    if not args.input.is_dir():
        print(f"[ERROR] input not found: {args.input}", file=sys.stderr)
        return 1

    # ラベル側がファイルの場合はエラー（ディレクトリにしてほしい）
    if args.labels.is_file():
        print(f"[ERROR] labels should be a directory when input is a directory.", file=sys.stderr)
        return 1

    img_paths = [p for p in args.input.rglob("*") if p.suffix.lower() in exts]
    img_paths.sort()
    if not img_paths:
        print(f"[WARN] no images found under: {args.input}", file=sys.stderr)

    n_ok = 0
    for img_path in img_paths:
        # 入力と同じ相対パス構造で出力（サブディレクトリも再現）
        rel = img_path.relative_to(args.input)
        out_dir = args.out / rel.parent

        # ラベルは labels/ 以下で同stem.txt を探す（同じ相対パスである必要はなく、ファイル名一致でOK）
        label_path = args.labels / (img_path.stem + ".txt")
        if not label_path.exists():
            if args.skip_missing:
                print(f"[INFO] skip (label not found): {label_path}", file=sys.stderr)
                continue
            else:
                print(f"[INFO] label not found, drawing none: {label_path}", file=sys.stderr)

        ok = draw_one_image(img_path, label_path, out_dir, args.thickness, args.min_area, classnames)
        if ok:
            n_ok += 1

    print(f"[DONE] wrote {n_ok} images to: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
