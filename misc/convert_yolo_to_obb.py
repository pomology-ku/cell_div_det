#!/usr/bin/env python3
import os, argparse
from pathlib import Path
from PIL import Image
import yaml

def load_classes(yolo_root: Path):
    txt = yolo_root / "classes.txt"
    yaml_file = yolo_root / "data.yaml"

    if txt.exists():
        with open(txt, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return {i: n for i, n in enumerate(names)}

    elif yaml_file.exists():
        with open(yaml_file, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        # YOLOv5 data.yaml の "names" は list or dict
        if isinstance(y.get("names"), dict):
            names = [y["names"][i] for i in range(len(y["names"]))]
        else:
            names = y.get("names", [])
        return {i: str(n) for i, n in enumerate(names)}

    else:
        # fallback: class0, class1, ...
        print("[WARN] No classes.txt or data.yaml found; using dummy class names")
        return {}

def polygon_area(pts):  # shoelace
    s = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        s += x1 * y2 - x2 * y1
    return 0.5 * s

def to_clockwise(pts):
    # DOTAは時計回り（CW）を推奨。shoelace符号が正ならCCWなので反転
    return pts[::-1] if polygon_area(pts) > 0 else pts

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def denorm_points(xs, ys, w, h):
    pts = [(xs[i] * w, ys[i] * h) for i in range(4)]
    # 画像境界にクリップ（indexエラーや外れ値対策）
    pts = [(clamp(x, 0, w - 1e-3), clamp(y, 0, h - 1e-3)) for x, y in pts]
    return to_clockwise(pts)

def yolo_obb_to_dota_line(cls_name, pts, diff=0):
    flat = " ".join([f"{x:.1f} {y:.1f}" for (x, y) in pts])
    return f"{flat} {cls_name} {diff}"

def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # Linuxならハードリンクで高速
    except Exception:
        dst.write_bytes(Path(src).read_bytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", required=True, help="CVAT YOLO OBB root (images/, labels/, classes.txt)")
    ap.add_argument("--out_root", required=True, help="Output root for DOTA-like dataset")
    ap.add_argument("--split", default="train", help="train/val under out_root")
    ap.add_argument("--suffixes", default=".jpg,.png,.jpeg,.tif,.tiff", help="comma-separated image suffixes")
    args = ap.parse_args()

    yolo_root = Path(args.yolo_root)
    yimg, ylab = yolo_root / "images/train", yolo_root / "labels/train"
    classes = load_classes(yolo_root)

    out_img = Path(args.out_root) / args.split / "images"
    out_lab = Path(args.out_root) / args.split / "labelTxt"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    # labels優先でベース名集合を作る（画像拡張子が混在しても拾える）
    label_paths = sorted(ylab.glob("*.txt"))
    base_set = {p.stem for p in label_paths}

    # images 側からも拾う（負例画像に対応）
    suffs = tuple(s.strip() for s in args.suffixes.split(",") if s.strip())
    img_paths = {p.stem: p for s in suffs for p in yimg.glob(f"*{s}")}

    count = 0
    for stem, ip in img_paths.items():
        lp = ylab / f"{stem}.txt"
        # 画像サイズ
        with Image.open(ip) as im:
            w, h = im.size

        hardlink_or_copy(ip, out_img / ip.name)

        dota_lines = []
        if lp.exists() and lp.stat().st_size > 0:
            with open(lp, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    # class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized, YOLO OBB)
                    try:
                        cls_id = int(parts[0])
                        xs = [float(parts[i]) for i in [1, 3, 5, 7]]
                        ys = [float(parts[i]) for i in [2, 4, 6, 8]]
                    except Exception:
                        continue
                    pts = denorm_points(xs, ys, w, h)
                    cls_name = classes.get(cls_id, f"class{cls_id}")
                    dota_lines.append(yolo_obb_to_dota_line(cls_name, pts, diff=0))
        # DOTAは負例でも空txtを用意
        (out_lab / f"{stem}.txt").write_text("\n".join(dota_lines) + ("\n" if dota_lines else ""), encoding="utf-8")
        count += 1

    print(f"[OK] Converted {count} images to DOTA under: {args.out_root}/{args.split}")

if __name__ == "__main__":
    main()
