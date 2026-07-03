#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a CVAT-importable COCO JSON + filtered images from inference_obb.py output.

Reads one or two COCO JSONs (two models → concatenated detections per image).
Only images that have at least one detection are copied.

Default output (COCO 1.0 for CVAT):
    out/
      images/                ← source images that have ≥1 detection
          X001 Y006.tif
      annotations.json       ← COCO JSON (segmentation = OBB polygon, pixel coords)

CVAT import:
  1. Create task and upload images/ contents
  2. Task → Actions → Upload annotations → COCO 1.0 (instances) → annotations.json
     ※ images[].file_name must match CVAT task frame names exactly.
        By default file_name is written with extension (e.g. "X001 Y006.tif").
        Use --no-strip-ext to keep extension if your task was created with extension,
        or omit it (default) for extension-less frame names.

Optional YOLO OBB output (--yolo):
    out/
      labels/                ← cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
      classes.txt / obj.names / obj.data / data.yaml / train.txt

Note:
  - segmentation field (OBB polygon, pixel coords) must be present in input COCO JSON.
    inference_obb.py writes it automatically.
  - If segmentation is absent, falls back to AABB bbox as axis-aligned OBB.
  - Detections from both JSONs are concatenated per image; overlapping or identical
    detections are not de-duplicated.
  - --conf filters by the "score" field written by inference_obb.py.
"""

import argparse
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_coco_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(text.lstrip())
        print(f"[WARN] {json_path.name}: extra data after JSON; using first object only.")
        return obj


def normalize_stem(s: str) -> str:
    """Lowercase stem (strip extension and leading path)."""
    s = str(s).replace("\\", "/").rsplit("/", 1)[-1]
    return s.rsplit(".", 1)[0].lower()


def seg_to_8pt_px(seg) -> list[float] | None:
    """Extract pixel-coord OBB polygon [[x1,y1,...,x4,y4]] → flat list of 8 floats."""
    if not isinstance(seg, list) or not seg or not isinstance(seg[0], (list, tuple)):
        return None
    flat = seg[0]
    if len(flat) < 8:
        return None
    return [float(v) for v in flat[:8]]


def bbox_to_8pt_px(bbox) -> list[float]:
    """AABB [x,y,w,h] → axis-aligned 4-point polygon in pixel coords."""
    x, y, w, h = map(float, bbox[:4])
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def poly8_to_aabb(poly8: list[float]) -> list[float]:
    """Pixel-coord OBB polygon → AABB [x, y, w, h]."""
    xs = poly8[0::2]
    ys = poly8[1::2]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def polygon_area(poly8: list[float]) -> float:
    """Shoelace area of 4-point polygon."""
    pts = [(poly8[i * 2], poly8[i * 2 + 1]) for i in range(4)]
    n = len(pts)
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


@lru_cache(maxsize=None)
def _image_files_by_stem(img_dir: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    """Build one reusable image index for each source directory/extension set."""
    return {
        p.stem.lower(): p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    }


def find_image_file(stem_lower: str, img_dir: Path,
                    exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")) -> Path | None:
    return _image_files_by_stem(img_dir, tuple(exts)).get(stem_lower)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Export CVAT-importable COCO JSON + images from inference_obb.py COCO JSON(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-j", "--json", action="append", required=True, dest="jsons",
                    metavar="COCO_JSON",
                    help="COCO JSON produced by inference_obb.py (--out-json). "
                         "Specify once or twice for two models.")
    ap.add_argument("-i", "--img-dir", required=True,
                    help="Directory containing the original tile images.")
    ap.add_argument("-o", "--out", required=True,
                    help="Output directory.")
    ap.add_argument("-c", "--conf", type=float, default=0.0,
                    help="Minimum detection score to include (default: 0.0 = all).")
    ap.add_argument("-f", "--include", default=None, metavar="NAME_LIST",
                    help="Text file listing image names (stems) to include, one per line. "
                         "Blank lines and '#' lines are ignored. Extension is stripped for "
                         "matching. If omitted, all images are included.")
    ap.add_argument("--no-strip-ext", action="store_true",
                    help="Keep extension in images[].file_name in the output COCO JSON "
                         "(e.g. 'X001 Y006.tif'). Default: strip extension ('X001 Y006'). "
                         "Match this to how your CVAT task was created.")
    ap.add_argument("--out-json", default="annotations.json",
                    help="Filename for the output COCO JSON (default: annotations.json).")
    ap.add_argument("--yolo", action="store_true",
                    help="Also write YOLO OBB label files (labels/, classes.txt, data.yaml, "
                         "obj.data, train.txt) in addition to the COCO JSON output.")
    ap.add_argument("--label-subdir", default="labels",
                    help="Label subdirectory name for --yolo output (default: labels). "
                         "Use 'obj_train_data' for CVAT YOLO 1.1 layout.")
    args = ap.parse_args()

    if len(args.jsons) > 2:
        print("[ERROR] --json can be specified at most twice.", file=sys.stderr)
        sys.exit(1)

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    if args.yolo:
        lbl_out = out_dir / args.label_subdir
        lbl_out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # 0.  Build include-set (optional name filter)
    # ---------------------------------------------------------------------- #

    include_stems: set[str] | None = None
    if args.include:
        include_path = Path(args.include)
        if not include_path.exists():
            print(f"[ERROR] --include file not found: {include_path}", file=sys.stderr)
            sys.exit(1)
        with open(include_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
        include_stems = {
            normalize_stem(ln.strip())
            for ln in raw_lines
            if ln.strip() and not ln.strip().startswith("#")
        }
        print(f"[INFO] --include: {len(include_stems)} image stems loaded from {include_path.name}")

    # ---------------------------------------------------------------------- #
    # 1.  Load all COCO JSONs and accumulate detections per image stem
    # ---------------------------------------------------------------------- #

    class_name_order: list[str] = []
    # stem → {"img_w", "img_h", "src_filename", "dets": [{"cls_name", "poly8_px", "score"}]}
    per_image: dict[str, dict] = {}

    for json_idx, json_path_str in enumerate(args.jsons):
        json_path = Path(json_path_str)
        coco = load_coco_json(json_path)
        label = chr(ord("A") + json_idx)

        cat_by_id: dict[int, str] = {}
        for c in coco.get("categories", []):
            cid  = int(c["id"])
            name = str(c.get("name", cid))
            cat_by_id[cid] = name
            if name not in class_name_order:
                class_name_order.append(name)

        img_meta: dict[int, dict] = {int(im["id"]): im for im in coco.get("images", [])}

        ann_total = len(coco.get("annotations", []))
        ann_kept  = 0
        warned_no_seg = False

        for ann in coco.get("annotations", []):
            score = float(ann.get("score", 1.0))
            if score < args.conf:
                continue

            img_id = int(ann["image_id"])
            im = img_meta.get(img_id)
            if im is None:
                continue

            img_w = float(im.get("width",  0))
            img_h = float(im.get("height", 0))
            if img_w <= 0 or img_h <= 0:
                continue

            cls_name = cat_by_id.get(int(ann["category_id"]), str(ann["category_id"]))

            seg = ann.get("segmentation")
            poly8_px = seg_to_8pt_px(seg) if seg else None
            if poly8_px is None:
                bbox = ann.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                poly8_px = bbox_to_8pt_px(bbox)
                if not warned_no_seg:
                    print(f"[INFO] {json_path.name}: 'segmentation' absent; "
                          "using AABB bbox as axis-aligned OBB.")
                    warned_no_seg = True

            stem = normalize_stem(im.get("file_name", ""))
            if include_stems is not None and stem not in include_stems:
                continue

            if stem not in per_image:
                per_image[stem] = {
                    "img_w": img_w,
                    "img_h": img_h,
                    "src_filename": str(im.get("file_name", "")),
                    "dets": [],
                }
            per_image[stem]["dets"].append({
                "cls_name": cls_name,
                "poly8_px": poly8_px,
                "score": score,
            })
            ann_kept += 1

        print(f"[INFO] JSON {label} ({json_path.name}): "
              f"{ann_kept}/{ann_total} annotations kept (conf>={args.conf})")

    if not class_name_order:
        print("[ERROR] No categories found in any JSON.", file=sys.stderr)
        sys.exit(1)

    cls_to_idx: dict[str, int] = {name: i for i, name in enumerate(class_name_order)}

    # ---------------------------------------------------------------------- #
    # 2.  Copy images + build outputs
    # ---------------------------------------------------------------------- #

    # COCO output structures
    out_images: list[dict]      = []
    out_annotations: list[dict] = []
    out_categories: list[dict]  = [
        {"id": i + 1, "name": name, "supercategory": ""}
        for i, name in enumerate(class_name_order)
    ]
    coco_img_id = 0
    coco_ann_id = 0

    # YOLO output
    train_lines: list[str] = []

    copied         = 0
    skipped_no_img = 0

    for stem, info in sorted(per_image.items()):
        if not info["dets"]:
            continue

        src_img = find_image_file(stem, img_dir)
        if src_img is None:
            print(f"[WARN] Image not found for stem '{stem}' in {img_dir}")
            skipped_no_img += 1
            continue

        # Copy image
        shutil.copy2(src_img, img_out / src_img.name)
        copied += 1

        # file_name for COCO JSON: with or without extension
        coco_file_name = src_img.name if args.no_strip_ext else src_img.stem

        coco_img_id += 1
        out_images.append({
            "id":        coco_img_id,
            "file_name": coco_file_name,
            "width":     int(info["img_w"]),
            "height":    int(info["img_h"]),
        })

        for det in info["dets"]:
            poly8_px = det["poly8_px"]
            coco_ann_id += 1
            out_annotations.append({
                "id":          coco_ann_id,
                "image_id":    coco_img_id,
                "category_id": cls_to_idx[det["cls_name"]] + 1,
                "segmentation": [poly8_px],
                "bbox":        [round(v, 2) for v in poly8_to_aabb(poly8_px)],
                "area":        round(polygon_area(poly8_px), 2),
                "iscrowd":     0,
                "score":       round(det["score"], 6),
            })

        # YOLO OBB labels (optional)
        if args.yolo:
            lbl_path = lbl_out / (src_img.stem + ".txt")
            img_w, img_h = info["img_w"], info["img_h"]
            with open(lbl_path, "w", encoding="utf-8") as f:
                for det in info["dets"]:
                    idx = cls_to_idx[det["cls_name"]]
                    p   = det["poly8_px"]
                    coords = " ".join([
                        f"{p[0]/img_w:.6f}", f"{p[1]/img_h:.6f}",
                        f"{p[2]/img_w:.6f}", f"{p[3]/img_h:.6f}",
                        f"{p[4]/img_w:.6f}", f"{p[5]/img_h:.6f}",
                        f"{p[6]/img_w:.6f}", f"{p[7]/img_h:.6f}",
                    ])
                    f.write(f"{idx} {coords}\n")
            train_lines.append(f"images/{src_img.name}")

    # ---------------------------------------------------------------------- #
    # 3.  Write COCO JSON
    # ---------------------------------------------------------------------- #

    coco_out = {
        "images":      out_images,
        "annotations": out_annotations,
        "categories":  out_categories,
    }
    json_out_path = out_dir / args.out_json
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False)
    print(f"[OK] Wrote COCO JSON: {json_out_path}  "
          f"({len(out_images)} images, {len(out_annotations)} annotations)")

    # ---------------------------------------------------------------------- #
    # 4.  Write YOLO metadata (optional)
    # ---------------------------------------------------------------------- #

    if args.yolo:
        n_cls = len(class_name_order)
        with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
            for name in class_name_order:
                f.write(name + "\n")
        shutil.copy2(out_dir / "classes.txt", out_dir / "obj.names")
        with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
            for line in train_lines:
                f.write(line + "\n")
        with open(out_dir / "obj.data", "w", encoding="utf-8") as f:
            f.write(f"classes = {n_cls}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n")
        with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
            f.write(f"path:  {out_dir.resolve()}\ntrain: images\nnc: {n_cls}\n"
                    f"names: {class_name_order}\ntask:  obb\n")
        print(f"[OK] Wrote YOLO OBB labels → {lbl_out}")

    # ---------------------------------------------------------------------- #
    # 5.  Summary
    # ---------------------------------------------------------------------- #

    n_cls = len(class_name_order)
    print(f"\n[OK] Exported {copied} images → {out_dir / 'images'}")
    print(f"     Classes ({n_cls}): {', '.join(class_name_order)}")
    if skipped_no_img:
        print(f"[WARN] {skipped_no_img} image(s) not found in {img_dir}")

    ext_note = "with extension" if args.no_strip_ext else "without extension (default)"
    print(f"\nCVAT import (COCO 1.0):")
    print(f"  1. Create task and upload images/ contents")
    print(f"  2. Task → Actions → Upload annotations → COCO 1.0 (instances) → {args.out_json}")
    print(f"  ※ file_name in JSON is written {ext_note}.")
    print(f"     If frame names in CVAT don't match, re-run with/without --no-strip-ext.")


if __name__ == "__main__":
    main()
