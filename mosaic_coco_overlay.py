#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple tiles into a single mosaic using JEOL MappingInformation XML
and overlay COCO (AABB) detections on top. (GT overlay optional)

- XML: expects <Kobetu ... ImagePath="X042 Y001.tif" OffsetX="0" OffsetY="0" ImageWidth/Height=...>
- COCO: produced by export_coco_from_yolo_obb.py (bbox = [x, y, w, h], AABB in pixel coords).
        images[].file_name may be written without extension depending on --no-strip-ext.
- GT (optional): YOLO-OBB label files under --gt-dir (recursive). Supported formats per line:
    (A) cls x1 y1 x2 y2 x3 y3 x4 y4   # 8点ポリゴン (正規化 0..1)
    (B) cls xc yc w h angle_rad        # 中心・幅高・角度 (正規化, ラジアン)

New options:
  --gt-dir PATH           : provide to overlay GT polygons
  --gt-color R,G,B        : color for GT (default 255,0,255)
  --gt-line INT           : line width for GT (default 3)
  --gt-alpha INT          : fill alpha for GT (0..255, default 0 no fill)
  --no-overlay            : モザイクのみ保存
"""

import argparse
import json
import math
import sys
import re
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

# ---------------------------- Helpers ----------------------------

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def choose_scale(canvas_w, canvas_h, scale_opt, max_w, max_h):
    """Decide final scale factor (<=1.0)."""
    if scale_opt is not None:
        return float(scale_opt)
    sw = (max_w / canvas_w) if max_w else 1.0
    sh = (max_h / canvas_h) if max_h else 1.0
    s = min(sw, sh, 1.0)
    if s <= 0:
        s = 1.0
    return s


def load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def hash_color(k: int):
    """Return a visually distinct RGB color from an integer key (category_id)."""
    phi = (1 + 5 ** 0.5) / 2
    hue = (k * phi) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def normalize_key(path_like: str) -> str:
    """Key for matching: lowercase stem (w/o extension), and normalize path seps."""
    s = str(path_like).replace("\\", "/").rsplit("/", 1)[-1]
    stem = s.rsplit(".", 1)[0]
    return stem.lower()


def parse_rgb(csv: str, default=(255, 0, 255)):
    try:
        a = [int(x.strip()) for x in csv.split(",")]
        if len(a) != 3:
            return default
        return tuple(max(0, min(255, v)) for v in a)
    except Exception:
        return default

# ------------------------- XML Parsing ---------------------------

def parse_mapping_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x_is_reverse_attr = root.attrib.get("XIsReverse")
    x_is_reverse = None
    if isinstance(x_is_reverse_attr, str):
        x_is_reverse = x_is_reverse_attr.strip().lower() in ("true", "1", "yes")

    tiles = []
    for el in root.findall(".//KobetuInformation/Kobetu"):
        image_path = el.attrib.get("ImagePath")
        w = _safe_int(el.attrib.get("ImageWidth", 0))
        h = _safe_int(el.attrib.get("ImageHeight", 0))
        offx = _safe_int(el.attrib.get("OffsetX", 0))
        offy = _safe_int(el.attrib.get("OffsetY", 0))
        idx_x = _safe_int(el.attrib.get("IndexX", 0))
        idx_y = _safe_int(el.attrib.get("IndexY", 0))
        kob_id = el.attrib.get("ID", "")
        cx = _safe_float(el.attrib.get("X"), None)
        cy = _safe_float(el.attrib.get("Y"), None)
        vw = _safe_float(el.attrib.get("ViewWidth"), None)
        vh = _safe_float(el.attrib.get("ViewHeight"), None)
        tiles.append({
            "image_path": image_path,
            "w": w, "h": h,
            "offx": offx, "offy": offy,
            "idx_x": idx_x, "idx_y": idx_y,
            "id": kob_id,
            "center_x": cx, "center_y": cy,
            "view_w": vw, "view_h": vh,
        })
    tiles.sort(key=lambda t: (t["idx_y"], t["idx_x"], t["image_path"]))
    meta = {"x_is_reverse": x_is_reverse}
    return tiles, meta

# ------------------------- COCO Parsing --------------------------

def parse_coco(coco_path: Path):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    img_by_id = {}
    imgs_by_key = defaultdict(list)
    for im in coco.get("images", []):
        img_by_id[im["id"]] = im
        key = normalize_key(im.get("file_name", ""))
        imgs_by_key[key].append(im)

    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)
    return img_by_id, imgs_by_key, anns_by_img, cat_name

# --------------------- YOLO-OBB GT utilities --------------------

def strip_prefix(name: str, pattern: str) -> str:
    """Remove leading prefix matched by regex pattern from a filename (no dirs, no ext)."""
    return re.sub(pattern, '', name, count=1, flags=re.IGNORECASE)

def build_gt_index(gt_root: Path, strip_regex: str):
    """
    Make a dict: stripped_stem(lower) -> [label_file_paths...]
    - Strips leading prefix matched by strip_regex from file *stem* (basename without ext)
    - Aggregates multiple files per stripped key
    """
    idx = defaultdict(list)
    for p in gt_root.rglob("*.txt"):
        base = p.name  # e.g., '332_X005 Y008.txt'
        stem = base.rsplit(".", 1)[0]
        stripped = strip_prefix(stem, strip_regex)
        key = normalize_key(stripped)  # lower
        idx[key].append(p)
    return idx

def parse_yolo_obb_line(line: str, tile_w: int, tile_h: int):
    """
    Returns polygon points in pixel coords: [(x1,y1),...,(x4,y4)]
    Supports:
      - cls x1 y1 x2 y2 x3 y3 x4 y4   (normalized)
      - cls xc yc w h angle_rad       (normalized, radians)
    """
    parts = line.strip().split()
    if not parts or parts[0].startswith("#"):
        return None
    if len(parts) not in (6, 9):  # 1+5 or 1+8
        return None
    try:
        cls = int(float(parts[0]))  # not used here
    except Exception:
        return None

    if len(parts) == 9:
        vals = list(map(float, parts[1:9]))
        x1,y1,x2,y2,x3,y3,x4,y4 = vals
        pts = [
            (x1 * tile_w, y1 * tile_h),
            (x2 * tile_w, y2 * tile_h),
            (x3 * tile_w, y3 * tile_h),
            (x4 * tile_w, y4 * tile_h),
        ]
        return pts

    # len==6 -> (xc,yc,w,h,angle)
    _, xc, yc, w, h, a = parts
    xc = float(xc) * tile_w
    yc = float(yc) * tile_h
    w  = float(w)  * tile_w
    h  = float(h)  * tile_h
    a  = float(a)  # radians
    # rectangle corners around center before rotation
    hw, hh = w / 2.0, h / 2.0
    rect = [(-hw,-hh), ( hw,-hh), ( hw, hh), (-hw, hh)]
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for (dx, dy) in rect:
        rx = dx * ca - dy * sa + xc
        ry = dx * sa + dy * ca + yc
        pts.append((rx, ry))
    return pts

# -------------------------- Main Logic ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge tiles from XML and overlay COCO bboxes (+ optional GT)")
    ap.add_argument("--xml", '-x', required=True, help="Path to MappingInformation XML")
    ap.add_argument("--coco", '-j', required=True, help="Path to COCO JSON (AABB)")
    ap.add_argument("--img-root", '-i', required=True, help="Directory containing tile images")
    ap.add_argument("--out", '-o', required=True, help="Output mosaic image (.png recommended)")

    # Sizing
    g = ap.add_argument_group("Size/scale")
    g.add_argument("--scale", '-s', type=float, default=None, help="Downscale factor (e.g., 0.5). If set, overrides max-*)")
    g.add_argument("--max-width", type=int, default=None, help="Max width for auto scaling (pixels)")
    g.add_argument("--max-height", type=int, default=None, help="Max height for auto scaling (pixels)")

    # Placement
    p = ap.add_argument_group("Placement")
    p.add_argument("--place-mode", choices=["offset", "stage", "stage+offset", "grid"], default="stage+offset",
                   help="Tile placement: use OffsetX/Y only, Stage X/Y with ViewWidth/Height, both, or grid")
    p.add_argument("--x-reverse", choices=["auto", "true", "false"], default="auto",
                   help="Flip stage X axis (mirror). 'auto' reads XML XIsReverse if present")

    # Drawing style
    s = ap.add_argument_group("Overlay style (detections)")
    s.add_argument("--line", type=int, default=3, help="BBox line thickness in output pixels")
    s.add_argument("--alpha", type=int, default=0, help="BBox fill alpha 0..255 (0=transparent)")
    s.add_argument("--show-label", choices=["none", "class_id", "class_name"], default="class_name", help="What label to draw on boxes")
    s.add_argument("--font-size", type=int, default=16, help="Label font size (after scaling)")
    s.add_argument("--tile-border", action="store_true", help="Draw tile borders for debugging")
    s.add_argument("--tile-label", action="store_true", help="Draw tile (IndexX,IndexY) label")

    # GT overlay
    gt = ap.add_argument_group("GT overlay (YOLO-OBB)")
    gt.add_argument("--gt-dir", type=str, default=None, help="Root directory containing YOLO-OBB label files (recursive).")
    gt.add_argument("--gt-color", type=str, default="255,0,255", help="GT RGB color as 'R,G,B' (default magenta)")
    gt.add_argument("--gt-line", type=int, default=3, help="GT line width (pixels)")
    gt.add_argument("--gt-alpha", type=int, default=0, help="GT fill alpha 0..255 (0=no fill)")
    gt.add_argument("--gt-strip-prefix", type=str, default=r"^[0-9]+_", help="Regex to strip from the beginning of GT filenames (stem). Default: '^[0-9]+_'")

    # Grid params (when --place-mode grid)
    p.add_argument("--grid-overlap-x", type=float, default=0.0,
                   help="Horizontal overlap ratio [0..0.9] used in grid mode (0=no overlap)")
    p.add_argument("--grid-overlap-y", type=float, default=0.0,
                   help="Vertical overlap ratio [0..0.9] used in grid mode (0=no overlap)")

    # Cropping
    c = ap.add_argument_group("Cropping")
    c.add_argument("--crop", choices=["none", "tiles"], default="none",
                   help="Auto-crop output. 'tiles' crops to union of pasted tiles.")
    c.add_argument("--crop-margin", type=int, default=0,
                   help="Extra margin (px) around crop box after scaling")

    # Overlay control
    oc = ap.add_argument_group("Overlay control")
    oc.add_argument("--no-overlay", action="store_true",
        help="Disable COCO/GT overlays and tile borders/labels; save plain mosaic only.")

    args = ap.parse_args()

    xml_path = Path(args.xml)
    coco_path = Path(args.coco)
    img_root = Path(args.img_root)
    out_path = Path(args.out)

    tiles, meta = parse_mapping_xml(xml_path)
    if not tiles:
        print("[ERROR] No <Kobetu> entries found in XML.", file=sys.stderr)
        sys.exit(1)

    img_by_id, imgs_by_key, anns_by_img, cat_name = parse_coco(coco_path)

    # Compute tile absolute positions (pre-shift)
    positions = []  # list of (tile, x, y)
    min_x = 1e18; min_y = 1e18
    max_x = -1e18; max_y = -1e18

    # Decide X reverse
    if args.x_reverse == "true":
        x_rev = True
    elif args.x_reverse == "false":
        x_rev = False
    else:
        x_rev = bool(meta.get("x_is_reverse")) if meta.get("x_is_reverse") is not None else False

    # Stage-to-pixel conversion
    def stage_to_px(t):
        cx = t.get("center_x")
        cy = t.get("center_y")
        vw = t.get("view_w")
        vh = t.get("view_h")
        if None in (cx, cy, vw, vh) or vw == 0 or vh == 0:
            return None
        sx = -cx if x_rev else cx
        px_per_u_x = t["w"] / vw
        px_per_u_y = t["h"] / vh
        tlx = (sx - vw / 2.0) * px_per_u_x
        tly = (cy - vh / 2.0) * px_per_u_y
        return tlx, tly

    # --- grid mode precompute ---
    min_idx_x = min(t["idx_x"] for t in tiles)
    min_idx_y = min(t["idx_y"] for t in tiles)
    tile_w_ref = sum(t["w"] for t in tiles) / max(1, len(tiles))
    tile_h_ref = sum(t["h"] for t in tiles) / max(1, len(tiles))
    stride_x = tile_w_ref * max(0.1, 1.0 - max(0.0, min(0.9, args.grid_overlap_x)))
    stride_y = tile_h_ref * max(0.1, 1.0 - max(0.0, min(0.9, args.grid_overlap_y)))

    for t in tiles:
        if args.place_mode == "grid":
            x = (t["idx_x"] - min_idx_x) * stride_x
            y = (t["idx_y"] - min_idx_y) * stride_y
        elif args.place_mode == "offset":
            x = t["offx"]; y = t["offy"]
        elif args.place_mode == "stage":
            st = stage_to_px(t)
            if st is None:
                x = t["offx"]; y = t["offy"]
            else:
                x, y = st
        else:  # stage+offset
            st = stage_to_px(t)
            if st is None:
                x = t["offx"]; y = t["offy"]
            else:
                sx, sy = st
                x = sx + t["offx"]
                y = sy + t["offy"]

        positions.append((t, x, y))
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + t["w"])
        max_y = max(max_y, y + t["h"])

    # Shift so min is 0,0
    shift_x = -min_x if min_x < 0 else 0
    shift_y = -min_y if min_y < 0 else 0

    canvas_w = int(math.ceil(max_x + shift_x))
    canvas_h = int(math.ceil(max_y + shift_y))
    if canvas_w <= 0 or canvas_h <= 0:
        print("[ERROR] Invalid canvas size computed.", file=sys.stderr)
        sys.exit(1)

    scale = choose_scale(canvas_w, canvas_h, args.scale, args.max_width, args.max_height)

    # Create canvas
    first_img_path = img_root / tiles[0]["image_path"]
    try:
        with Image.open(first_img_path) as im0:
            mode = "RGB" if im0.mode in ("RGB", "RGBA") else "L"
    except Exception:
        mode = "L"

    out_full = Image.new(mode, (canvas_w, canvas_h), 0 if mode == "L" else (0, 0, 0))

    # Paste tiles
    for t, x, y in positions:
        tile_path = img_root / t["image_path"]
        if not tile_path.exists():
            print(f"[WARN] Tile not found: {tile_path}")
            continue
        try:
            with Image.open(tile_path) as imt:
                imt = imt.convert(mode)
                out_full.paste(imt, (int(x + shift_x), int(y + shift_y)))
        except Exception as e:
            print(f"[WARN] Failed to paste {tile_path}: {e}")

    # Prepare scaled image
    if scale != 1.0:
        new_w = max(1, int(round(canvas_w * scale)))
        new_h = max(1, int(round(canvas_h * scale)))
        out_scaled = out_full.resize((new_w, new_h), Image.BILINEAR)
    else:
        out_scaled = out_full

    if out_scaled.mode != "RGBA":
        out_scaled = out_scaled.convert("RGBA")

    # -------------------- Overlays (optional) --------------------
    if not args.no_overlay:
        draw = ImageDraw.Draw(out_scaled)

        # Debug tile borders/labels
        if args.tile_border or args.tile_label:
            font = load_font(max(10, int(args.font_size * (scale if scale != 0 else 1))))
            for t, x, y in positions:
                x0 = int(round((x + shift_x) * scale))
                y0 = int(round((y + shift_y) * scale))
                x1 = int(round((x + shift_x + t["w"]) * scale))
                y1 = int(round((y + shift_y + t["h"]) * scale))
                if args.tile_border:
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 128), width=max(1, int(args.line)))
                if args.tile_label:
                    txt = f"({t['idx_x']},{t['idx_y']})"
                    draw.text((x0 + 4, y0 + 4), txt, fill=(255, 255, 0, 200), font=font)

        # Build mapping stem -> (offset, size)
    tile_info = {}
    for t, x, y in positions:
        key = normalize_key(t["image_path"])
        tile_info[key] = {
            "shifted_x": x + shift_x,
            "shifted_y": y + shift_y,
            "w": t["w"],
            "h": t["h"],
            "path": str(img_root / t["image_path"]),
        }

    if not args.no_overlay:
        # Draw COCO detections (AABB)
        font = load_font(max(10, int(args.font_size * (scale if scale != 0 else 1))))
        matched = 0
        skipped = 0
        for img_id, im in list(img_by_id.items()):
            key = normalize_key(im.get("file_name", ""))
            tinfo = tile_info.get(key)
            if tinfo is None:
                print(f"[WARN] No tile match for COCO image: file_name='{im.get('file_name')}' (key='{key}')")
                skipped += 1
                continue

            matched += 1
            ann_list = anns_by_img.get(img_id, [])
            if not ann_list:
                continue

            ox = tinfo["shifted_x"]
            oy = tinfo["shifted_y"]

            for ann in ann_list:
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                gx0 = (ox + x) * scale
                gy0 = (oy + y) * scale
                gx1 = (ox + x + w) * scale
                gy1 = (oy + y + h) * scale

                cat_id = ann.get("category_id", 0)
                color = hash_color(cat_id)
                lw = max(1, int(args.line))
                if args.alpha > 0:
                    draw.rectangle([gx0, gy0, gx1, gy1], fill=(color[0], color[1], color[2], int(args.alpha)))
                draw.rectangle([gx0, gy0, gx1, gy1], outline=(color[0], color[1], color[2], 255), width=lw)

                if args.show_label != "none":
                    if args.show_label == "class_name":
                        label = cat_name.get(cat_id, str(cat_id))
                    else:
                        label = str(cat_id)
                    try:
                        l, t, r, b = draw.textbbox((0, 0), label, font=font)
                        tw, th = (r - l), (b - t)
                    except Exception:
                        tw, th = draw.textlength(label, font=font), font.size
                    bx0, by0 = gx0, gy0 - th - 2
                    bx1, by1 = gx0 + tw + 4, gy0
                    draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 160))
                    draw.text((gx0 + 2, gy0 - th - 1), label, fill=(255, 255, 255, 230), font=font)

        print(f"[INFO] Matched COCO images to tiles: {matched}, skipped (no tile): {skipped}")

        # --------------------- GT overlay (optional) ---------------------
        if args.gt_dir:
            gt_root = Path(args.gt_dir)
            if not gt_root.exists():
                print(f"[WARN] --gt-dir not found: {gt_root}")
            else:
                gt_index = build_gt_index(gt_root, args.gt_strip_prefix)
                gt_color = parse_rgb(args.gt_color, default=(255, 0, 255))
                gt_alpha = max(0, min(255, int(args.gt_alpha)))
                gt_line = max(1, int(args.gt_line))

                gt_found_tiles = 0
                gt_missing_tiles = 0
                total_gt_files_used = 0

                for stem, tinfo in tile_info.items():
                    lbl_paths = gt_index.get(stem)
                    if not lbl_paths:
                        gt_missing_tiles += 1
                        continue

                    gt_found_tiles += 1
                    ox = tinfo["shifted_x"]; oy = tinfo["shifted_y"]
                    tw = tinfo["w"];         th = tinfo["h"]

                    # 同じ stem に複数 GT があれば全部重ねる
                    for lbl_path in lbl_paths:
                        try:
                            with open(lbl_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                        except Exception as e:
                            print(f"[WARN] Failed to read GT file {lbl_path}: {e}")
                            continue

                        used_any = False
                        for ln in lines:
                            poly = parse_yolo_obb_line(ln, tw, th)
                            if not poly:
                                continue
                            poly_scaled = [((ox + px) * scale, (oy + py) * scale) for (px, py) in poly]
                            if gt_alpha > 0:
                                draw.polygon(poly_scaled, fill=(gt_color[0], gt_color[1], gt_color[2], gt_alpha))
                            draw.polygon(poly_scaled, outline=(gt_color[0], gt_color[1], gt_color[2], 255))
                            if gt_line > 1:
                                for i in range(4):
                                    x0, y0 = poly_scaled[i]
                                    x1, y1 = poly_scaled[(i + 1) % 4]
                                    draw.line(
                                        [x0, y0, x1, y1],
                                        fill=(gt_color[0], gt_color[1], gt_color[2], 255),
                                        width=gt_line,
                                    )
                            used_any = True

                        if used_any:
                            total_gt_files_used += 1

                print(
                    f"[INFO] GT overlay: tiles matched {gt_found_tiles}, "
                    f"tiles without GT {gt_missing_tiles}, GT files used {total_gt_files_used}"
                )

    # ---------------- Auto-crop (optional) ----------------
    if args.crop == "tiles":
        # タイル矩形の union をスケール後座標系で計算
        min_x_s = 1e18; min_y_s = 1e18; max_x_s = -1e18; max_y_s = -1e18
        for t, x, y in positions:
            x0 = int(round((x + shift_x) * scale))
            y0 = int(round((y + shift_y) * scale))
            x1 = int(round((x + shift_x + t["w"]) * scale))
            y1 = int(round((y + shift_y + t["h"]) * scale))
            min_x_s = min(min_x_s, x0)
            min_y_s = min(min_y_s, y0)
            max_x_s = max(max_x_s, x1)
            max_y_s = max(max_y_s, y1)

        # マージン付与＆クリップ
        mg = max(0, int(args.crop_margin))
        x0c = max(0, min_x_s - mg)
        y0c = max(0, min_y_s - mg)
        x1c = min(out_scaled.width,  max_x_s + mg)
        y1c = min(out_scaled.height, max_y_s + mg)

        if x1c > x0c and y1c > y0c:
            out_cropped = out_scaled.crop((x0c, y0c, x1c, y1c))
            out_scaled = out_cropped
            print(f"[INFO] Cropped to tiles: ({x0c},{y0c})-({x1c},{y1c}) size={x1c-x0c}x{y1c-y0c}")

    # Save
    out_scaled.save(out_path)
    status = "without overlays" if args.no_overlay else "with overlays"
    print(f"[OK] Wrote mosaic {status}: {out_path}  (scale={scale:.4f}, canvas={canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()

