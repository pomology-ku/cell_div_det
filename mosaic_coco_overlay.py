#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple tiles into a single mosaic using JEOL MappingInformation XML
and overlay COCO (AABB) detections on top.

- XML: expects <Kobetu ... ImagePath="X042 Y001.tif" OffsetX="0" OffsetY="0" ImageWidth/Height=...>
- COCO: produced by export_coco_from_yolo_obb.py (bbox = [x, y, w, h], AABB in pixel coords). 
        images[].file_name may be written without extension depending on --no-strip-ext.

Features
--------
* Robust matching between COCO images[] and XML tiles by basename (with/without extension).
* Handles negative offsets; auto-shifts to positive canvas coordinates.
* Downscaling options: fixed --scale OR fit into --max-width/--max-height while preserving aspect.
* BBox styling: color per class (deterministic), alpha fill, line thickness, optional labels.
* Optional tile border overlay and per-tile index label for debugging.

Usage examples
--------------
  python mosaic_coco_overlay.py \
      --xml MappingInformation.xml \
      --coco coco.json \
      --img-root /path/to/tiles \
      --out mosaic.png \
      --max-width 6000 --max-height 6000 \
      --line 3 --alpha 64 --show-label class_name

  # Fixed downscale (e.g., 0.25x) regardless of size
  python mosaic_coco_overlay.py --xml map.xml --coco coco.json \
      --img-root . --out mosaic_25.png --scale 0.25

Notes
-----
* Paste order: sorted by (IndexY, IndexX) if available, else by ImagePath.
* Overlaps are pasted in that order (later tiles overwrite earlier pixels).
* Colors are hashed from category_id and stable across runs.
"""

import argparse
import json
import math
import sys
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
    # Try to load a common TrueType font; fallback to PIL default
    try:
        # DejaVuSans is usually bundled with PIL
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def hash_color(k: int):
    """Return a visually distinct RGB color from an integer key (category_id)."""
    # Use golden ratio to distribute hues
    phi = (1 + 5 ** 0.5) / 2
    hue = (k * phi) % 1.0
    # Convert HSV->RGB (simple)
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def normalize_key(path_like: str) -> str:
    """Key for matching: lowercase stem (w/o extension), and normalize path seps."""
    s = str(path_like).replace("\\", "/").rsplit("/", 1)[-1]
    stem = s.rsplit(".", 1)[0]
    return stem.lower()


# ------------------------- XML Parsing ---------------------------

def parse_mapping_xml(xml_path: Path):
    """Parse MappingInformation XML and return (tiles, meta).

    Each tile dict has keys: image_path, w, h, offx, offy, idx_x, idx_y, id,
    center_x, center_y, view_w, view_h
    meta contains: x_is_reverse (bool or None)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Root may have XIsReverse="True"
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
        # Stage-based fields
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
    # sort for paste order
    tiles.sort(key=lambda t: (t["idx_y"], t["idx_x"], t["image_path"]))
    meta = {"x_is_reverse": x_is_reverse}
    return tiles, meta


# ------------------------- COCO Parsing --------------------------

def parse_coco(coco_path: Path):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    # images: list of {id, file_name, width, height}
    img_by_id = {}
    imgs_by_key = defaultdict(list)
    for im in coco.get("images", []):
        img_by_id[im["id"]] = im
        key = normalize_key(im.get("file_name", ""))
        imgs_by_key[key].append(im)

    # categories: id->name
    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}

    # annotations grouped by image_id
    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    return img_by_id, imgs_by_key, anns_by_img, cat_name


# -------------------------- Main Logic ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge tiles from XML and overlay COCO bboxes")
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
    p.add_argument("--place-mode", choices=["offset", "stage", "stage+offset"], default="stage+offset",
                   help="Tile placement: use OffsetX/Y only, Stage X/Y with ViewWidth/Height, or both")
    p.add_argument("--x-reverse", choices=["auto", "true", "false"], default="auto",
                   help="Flip stage X axis (mirror). 'auto' reads XML XIsReverse if present")

    # Drawing style
    s = ap.add_argument_group("Overlay style")
    s.add_argument("--line", type=int, default=3, help="BBox line thickness in output pixels")
    s.add_argument("--alpha", type=int, default=0, help="BBox fill alpha 0..255 (0=transparent)")
    s.add_argument("--show-label", choices=["none", "class_id", "class_name"], default="class_name", help="What label to draw on boxes")
    s.add_argument("--font-size", type=int, default=16, help="Label font size (after scaling)")
    s.add_argument("--tile-border", action="store_true", help="Draw tile borders for debugging")
    s.add_argument("--tile-label", action="store_true", help="Draw tile (IndexX,IndexY) label")

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

    # If stage-based placement, compute pixel-per-unit using view sizes
    def stage_to_px(t):
        # Center coordinates in stage units, convert to pixel top-left using view size
        cx = t.get("center_x")
        cy = t.get("center_y")
        vw = t.get("view_w")
        vh = t.get("view_h")
        if None in (cx, cy, vw, vh) or vw == 0 or vh == 0:
            return None
        sx = -cx if x_rev else cx
        # pixels per stage-unit for this tile (usually constant across tiles)
        px_per_u_x = t["w"] / vw
        px_per_u_y = t["h"] / vh
        # top-left in pixel coords (relative, will be shifted later)
        tlx = (sx - vw / 2.0) * px_per_u_x
        tly = (cy - vh / 2.0) * px_per_u_y
        return tlx, tly

    for t in tiles:
        if args.place_mode == "offset":
            x = t["offx"]
            y = t["offy"]
        elif args.place_mode == "stage":
            st = stage_to_px(t)
            if st is None:
                # Fallback to offsets if stage fields missing
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

    # Create canvas (8-bit grayscale or RGB)
    # We don't know the tile mode; open first tile to decide
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
    # Prepare scaled image and drawing context
    if scale != 1.0:
        new_w = max(1, int(round(canvas_w * scale)))
        new_h = max(1, int(round(canvas_h * scale)))
        out_scaled = out_full.resize((new_w, new_h), Image.BILINEAR)
    else:
        out_scaled = out_full

    # Ensure drawing surface supports RGBA (for colored/alpha overlays)
    if out_scaled.mode != "RGBA":
        out_scaled = out_scaled.convert("RGBA")

    draw = ImageDraw.Draw(out_scaled)
    # Optionally draw tile borders/labels for debugging
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

    # Build mapping from tile stem -> (offset, size)
    tile_info = {}
    for t, x, y in positions:
        key = normalize_key(t["image_path"])  # e.g., "x042 y001"
        tile_info[key] = {
            "shifted_x": x + shift_x,
            "shifted_y": y + shift_y,
            "w": t["w"],
            "h": t["h"],
            "path": str(img_root / t["image_path"]),
        }

    # Draw COCO bboxes
    font = load_font(max(10, int(args.font_size * (scale if scale != 0 else 1))))

    # Create mapping from COCO image entry -> tile_info
    # The COCO may have file_name without ext or with a root prefix; we match by stem
    # If multiple images share the same stem, we take the first match (and warn).
    matched = 0
    skipped = 0

    for img_id, im in list(img_by_id.items()):
        key = normalize_key(im.get("file_name", ""))
        # If not found directly, try to strip any additional suffix/prefix heuristically
        tinfo = tile_info.get(key)
        if tinfo is None:
            # Try with stem of basename again (already normalized). If still missing, skip.
            # Optionally, try to match by exact filename including extension if present in XML
            # (we already used stem which is the best we can do), so just warn.
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
            # transform into mosaic coords
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
                # background box for readability
                # text size (compat across Pillow versions)
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

    # Save
    out_scaled.save(out_path)
    print(f"[OK] Wrote mosaic with overlays: {out_path}  (scale={scale:.4f}, canvas={canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
