#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patchify images and YOLO-OBB polygon labels (4-point polygons in normalized coords).

Input tree (recursive):
  - Images under --img_dir (any subdir structure)
  - Labels under --label_dir mirroring image relative paths, with .txt files where
    each line is: cls x1 y1 x2 y2 x3 y3 x4 y4  (coords normalized to [0,1])

Output tree (mirrors input subdirs):
  --out_dir/
      images/<relpath_without_ext>__r{row}_c{col}.{ext}
      labels/<relpath_without_ext>__r{row}_c{col}.txt

Rules:
  * Each object is written once. By default, objects that can share a patch are
    grouped and assigned to one common patch. Use --assign_mode best to assign
    each object independently, or all to duplicate objects into all overlaps.
  * Optional (--shift_crop_to_fit): try to shift the crop origin so ALL objects
    in the same tile fit without clipping. If full fit is impossible, fallback to
    default crop origin (original behavior).
  * Slide algorithm = move a pair of vertices along the longer adjacent edge
    (shape may change slightly). No hard-clip mode is supported.
  * Polygons are shifted to patch-local coords, slid into [0,P]^2,
    normalized back to [0,1], and dropped if post-slide area < --min_area.
  * Optionally keep negative patches with --keep_neg and ratio --max_neg_ratio.
"""

import argparse
from pathlib import Path
import math
from collections import defaultdict
from typing import List
import cv2, numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------- geometry helpers (original slide algorithm) ----------

def _poly_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _ensure_ccw(P: np.ndarray) -> np.ndarray:
    cross = np.cross(P[1] - P[0], P[2] - P[1])
    return P if cross >= 0 else P[::-1]

def _edge_lengths(P: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(P[(i + 1) % 4] - P[i]) for i in range(4)], float)

def _is_oob_point(p: np.ndarray) -> bool:
    return (p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1)

def _feasible_t_interval_for_point(q: np.ndarray, d: np.ndarray):
    t_low, t_high = -np.inf, np.inf
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if q[i] < 0 or q[i] > 1:
                return None
        else:
            t0 = (0 - q[i]) / d[i]
            t1 = (1 - q[i]) / d[i]
            lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
            t_low = max(t_low, lo)
            t_high = min(t_high, hi)
    return None if t_low > t_high else (t_low, t_high)

def _candidate_t_to_touch_boundary(p: np.ndarray, d: np.ndarray):
    c = []
    for i in range(2):
        if abs(d[i]) > 1e-12:
            c += [(0 - p[i]) / d[i], (1 - p[i]) / d[i]]
    return [t for t in c if np.isfinite(t)]

def _move_pair_along_long_edge(P: np.ndarray, i: int, area_min: float = 1e-6):
    L = _edge_lengths(P)
    e_prev, e_next = L[(i - 1) % 4], L[i % 4]
    if e_prev >= e_next:
        d = P[i] - P[(i - 1) % 4]; j = (i + 1) % 4
    else:
        d = P[(i + 1) % 4] - P[i]; j = (i - 1) % 4

    n = np.linalg.norm(d)
    if n < 1e-12:
        return False, P
    d = d / n

    def try_with_direction(d_vec: np.ndarray):
        p = P[i].copy()
        q = P[j].copy()
        Ip = _feasible_t_interval_for_point(p, d_vec)
        Iq = _feasible_t_interval_for_point(q, d_vec)
        if Ip is None or Iq is None:
            return None

        t_low = max(Ip[0], Iq[0])
        t_high = min(Ip[1], Iq[1])
        if t_low > t_high:
            return None

        def target_ts(qpt: np.ndarray):
            ts = []
            if qpt[0] < 0 and d_vec[0] > 1e-12:
                ts.append((0.0 - qpt[0]) / d_vec[0])
            elif qpt[0] > 1 and d_vec[0] < -1e-12:
                ts.append((1.0 - qpt[0]) / d_vec[0])
            if qpt[1] < 0 and d_vec[1] > 1e-12:
                ts.append((0.0 - qpt[1]) / d_vec[1])
            elif qpt[1] > 1 and d_vec[1] < -1e-12:
                ts.append((1.0 - qpt[1]) / d_vec[1])
            return ts

        pref = []
        for t in (target_ts(p) + target_ts(q)):
            if t_low - 1e-12 <= t <= t_high + 1e-12:
                pp = p + t * d_vec; qq = q + t * d_vec
                if (pp[0] >= -1e-9 and pp[0] <= 1 + 1e-9 and
                    pp[1] >= -1e-9 and pp[1] <= 1 + 1e-9 and
                    qq[0] >= -1e-9 and qq[0] <= 1 + 1e-9 and
                    qq[1] >= -1e-9 and qq[1] <= 1 + 1e-9):
                    pref.append(t)

        if pref:
            t = min(pref, key=lambda x: abs(x))
        else:
            cand = []
            for qpt in (p, q):
                cand.extend([t for t in _candidate_t_to_touch_boundary(qpt, d_vec)
                             if t_low - 1e-12 <= t <= t_high + 1e-12])
            if cand:
                t = min(cand, key=lambda x: abs(x))
            else:
                t = t_low if abs(t_low) < abs(t_high) else t_high

        v = t * d_vec
        P2 = P.copy()
        P2[i] += v
        P2[j] += v
        P2 = np.clip(P2, 0, 1)

        if _poly_area(P2) < area_min:
            return None
        return P2

    P_try = try_with_direction(d)
    if P_try is not None:
        return True, P_try

    P_try = try_with_direction(-d)
    if P_try is not None:
        return True, P_try

    return False, P

def slide_clip_normalized(P01: np.ndarray, area_min01: float = 1e-6):
    P = _ensure_ccw(P01.copy())

    if not any(_is_oob_point(P[k]) for k in range(4)):
        return P if _poly_area(P) >= area_min01 else None

    for _ in range(4):
        oob = next((k for k in range(4) if _is_oob_point(P[k])), None)
        if oob is None:
            break
        ok, Pn = _move_pair_along_long_edge(P, oob, area_min01)
        if not ok:
            return None
        P = Pn

    return _ensure_ccw(P) if _poly_area(P) >= area_min01 else None

def slide_fit_quad_in_patch(pts_px: np.ndarray, P: int, area_min_px: float = 16.0):
    Q01 = slide_clip_normalized(pts_px / float(P), area_min_px / (P ** 2))
    return None if Q01 is None else Q01 * float(P)

# ---------- IO and patch main ----------

def find_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS])

def read_label(lbl: Path, W: int, H: int):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        sp = line.split()
        if len(sp) != 9:
            continue
        cls = int(sp[0])
        pts = np.array(list(map(float, sp[1:])), float).reshape(4, 2)
        abs_ = np.stack([pts[:, 0] * W, pts[:, 1] * H], 1)
        out.append((cls, abs_))
    return out

def poly_centroid(pts: np.ndarray):
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())

def save_label(path: Path, items, tile_W: int, tile_H: int):
    with open(path, 'w') as f:
        for cls, pts in items:
            norm = np.empty_like(pts)
            norm[:, 0] = np.clip(pts[:, 0] / tile_W, 0, 1)
            norm[:, 1] = np.clip(pts[:, 1] / tile_H, 0, 1)
            f.write(str(cls) + ' ' + ' '.join(f"{v:.6f}" for v in norm.reshape(-1)) + "\n")

def _tile_origin(r: int, c: int, W: int, H: int, P: int, S: int):
    x0 = min(c * S, max(0, W - P))
    y0 = min(r * S, max(0, H - P))
    return x0, y0

def _owning_tile(cx: float, cy: float, W: int, H: int, P: int, S: int):
    n_rows = max(1, math.ceil((H - P) / S) + 1)
    n_cols = max(1, math.ceil((W - P) / S) + 1)
    c = min(int(cx // S), n_cols - 1)
    r = min(int(cy // S), n_rows - 1)
    return r, c

def _neighbor_tiles(r0: int, c0: int, W: int, H: int, P: int, S: int):
    n_rows = max(1, math.ceil((H - P) / S) + 1)
    n_cols = max(1, math.ceil((W - P) / S) + 1)
    tiles = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = r0 + dr; c = c0 + dc
            if 0 <= r < n_rows and 0 <= c < n_cols:
                tiles.append((r, c))
    return tiles

def _intersect_interval(a0: float, a1: float, b0: float, b1: float):
    lo = max(a0, b0)
    hi = min(a1, b1)
    return None if lo > hi else (lo, hi)

def _nearest_in_interval(v: float, lo: float, hi: float):
    return lo if v < lo else hi if v > hi else v

def _suggest_shifted_origin_for_object(pts_abs: np.ndarray, x0: int, y0: int, W: int, H: int, P: int):
    """
    Compute the crop origin closest to (x0, y0) that fully contains the object bbox.
    Returns (xs, ys) or None if the object is larger than the patch in any dimension.
    """
    max_x0 = max(0, W - P)
    max_y0 = max(0, H - P)

    minx = float(np.min(pts_abs[:, 0]))
    maxx = float(np.max(pts_abs[:, 0]))
    miny = float(np.min(pts_abs[:, 1]))
    maxy = float(np.max(pts_abs[:, 1]))

    # Integer valid range for origin:
    #   xs must satisfy ceil(maxx - P) <= xs <= floor(minx)  so bbox fully inside [xs, xs+P]
    #   xs must also be in [0, max_x0]
    xs_lo = max(0, math.ceil(maxx - P))
    xs_hi = min(max_x0, math.floor(minx))
    ys_lo = max(0, math.ceil(maxy - P))
    ys_hi = min(max_y0, math.floor(miny))

    if xs_lo > xs_hi or ys_lo > ys_hi:
        return None  # object larger than patch or doesn't fit in image

    xs = int(_nearest_in_interval(float(x0), float(xs_lo), float(xs_hi)))
    ys = int(_nearest_in_interval(float(y0), float(ys_lo), float(ys_hi)))
    return xs, ys

def _suggest_best_effort_origin_for_object(pts_abs: np.ndarray, x0: int, y0: int, W: int, H: int, P: int):
    """
    Like _suggest_shifted_origin_for_object but tolerates OOB vertices.
    Clamps the effective bbox to image bounds [0,W]x[0,H] and returns the
    origin that maximises in-image coverage.  Pushes the crop toward the
    OOB side when the in-image portion is wider/taller than P.
    Returns (xs, ys) or None if the object is entirely outside the image.
    """
    max_x0 = max(0, W - P)
    max_y0 = max(0, H - P)

    eff_minx = max(0.0, float(np.min(pts_abs[:, 0])))
    eff_maxx = min(float(W), float(np.max(pts_abs[:, 0])))
    eff_miny = max(0.0, float(np.min(pts_abs[:, 1])))
    eff_maxy = min(float(H), float(np.max(pts_abs[:, 1])))

    if eff_minx >= eff_maxx or eff_miny >= eff_maxy:
        return None  # object entirely outside image

    xs_lo = max(0, math.ceil(eff_maxx - P))
    xs_hi = min(max_x0, math.floor(eff_minx))
    ys_lo = max(0, math.ceil(eff_maxy - P))
    ys_hi = min(max_y0, math.floor(eff_miny))

    # If interval is valid, pick origin nearest to current tile origin
    if xs_lo <= xs_hi:
        xs = int(_nearest_in_interval(float(x0), float(xs_lo), float(xs_hi)))
    else:
        # In-image x portion is wider than P; push crop toward the OOB side
        xs = max(0, min(max_x0, math.ceil(eff_maxx - P)))

    if ys_lo <= ys_hi:
        ys = int(_nearest_in_interval(float(y0), float(ys_lo), float(ys_hi)))
    else:
        ys = max(0, min(max_y0, math.ceil(eff_maxy - P)))

    return xs, ys

def _eval_object_at_origin(pts_abs: np.ndarray, x0: int, y0: int, P: int, min_area: float):
    """Shift pts to patch-local coords and apply slide. Returns (poly_px, area) or (None, -1)."""
    local = pts_abs.copy()
    local[:, 0] -= x0
    local[:, 1] -= y0
    poly = slide_fit_quad_in_patch(local, P, area_min_px=min_area)
    if poly is None:
        return None, -1.0
    area = _poly_area(poly / float(P)) * (P ** 2)
    return poly, area

def _best_tile_for_object(pts_abs, cx, cy, W, H, P, S, min_area):
    """
    Among owner+8 neighbors, choose the tile where slide keeps max area.
    Returns ((r, c), area) or (None, None).
    Origin shift is NOT considered here; handled in _resolve_tile_crop_and_labels.
    """
    r0, c0 = _owning_tile(cx, cy, W, H, P, S)
    candidates = _neighbor_tiles(r0, c0, W, H, P, S)

    best = None
    best_area = -1.0

    for (r, c) in candidates:
        x0, y0 = _tile_origin(r, c, W, H, P, S)
        poly, area = _eval_object_at_origin(pts_abs, x0, y0, P, min_area)
        if poly is None:
            continue
        score = (area, 1 if (r, c) == (r0, c0) else 0)
        if score > (best_area, 1 if best and best == (r0, c0) else 0):
            best_area = area
            best = (r, c)

    if best is None or best_area < min_area:
        return (None, None)
    return best, best_area

def _overlapping_tiles_for_object(pts_abs, W, H, P, S, min_area):
    """
    Return all tiles where the object's bbox overlaps the crop and the slid
    polygon keeps enough area. This is the usual detection-training behavior:
    overlapped crops should each get the visible object label.
    """
    n_rows = max(1, math.ceil((H - P) / S) + 1)
    n_cols = max(1, math.ceil((W - P) / S) + 1)

    minx = float(np.min(pts_abs[:, 0]))
    maxx = float(np.max(pts_abs[:, 0]))
    miny = float(np.min(pts_abs[:, 1]))
    maxy = float(np.max(pts_abs[:, 1]))

    tiles = []
    seen_origins = set()
    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = _tile_origin(r, c, W, H, P, S)
            origin = (x0, y0)
            if origin in seen_origins:
                continue
            seen_origins.add(origin)

            # Strict separation test for bbox-vs-crop. Touching only at an
            # edge has zero area and will be ignored.
            if maxx <= x0 or minx >= x0 + P or maxy <= y0 or miny >= y0 + P:
                continue

            poly, area = _eval_object_at_origin(pts_abs, x0, y0, P, min_area)
            if poly is not None and area >= min_area:
                tiles.append((r, c))
    return tiles

def _group_objects_by_shared_tiles(object_options):
    """
    Build connected components where objects are connected if they have at least
    one feasible tile in common. This keeps nearby objects together when a patch
    can show them together, while still writing each source object only once.
    """
    tile_to_objs = defaultdict(list)
    for idx, options in enumerate(object_options):
        for tile in options:
            tile_to_objs[tile].append(idx)

    parent = list(range(len(object_options)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for obj_ids in tile_to_objs.values():
        if len(obj_ids) < 2:
            continue
        first = obj_ids[0]
        for obj_id in obj_ids[1:]:
            union(first, obj_id)

    groups = defaultdict(list)
    for idx in range(len(object_options)):
        groups[find(idx)].append(idx)
    return list(groups.values())

def _group_indices_by_shared_tiles(indices, object_options):
    local_options = [object_options[idx] for idx in indices]
    local_groups = _group_objects_by_shared_tiles(local_options)
    return [[indices[local_idx] for local_idx in group] for group in local_groups]

def _best_shared_tile_for_group(group, object_options, objects, W, H, P, S, min_area):
    """
    Pick one tile for a group. Prefer tiles that can keep the most group objects,
    then the largest retained area. This guarantees each source object is
    assigned once, even when no single tile can keep the whole group.
    """
    candidate_tiles = sorted(set().union(*(object_options[i] for i in group)))
    best_tile = None
    best_members = []
    best_score = (-1, -1.0)

    for tile in candidate_tiles:
        r, c = tile
        x0, y0 = _tile_origin(r, c, W, H, P, S)
        members = []
        total_area = 0.0
        for idx in group:
            cls, pts = objects[idx]
            if tile not in object_options[idx]:
                continue
            poly, area = _eval_object_at_origin(pts, x0, y0, P, min_area)
            if poly is None or area < min_area:
                continue
            members.append(idx)
            total_area += area

        score = (len(members), total_area)
        if score > best_score:
            best_score = score
            best_tile = tile
            best_members = members

    return best_tile, best_members

def _resolve_tile_crop_and_labels(items, r, c, W, H, P, S, min_area, shift_crop_to_fit=False):
    """
    Decide the crop origin for this tile and return (origin, kept_labels).
    items: list of (cls, pts_abs) in image-pixel coords.

    Without --shift_crop_to_fit (default):
      Use the default tile origin. Partial results (some objects clipped) are OK.

    With --shift_crop_to_fit:
      Try to find an origin that keeps ALL objects unclipped.
      Candidates: default origin + per-object ideal origins.
      If no candidate keeps all objects → fallback to default (partial OK).
    """
    x0, y0 = _tile_origin(r, c, W, H, P, S)
    default_origin = (x0, y0)

    def eval_partial(origin):
        """Evaluate origin; accept partial results (skip failed objects)."""
        ox, oy = origin
        kept = []
        total_area = 0.0
        for cls, pts_abs in items:
            poly, area = _eval_object_at_origin(pts_abs, ox, oy, P, min_area)
            if poly is None:
                continue
            kept.append((cls, poly))
            total_area += area
        return kept, total_area

    def eval_all(origin):
        """Evaluate origin; return None if ANY object is not fully inside [0,P]."""
        ox, oy = origin
        kept = []
        total_area = 0.0
        for cls, pts_abs in items:
            local_pts = pts_abs.copy()
            local_pts[:, 0] -= ox
            local_pts[:, 1] -= oy
            if np.any(local_pts < 0) or np.any(local_pts > P):
                return None, -1.0
            poly, area = _eval_object_at_origin(pts_abs, ox, oy, P, min_area)
            if poly is None:
                return None, -1.0
            kept.append((cls, poly))
            total_area += area
        return kept, total_area

    if not shift_crop_to_fit:
        kept, _ = eval_partial(default_origin)
        return default_origin, kept

    # --- shift_crop_to_fit path ---
    # Check if default already keeps everything
    default_kept_all, default_area_all = eval_all(default_origin)
    if default_kept_all is not None:
        return default_origin, default_kept_all

    # Build candidate origins:
    #   _suggest_shifted_origin_for_object  – strict: only for fully in-image objects
    #   _suggest_best_effort_origin_for_object – tolerant: also handles OOB vertices
    candidates = set()
    for _, pts_abs in items:
        s = _suggest_shifted_origin_for_object(pts_abs, x0, y0, W, H, P)
        if s is not None:
            candidates.add(s)
        be = _suggest_best_effort_origin_for_object(pts_abs, x0, y0, W, H, P)
        if be is not None:
            candidates.add(be)

    # Phase 1: find a candidate where ALL objects fit fully (no OOB vertices at all)
    best_origin = None
    best_kept = None
    best_area = -1.0

    for origin in candidates:
        kept, total_area = eval_all(origin)
        if kept is not None and total_area > best_area:
            best_area = total_area
            best_origin = origin
            best_kept = kept

    if best_origin is not None:
        return best_origin, best_kept

    # Phase 2: some vertices are irreversibly OOB (outside image bounds).
    # Among all candidates + default, pick the origin that maximises total
    # retained area (using slide algorithm via eval_partial).
    candidates.add(default_origin)
    best_origin = None
    best_kept = None
    best_area = -1.0

    for origin in candidates:
        kept, total_area = eval_partial(origin)
        if total_area > best_area:
            best_area = total_area
            best_origin = origin
            best_kept = kept

    if best_origin is not None:
        return best_origin, best_kept

    # Ultimate fallback
    kept, _ = eval_partial(default_origin)
    return default_origin, kept

def process_one(img_path, rel, lbl_root, out_root, P, S, min_area,
                keep_neg, max_neg, force_ext, dry_run, shift_crop_to_fit=False,
                assign_mode="group"):
    lbl_path = lbl_root / rel.with_suffix('.txt')
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return (0, 0)
    H, W = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    objs = read_label(lbl_path, W, H)
    if len(objs) == 0 and not keep_neg:
        return (0, 0)

    # Pre-assign each object to target tiles.
    assigned = {}
    indexed_objs = []
    object_options = []

    for cls, pts in objs:
        if assign_mode == "best":
            cx, cy = poly_centroid(pts)
            best, best_area = _best_tile_for_object(
                pts_abs=pts, cx=cx, cy=cy, W=W, H=H, P=P, S=S, min_area=min_area)
            tiles = [] if best is None or best_area < min_area else [best]
        else:
            tiles = _overlapping_tiles_for_object(
                pts_abs=pts, W=W, H=H, P=P, S=S, min_area=min_area)

        if not tiles:
            continue

        indexed_objs.append((cls, pts))
        object_options.append(tiles)

    if assign_mode == "all":
        for idx, tiles in enumerate(object_options):
            cls, pts = indexed_objs[idx]
            for tile in tiles:
                if tile not in assigned:
                    assigned[tile] = []
                assigned[tile].append((cls, pts))
    elif assign_mode == "group":
        pending_groups = _group_objects_by_shared_tiles(object_options)
        while pending_groups:
            next_groups = []
            for group in pending_groups:
                tile, members = _best_shared_tile_for_group(
                    group, object_options, indexed_objs, W, H, P, S, min_area)
                if tile is None or not members:
                    continue
                if tile not in assigned:
                    assigned[tile] = []
                for idx in members:
                    assigned[tile].append(indexed_objs[idx])

                member_set = set(members)
                leftovers = [idx for idx in group if idx not in member_set]
                if leftovers:
                    next_groups.extend(_group_indices_by_shared_tiles(leftovers, object_options))
            pending_groups = next_groups
    else:
        for idx, tiles in enumerate(object_options):
            cls, pts = indexed_objs[idx]
            tile = tiles[0]
            if tile not in assigned:
                assigned[tile] = []
            assigned[tile].append((cls, pts))

    # Resolve crop origin and labels for each occupied tile
    tile_outputs = {}
    for (r, c), items in assigned.items():
        origin, kept = _resolve_tile_crop_and_labels(
            items=items, r=r, c=c, W=W, H=H, P=P, S=S, min_area=min_area,
            shift_crop_to_fit=shift_crop_to_fit)
        if kept:
            tile_outputs[(r, c)] = (origin, kept)

    n_rows = max(1, math.ceil((H - P) / S) + 1)
    n_cols = max(1, math.ceil((W - P) / S) + 1)
    ext = (force_ext or img_path.suffix[1:])
    pos = neg = 0

    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = _tile_origin(r, c, W, H, P, S)
            kept = []
            if (r, c) in tile_outputs:
                (x0, y0), kept = tile_outputs[(r, c)]
            x1, y1 = x0 + P, y0 + P
            relname = f"{rel.as_posix()}__r{r}_c{c}"
            out_img = out_root / 'images' / f"{relname}.{ext}"
            out_lbl = out_root / 'labels' / f"{relname}.txt"

            if not kept:
                if keep_neg and neg < math.ceil(max_neg * max(1, len(objs))) and not dry_run:
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    out_lbl.parent.mkdir(parents=True, exist_ok=True)
                    tile = img[y0:y1, x0:x1]
                    cv2.imencode(f'.{ext}', tile)[1].tofile(str(out_img))
                    open(out_lbl, 'w').close()
                neg += 1
                continue

            if not dry_run:
                out_img.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                tile = img[y0:y1, x0:x1]
                cv2.imencode(f'.{ext}', tile)[1].tofile(str(out_img))
                actual_tile_H, actual_tile_W = tile.shape[:2]
                save_label(out_lbl, kept, actual_tile_W, actual_tile_H)
            pos += 1
    return pos, neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img_dir', required=True)
    ap.add_argument('-l', '--label_dir', required=True)
    ap.add_argument('-o', '--out_dir', required=True)
    ap.add_argument('-p', '--patch_size', type=int, default=1280)
    ap.add_argument('-s', '--stride', type=int, default=384)
    ap.add_argument('--min_area', type=float, default=16.)
    ap.add_argument('--keep_neg', action='store_true')
    ap.add_argument('--max_neg_ratio', type=float, default=1.)
    ap.add_argument('--shift_crop_to_fit', action='store_true',
                    help='Move crop origin to avoid cutting bboxes when possible; '
                         'fallback to default crop if full fit is impossible.')
    ap.add_argument('--assign_mode', choices=['group', 'best', 'all'], default='group',
                    help='Label assignment mode. group writes each object once and keeps shared '
                         'objects together when possible; best assigns objects independently; '
                         'all duplicates objects into every overlapping patch.')
    ap.add_argument('--ext', default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    img_root, lbl_root, out_root = Path(args.img_dir), Path(args.label_dir), Path(args.out_dir)
    (out_root / 'images').mkdir(parents=True, exist_ok=True)
    (out_root / 'labels').mkdir(parents=True, exist_ok=True)

    total_pos = total_neg = 0
    for img_path in find_images(img_root):
        rel = img_path.relative_to(img_root).with_suffix('')
        pos, neg = process_one(
            img_path=img_path, rel=rel, lbl_root=Path(args.label_dir), out_root=Path(args.out_dir),
            P=args.patch_size, S=args.stride,
            min_area=args.min_area, keep_neg=args.keep_neg, max_neg=args.max_neg_ratio,
            force_ext=args.ext, dry_run=args.dry_run, shift_crop_to_fit=args.shift_crop_to_fit,
            assign_mode=args.assign_mode)
        total_pos += pos; total_neg += neg
    print(f"[DONE] Pos={total_pos} Neg={total_neg}")

if __name__ == '__main__':
    main()
