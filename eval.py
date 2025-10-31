#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay GT and inference OBBs on images and save, with optional TTA (flip/rotate) and rotated-NMS merge.
"""
import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import json
import csv
import shutil
from glob import glob
from typing import List, Tuple, Dict, Optional

import torch
from torchvision.ops import nms as tv_nms
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------- IO helpers ----------------------

def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_images(entry) -> List[Path]:
    paths: List[Path] = []
    if isinstance(entry, (list, tuple)):
        for e in entry:
            paths.extend(find_images(e))
        return sorted(set(paths))
    entry = Path(entry)
    if entry.is_dir():
        for ext in IMG_EXTS:
            paths.extend(entry.rglob(f"*{ext}"))
    elif entry.is_file():
        if entry.suffix.lower() in {".txt", ".lst", ".list"}:
            with open(entry, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        p = Path(s)
                        if p.exists() and p.suffix.lower() in IMG_EXTS:
                            paths.append(p)
        elif entry.suffix.lower() in IMG_EXTS:
            paths.append(entry)
    return sorted(paths)

def map_image_to_label(img_path: Path) -> Path | None:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        lbl_parts = parts.copy()
        lbl_parts[idx] = "labels"
        lbl_dir = Path(*lbl_parts[:-1])
        cand = lbl_dir / f"{img_path.stem}.txt"
        if cand.exists():
            return cand
        found = list(lbl_dir.rglob(f"{img_path.stem}.txt"))
        if found:
            return found[0]
    root = img_path.parents[2] if len(img_path.parents) >= 3 else img_path.parent
    for lbl_root in [root / "labels", img_path.parent.parent / "labels"]:
        if lbl_root.exists():
            found = list(lbl_root.rglob(f"{img_path.stem}.txt"))
            if found:
                return found[0]
    cand2 = img_path.with_suffix(".txt")
    return cand2 if cand2.exists() else None

# ---------------------- Geometry ----------------------

def normalize_angle_deg(a: float) -> float:
    # normalize to (-90, 90]
    while a <= -90.0:
        a += 180.0
    while a > 90.0:
        a -= 180.0
    return a

def obb_to_polygon(cx, cy, w, h, angle_deg):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    hw, hh = w / 2.0, h / 2.0
    rect = np.array([[-hw, -hh],
                     [ hw, -hh],
                     [ hw,  hh],
                     [-hw,  hh]], dtype=np.float32)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    pts = rect @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts

def poly_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """凸四角形どうしのIoU（点順序をCCWに正規化してから intersectConvexConvex）。"""
    p1 = _ensure_ccw(poly1.astype(np.float32))
    p2 = _ensure_ccw(poly2.astype(np.float32))

    a1 = abs(cv2.contourArea(p1))
    a2 = abs(cv2.contourArea(p2))
    if a1 <= 0.0 or a2 <= 0.0:
        return 0.0

    inter, _ = cv2.intersectConvexConvex(p1, p2)
    if inter <= 0.0:
        return 0.0
    return float(inter / max(a1 + a2 - inter, 1e-9))

def _signed_area(poly: np.ndarray) -> float:
    # poly: (N,2)
    x = poly[:,0]; y = poly[:,1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    # OpenCVの intersectConvexConvex は CCW が安定
    if _signed_area(poly) < 0:
        return poly[::-1].copy()
    return poly

def _compute_ap(rec, prec):
    """COCO風：precisionのエンベロープ化→面積（数値積分）。"""
    mrec  = np.concatenate(([0.0], rec,  [1.0]))
    mprec = np.concatenate(([0.0], prec, [0.0]))
    # precisionの単調減少包絡
    for i in range(mprec.size - 1, 0, -1):
        mprec[i - 1] = max(mprec[i - 1], mprec[i])
    # 変化点だけ積分
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1]))
    return ap


def eval_tta_detections(records, class_names: dict | None, iou_thrs: list[float]):
    """
    records: list of dict per image:
      { 'image_id': str,
        'gt':   [{'cls':int, 'poly':np.ndarray(4,2)}, ...],
        'pred': [{'cls':int, 'conf':float, 'poly':np.ndarray(4,2)}, ...] }
    iou_thrs: [0.50,0.55,...] 等
    戻り値: dict（mAP50, mAP50_95, per_classなど）
    """
    # クラス一覧
    if class_names:
        cls_ids = sorted(class_names.keys())
    else:
        # 推定：recordsから出現クラスを収集
        s = set()
        for r in records:
            for g in r['gt']:   s.add(int(g['cls']))
            for p in r['pred']: s.add(int(p['cls']))
        cls_ids = sorted(int(c) for c in s if c is not None and c >= 0)

    # 画像ごとのGTをクラス単位に索引化
    gt_by_img_cls = {}  # (img, cls) -> list of {'poly','matched':False}
    npos_by_cls = {cid: 0 for cid in cls_ids}
    for r in records:
        img_id = r['image_id']
        gts = {}
        for g in r['gt']:
            cid = int(g['cls'])
            if cid not in cls_ids: 
                continue
            gts.setdefault(cid, []).append({'poly': g['poly'], 'matched': False})
            npos_by_cls[cid] += 1
        for cid, arr in gts.items():
            gt_by_img_cls[(img_id, cid)] = arr

    # しきい値ごとにAP計算
    aps_50 = {}     # cls -> AP@0.50
    aps_range = {}  # cls -> mean AP over iou_thrs
    for cid in cls_ids:
        # 予測を集約してconf降順
        preds = []
        for r in records:
            img_id = r['image_id']
            for p in r['pred']:
                if int(p['cls']) == cid:
                    preds.append((img_id, float(p['conf']), p['poly']))
        preds.sort(key=lambda x: -x[1])

        # しきい値ごとのAPを蓄積
        ap_per_thr = []
        for thr in iou_thrs:
            tp = np.zeros(len(preds), dtype=np.float32)
            fp = np.zeros(len(preds), dtype=np.float32)

            # 各画像のGTマッチ状態をリセット
            for key, arr in gt_by_img_cls.items():
                if key[1] == cid:
                    for it in arr: 
                        it['matched'] = False

            for i, (img_id, conf, ppoly) in enumerate(preds):
                gts = gt_by_img_cls.get((img_id, cid), [])
                iou_max = 0.0
                jmax = -1
                for j, g in enumerate(gts):
                    if g['matched']:
                        continue
                    iou = poly_iou(ppoly, g['poly'])
                    if iou > iou_max:
                        iou_max = iou; jmax = j
                if iou_max >= thr and jmax >= 0:
                    tp[i] = 1.0
                    gts[jmax]['matched'] = True
                else:
                    fp[i] = 1.0

            # 適合率・再現率
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            denom = np.maximum(cum_tp + cum_fp, 1e-12)
            prec = cum_tp / denom
            rec  = cum_tp / max(npos_by_cls[cid], 1e-12)

            ap = _compute_ap(rec, prec) if npos_by_cls[cid] > 0 else 0.0
            ap_per_thr.append(ap)

            # thr==0.50 のときだけ保存
            if abs(thr - 0.50) < 1e-9:
                aps_50[cid] = ap

        aps_range[cid] = float(np.mean(ap_per_thr)) if ap_per_thr else 0.0

    # まとめ
    mAP50 = float(np.mean(list(aps_50.values()))) if aps_50 else 0.0
    mAP50_95 = float(np.mean(list(aps_range.values()))) if aps_range else 0.0
    per_class = {}
    for cid in cls_ids:
        name = class_names.get(cid, str(cid)) if class_names else str(cid)
        per_class[name] = {"AP50": float(aps_50.get(cid, 0.0)),
                           "AP50_95": float(aps_range.get(cid, 0.0))}
    return {
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "per_class": per_class,
        "num_images": len(records),
        "num_classes": len(cls_ids),
    }

def poly_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """凸四角形どうしのIoU（OpenCVの intersectConvexConvex 使用）。"""
    p1 = poly1.astype(np.float32)
    p2 = poly2.astype(np.float32)
    area1 = abs(cv2.contourArea(p1))
    area2 = abs(cv2.contourArea(p2))
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    if inter_area <= 0.0:
        return 0.0
    union = area1 + area2 - inter_area
    return float(inter_area / max(union, 1e-9))


def _xywhr_to_aabb(xywhr_list):
    """(cx,cy,w,h,deg) -> (x1,y1,x2,y2) のAABBに変換（NMSの前処理用）。"""
    aabbs = []
    for (cx, cy, w, h, ang) in xywhr_list:
        poly = obb_to_polygon(cx, cy, w, h, ang)
        x1, y1 = np.min(poly, axis=0)
        x2, y2 = np.max(poly, axis=0)
        aabbs.append([x1, y1, x2, y2])
    return torch.tensor(aabbs, dtype=torch.float32)


def merge_rotated_nms_fallback(xywhr_list, confs, classes, iou_thr=0.5, per_class=True):
    """
    速い二段構え：
      1) torchvision.ops.nms を AABB で実行（高速な粗選別）
      2) 残った集合で、ポリゴンIoUの貪欲NMS（精密仕上げ、回転考慮）
    """
    n = len(xywhr_list)
    if n == 0:
        return [], [], []

    # --- 粗選別: AABBで標準nms ---
    boxes_aabb = _xywhr_to_aabb(xywhr_list)
    scores = torch.tensor(confs, dtype=torch.float32)
    keep_idx = tv_nms(boxes_aabb, scores, 0.3)  # 緩めに（0.3推奨）
    keep = keep_idx.cpu().numpy().tolist()

    xywhr = [xywhr_list[i] for i in keep]
    confs_k = [confs[i] for i in keep]
    cls_k = [classes[i] for i in keep]
    polys = [obb_to_polygon(*b) for b in xywhr]

    # --- 仕上げ: 回転IoUの貪欲NMS ---
    order = np.argsort(-np.asarray(confs_k))
    used = np.zeros(len(order), dtype=bool)
    final = []

    for oi in order:
        if used[oi]:
            continue
        final.append(oi)
        used[oi] = True
        for oj in order:
            if used[oj]:
                continue
            if per_class and (cls_k[oi] != cls_k[oj]):
                continue
            if poly_iou(polys[oi], polys[oj]) >= iou_thr:
                used[oj] = True

    out_boxes = [xywhr[i] for i in final]
    out_scores = [confs_k[i] for i in final]
    out_cls = [cls_k[i] for i in final]
    return out_boxes, out_scores, out_cls

def load_gt_polys_auto(lbl_path: Path, img_w: int, img_h: int) -> list[tuple[np.ndarray, int]]:
    polys: list[tuple[np.ndarray, int]] = []
    if not lbl_path or not lbl_path.exists():
        return polys

    def is_float(s: str) -> bool:
        try:
            float(s); return True
        except: return False

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if not is_float(toks[0]):
                nums = [t for t in toks if is_float(t)]
            else:
                nums = toks

            if 6 <= len(nums) < 9:
                try:
                    cls = int(float(nums[0]))
                    cx, cy, w, h, ang = map(float, nums[1:6])
                except Exception:
                    continue
                cx, cy = cx * img_w, cy * img_h
                w, h = w * img_w, h * img_h
                ang_deg = ang if abs(ang) > 3.2 else np.rad2deg(ang)
                poly = obb_to_polygon(cx, cy, w, h, ang_deg)
                polys.append((poly, cls))
                continue

            if len(nums) >= 9:
                try:
                    cls = int(float(nums[0]))
                except Exception:
                    cls = 0
                vals = list(map(float, nums[1:9]))
                xs = vals[0::2]; ys = vals[1::2]
                frac_01 = sum(0.0 <= v <= 1.0 for v in vals) / len(vals)
                if frac_01 >= 0.75:
                    xs = [x * img_w for x in xs]
                    ys = [y * img_h for y in ys]
                poly = np.array(list(zip(xs, ys)), dtype=np.float32)
                if poly.shape == (4, 2):
                    polys.append((poly, cls))
                continue
    return polys

def as_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

def draw_poly(img: np.ndarray, poly: np.ndarray, color: Tuple[int,int,int], thick: int, label: Optional[str] = None):
    pts = poly.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
    if label:
        p0 = tuple(pts[0, 0].tolist())
        cv2.putText(img, label, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ---------------------- TTA transforms ----------------------

class TTATransform:
    """Defines a forward transform on image and the inverse mapping for OBB (cx,cy,w,h,deg)."""
    def __init__(self, name: str): self.name = name

    def apply_img(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.name == "identity":
            return img
        if self.name == "hflip":
            return cv2.flip(img, 1)
        if self.name == "vflip":
            return cv2.flip(img, 0)
        if self.name == "rot90":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.name == "rot180":
            return cv2.rotate(img, cv2.ROTATE_180)
        if self.name == "rot270":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        raise ValueError(self.name)

    def invert_box(self, cx, cy, w, h, ang_deg, orig_w, orig_h):
        """Map a box predicted on the TRANSFORMED image back to ORIGINAL image coords."""
        if self.name == "identity":
            pass
        elif self.name == "hflip":
            cx = orig_w - cx
            ang_deg = normalize_angle_deg(180.0 - ang_deg)
        elif self.name == "vflip":
            cy = orig_h - cy
            ang_deg = normalize_angle_deg(-ang_deg)
        elif self.name == "rot90":
            # forward: (x,y) -> (y, W-x); back = rotate +90 clockwise
            x_p, y_p = cx, cy
            cx, cy = orig_w - y_p, x_p
            w, h = h, w
            ang_deg = normalize_angle_deg(ang_deg - 90.0)
        elif self.name == "rot180":
            cx = orig_w - cx
            cy = orig_h - cy
            ang_deg = normalize_angle_deg(ang_deg - 180.0)
        elif self.name == "rot270":
            # forward: (x,y) -> (H-y, x); back = rotate +90 counter-clockwise
            x_p, y_p = cx, cy
            cx, cy = y_p, orig_h - x_p
            w, h = h, w
            ang_deg = normalize_angle_deg(ang_deg + 90.0)
        return cx, cy, w, h, ang_deg

def get_tta_list(mode: str) -> List[TTATransform]:
    if mode == "none":
        return [TTATransform("identity")]
    if mode == "flips":
        return [TTATransform(n) for n in ["identity", "hflip", "vflip"]]
    if mode == "rot90":
        return [TTATransform(n) for n in ["identity", "rot90", "rot180", "rot270"]]
    if mode == "all":
        # 十分な多様性＆重複少なめ
        return [TTATransform(n) for n in ["identity", "hflip", "vflip", "rot90", "rot270"]]
    raise ValueError(f"Unknown TTA mode: {mode}")

# ---------------------- NMS merge (rotated) ----------------------

def merge_rotated_nms(xywhr_list, confs, classes, iou_thr=0.5):
    """
    ABBで標準nms→回転IoUで仕上げを使う。
    """
    return merge_rotated_nms_fallback(xywhr_list, confs, classes, iou_thr=iou_thr, per_class=True)


# ---------------------- Prediction with TTA ----------------------

def predict_obb_with_tta(model: YOLO, img_bgr: np.ndarray, conf: float, iou: float,
                         imgsz: Optional[int], tta_mode: str, tta_merge_iou: float):
    """
    Run multiple predictions with flip/rotate TTA, map back to original coords, and merge by rotated-NMS.
    Returns: list of polys (4x2), classes, confs
    """
    H0, W0 = img_bgr.shape[:2]
    tta_list = get_tta_list(tta_mode)
    xywhr_all, cls_all, conf_all = [], [], []

    for t in tta_list:
        img_t = t.apply_img(img_bgr)
        # Ultralytics can take np.ndarray directly
        res = model.predict(source=img_t, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        # Prefer OBB
        if getattr(res, "obb", None) is not None and res.obb is not None and res.obb.xywhr is not None:
            xywhr = res.obb.xywhr.cpu().numpy()  # (N,5): cx,cy,w,h,angle(rad)

            # ★ クラスは obb.cls を最優先、無ければ boxes.cls を見る
            if getattr(res.obb, "cls", None) is not None:
                cls_arr = res.obb.cls.cpu().numpy().astype(int)
            elif getattr(res, "boxes", None) is not None and getattr(res.boxes, "cls", None) is not None:
                cls_arr = res.boxes.cls.cpu().numpy().astype(int)
            else:
                cls_arr = np.full(len(xywhr), -1, dtype=int)

            # ★ スコアは obb.conf を最優先、無ければ boxes.conf
            if getattr(res.obb, "conf", None) is not None:
                conf_arr = res.obb.conf.cpu().numpy().astype(float)
            elif getattr(res, "boxes", None) is not None and getattr(res.boxes, "conf", None) is not None:
                conf_arr = res.boxes.conf.cpu().numpy().astype(float)
            else:
                conf_arr = np.zeros(len(xywhr), dtype=float)

            # 長さを安全に揃える（まれに不一致が起きる対策）
            n = len(xywhr)
            if len(cls_arr) != n:
                if len(cls_arr) < n:
                    cls_pad = np.full(n - len(cls_arr), -1, dtype=int)
                    cls_arr = np.concatenate([cls_arr, cls_pad], axis=0)
                else:
                    cls_arr = cls_arr[:n]
            if len(conf_arr) != n:
                if len(conf_arr) < n:
                    conf_pad = np.zeros(n - len(conf_arr), dtype=float)
                    conf_arr = np.concatenate([conf_arr, conf_pad], axis=0)
                else:
                    conf_arr = conf_arr[:n]

            for k in range(n):
                cx, cy, ww, hh, ang = xywhr[k]
                ang_deg = np.degrees(ang) if abs(ang) <= 3.2 else ang
                cx2, cy2, w2, h2, a2 = t.invert_box(cx, cy, ww, hh, ang_deg, W0, H0)
                a2 = normalize_angle_deg(a2)
                xywhr_all.append((float(cx2), float(cy2), float(w2), float(h2), float(a2)))
                cls_all.append(int(cls_arr[k]))
                conf_all.append(float(conf_arr[k]))

        else:
            # Fallback: AABB -> OBB(angle=0)
            if getattr(res, "boxes", None) is not None and res.boxes is not None and res.boxes.xyxy is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.full(len(xyxy), -1)
                confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes.conf is not None else [0.0]*len(xyxy)
                h_t, w_t = img_t.shape[:2]
                for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
                    cx, cy = (x1 + x2)/2, (y1 + y2)/2
                    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
                    # AABBは回転0°で扱い、逆変換だけかける
                    cx2, cy2, w2, h2, a2 = t.invert_box(cx, cy, w, h, 0.0, W0, H0)
                    a2 = normalize_angle_deg(a2)
                    xywhr_all.append((float(cx2), float(cy2), float(w2), float(h2), float(a2)))
                    cls_all.append(int(c))
                    conf_all.append(float(s))

    # Merge by rotated-NMS
    xywhr_merged, conf_merged, cls_merged = merge_rotated_nms(xywhr_all, conf_all, cls_all, iou_thr=tta_merge_iou)

    # Convert to polygons for drawing
    polys = [obb_to_polygon(cx, cy, w, h, ang) for (cx, cy, w, h, ang) in xywhr_merged]
    return polys, cls_merged, conf_merged

# ---------------------- Metrics collection ----------------------

def try_collect_metrics_from_api(val_ret) -> Dict[str, float]:
    out = {}
    try:
        m = getattr(val_ret, "metrics", None) or val_ret
        target = None
        for key in ["obb", "box", "segment"]:
            if hasattr(m, key) and getattr(m, key) is not None:
                target = getattr(m, key)
                break
        if target is not None:
            if hasattr(target, "map50"):
                out["mAP50"] = float(target.map50)
            if hasattr(target, "map"):
                out["mAP50_95"] = float(target.map)
            if hasattr(target, "mp"):
                out["precision"] = float(target.mp)
            if hasattr(target, "mr"):
                out["recall"] = float(target.mr)
            if hasattr(target, "nc"):
                out["nc"] = int(target.nc)
        if hasattr(m, "speed") and hasattr(m.speed, "images"):
            out["n_images"] = int(m.speed.images)
    except Exception:
        pass
    return out

def try_collect_metrics_from_csv(metrics_dir: Path) -> Dict[str, float]:
    out = {}
    csv_path = metrics_dir / "results.csv"
    if not csv_path.exists():
        cands = list(metrics_dir.glob("**/results.csv"))
        if cands:
            csv_path = cands[0]
        else:
            return out
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f); rows = list(reader)
    if not rows: return out
    row = rows[-1]
    key_map = {
        "mAP50": ["metrics/mAP50(B)", "map50", "mAP50", "metrics/mAP50"],
        "mAP50_95": ["metrics/mAP50-95(B)", "map", "mAP50_95", "metrics/mAP50-95"],
        "precision": ["metrics/precision(B)", "precision", "mp"],
        "recall": ["metrics/recall(B)", "recall", "mr"],
    }
    for k, cand_keys in key_map.items():
        for ck in cand_keys:
            if ck in row and row[ck] not in (None, "", "nan"):
                try:
                    out[k] = float(row[ck]); break
                except Exception:
                    continue
    return out

def copy_if_exists(src: Path, dst_dir: Path):
    if src.exists():
        ensure_dir(dst_dir)
        shutil.copy2(src, dst_dir / src.name)

def harvest_val_artifacts(val_proj_dir: Path, dest_dir: Path):
    ensure_dir(dest_dir)
    candidates = [
        "PR_curve.png",
        "PR_curve_classes.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.csv",
        "labels_correlogram.jpg",
        "metrics.json",
    ]
    for fn in candidates:
        for p in val_proj_dir.rglob(fn):
            copy_if_exists(p, dest_dir)

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", '-d', required=True, help="data.yaml (must contain the split to visualize)")
    ap.add_argument("--weights", '-w', required=True, help="path to best.pt")
    ap.add_argument("--out", '-o', required=True, help="output directory")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"], help="which split")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for predictions")
    ap.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold (per-call)")
    ap.add_argument("--angle_unit", choices=["deg", "rad"], default="deg", help="GT angle unit")
    ap.add_argument("--thick", type=int, default=4, help="line thickness for overlay")
    ap.add_argument("--max_images", type=int, default=0, help="limit number of visualized images (0=all)")
    ap.add_argument("--device", default=None, help="device for Ultralytics (e.g., 0, '0,1', 'cpu')")
    ap.add_argument("--skip_vis", action="store_true", help="skip overlays, compute metrics only")
    # --- TTA options ---
    ap.add_argument("--tta", default="none", choices=["none","flips","rot90","all"], help="flip/rotate test-time augmentation")
    ap.add_argument("--tta_merge_iou", type=float, default=0.50, help="merge IoU for rotated-NMS across TTA views")
    ap.add_argument("--cfg", type=str, default=None, help="YAML file containing train section (with imgsz)")
    ap.add_argument("--map_iou_range", default="0.50:0.95:0.05",
                help="IoU range for mAP (start:end:step), e.g., 0.50:0.95:0.05")
    args = ap.parse_args()

    # mAP用IoUしきい値列を作成
    try:
        s, e, st = map(float, args.map_iou_range.split(":"))
        map_iou_thrs = [round(x, 2) for x in np.arange(s, e + 1e-9, st)]
    except Exception:
        print(f"[WARN] Invalid --map_iou_range '{args.map_iou_range}', fallback to 0.50")
        map_iou_thrs = [0.50]

    # ---------------- YAML読み込み ----------------
    data_yaml = Path(args.data).resolve()
    cfg = load_yaml(data_yaml)

    train_yaml = Path(args.cfg).resolve()
    train_cfg = load_yaml(train_yaml)
    try:
        imgsz = int(train_cfg.get("train", {}).get("imgsz", None))
        if imgsz is None:
            raise KeyError("train.imgsz not found")
        print(f"[INFO] Using imgsz={imgsz} from {train_yaml}")
    except Exception as e:
        raise RuntimeError(f"Failed to read train.imgsz from {train_yaml}: {e}")

    split_key = args.split
    assert split_key in cfg, f"'{split_key}' not found in {data_yaml}"
    test_entry = cfg[split_key]

    # class names (optional)
    class_names = None
    if "names" in cfg:
        names = cfg["names"]
        if isinstance(names, dict):
            class_names = {int(k): v for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            class_names = {i: n for i, n in enumerate(names)}

    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)
    metrics_dir = out_dir / "metrics"
    ensure_dir(metrics_dir)

    # Colors (BGR): GT=cyan, Pred=magenta
    C_GT = (255, 255, 0)
    C_PR = (255, 0, 255)
    thick = int(args.thick)

    # Load model
    model = YOLO(args.weights)
    if args.device is not None:
        try:
            model.to(args.device)
        except Exception:
            pass  # Ultralytics handles device generally via predict/val args

    # ---------- Overlays ----------
    if not args.skip_vis:
        img_paths = find_images(test_entry)
        if args.max_images and args.max_images > 0:
            img_paths = img_paths[: args.max_images]
        assert img_paths, f"No images found under '{test_entry}'"
        tta_records = []  # 各画像の GT/PRED を蓄積して後でmAP計算

        for i, img_path in enumerate(img_paths, 1):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            H, W = img.shape[:2]
            canvas = as_bgr(img.copy())

            # GT
            gt_items = []
            lbl_path = map_image_to_label(img_path)
            if lbl_path:
                for poly, cls in load_gt_polys_auto(lbl_path, W, H):
                    txt = f"GT:{class_names.get(cls, str(cls))}" if class_names else None
                    draw_poly(canvas, poly, C_GT, thick, label=txt)
                    gt_items.append({"cls": int(cls), "poly": poly.astype(np.float32)})

            # Predictions with TTA merge
            polys, pred_classes, pred_confs = predict_obb_with_tta(
                model=model, img_bgr=as_bgr(img), conf=args.conf, iou=args.iou,
                imgsz=imgsz, tta_mode=args.tta, tta_merge_iou=args.tta_merge_iou
            )

            # Draw
            for poly, cls, conf in zip(polys, pred_classes, pred_confs):
                if class_names is not None and cls is not None and cls >= 0:
                    name = class_names.get(int(cls), str(int(cls)))
                    txt = f"PR:{name}"
                else:
                    txt = "PR"
                if conf is not None:
                    txt = f"{txt} {conf:.2f}"
                draw_poly(canvas, poly, C_PR, thick, label=txt)

            # legend
            y0 = 30
            cv2.putText(canvas, "GT", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GT, 2, cv2.LINE_AA)
            cv2.line(canvas, (60, y0-8), (120, y0-8), C_GT, thick, cv2.LINE_AA)
            cv2.putText(canvas, "Pred (TTA)", (140, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_PR, 2, cv2.LINE_AA)
            cv2.line(canvas, (270, y0-8), (350, y0-8), C_PR, thick, cv2.LINE_AA)

            out_path = out_dir / f"{img_path.stem}_gt_pred.png"
            cv2.imwrite(str(out_path), canvas)
            print(f"[{i}/{len(img_paths)}] saved -> {out_path}")

            pred_items = []
            for poly, cls, conf in zip(polys, pred_classes, pred_confs):
                if cls is None or cls < 0: 
                    continue
                pred_items.append({"cls": int(cls), "conf": float(conf), "poly": poly.astype(np.float32)})

            tta_records.append({
                "image_id": str(img_path),
                "gt": gt_items,
                "pred": pred_items,
            })
        print("\n[TTA EVAL] Computing TTA-based metrics (mAP@0.50 and mAP@0.50:0.95) ...")
        tta_metrics = eval_tta_detections(tta_records, class_names, iou_thrs=map_iou_thrs)

        # 保存
        tta_json = metrics_dir / "metrics_tta_summary.json"
        with open(tta_json, "w", encoding="utf-8") as f:
            json.dump(tta_metrics, f, indent=2, ensure_ascii=False)

        # CSV（トップラインのみ）
        tta_csv = metrics_dir / "metrics_tta_summary.csv"
        with open(tta_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["mAP50", tta_metrics["mAP50"]])
            w.writerow(["mAP50_95", tta_metrics["mAP50_95"]])
            w.writerow(["num_images", tta_metrics["num_images"]])
            w.writerow(["num_classes", tta_metrics["num_classes"]])

        print(f"[TTA EVAL] mAP50={tta_metrics['mAP50']:.4f}  mAP50_95={tta_metrics['mAP50_95']:.4f}")
        print(f"[TTA EVAL] Saved: {tta_json}")

    # ---------- Metrics via Ultralytics val ----------
    print("\n[VAL] Running Ultralytics validation to compute metrics (no TTA) ...")
    val_ret = model.val(
        data=str(data_yaml),
        split=split_key,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        plots=True,
        project=str(metrics_dir),
        name=".",
        imgsz=imgsz,
        verbose=False,
    )

    metrics = try_collect_metrics_from_api(val_ret) or try_collect_metrics_from_csv(metrics_dir)

    sum_csv = metrics_dir / "metrics_summary.csv"
    sum_json = metrics_dir / "metrics_summary.json"
    if metrics:
        with open(sum_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["metric", "value"])
            for k, v in metrics.items():
                w.writerow([k, v])
        with open(sum_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[VAL] Summary saved: {sum_csv}")
    else:
        print("[VAL] Warning: could not parse metrics automatically (check Ultralytics CSV/JSON).")

    harvest_val_artifacts(metrics_dir, metrics_dir)
    print("[DONE] Visuals in:", out_dir)
    print("[DONE] Metrics in:", metrics_dir)

if __name__ == "__main__":
    main()
