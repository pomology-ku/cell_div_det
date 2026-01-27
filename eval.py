#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay GT and inference OBBs on images and save, with optional TTA (flip/rotate) and rotated-NMS merge.
Refactored to import TTA/NMS utilities from tta_obb.py
"""
import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import json
import csv
import shutil
from typing import List, Tuple, Dict, Optional

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from tta_obb import (
    predict_obb_with_tta,  # (model, img_bgr, conf, iou, imgsz, tta_mode, tta_merge_iou) -> (polys, classes, confs, [xywhr])
    obb_to_polygon,
    poly_iou,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------- IO helpers ----------------------

def parse_conf_list(conf_str: str) -> list[float]:
    if conf_str is None:
        return [0.25]
    if isinstance(conf_str, (int, float)):
        return [float(conf_str)]
    parts = [p.strip() for p in str(conf_str).split(",") if p.strip()]
    if not parts:
        return [0.25]
    confs = []
    for p in parts:
        confs.append(float(p))
    return confs

def extract_recall_overall(metrics: dict) -> Optional[float]:
    for k, v in metrics.items():
        if k.startswith("recall_overall"):
            try:
                return float(v)
            except Exception:
                return None
    return None

def print_conf_summary(rows: list[dict]):
    headers = [
        "conf",
        "tta_mAP50",
        "tta_mAP50_95",
        "tta_recall",
        "val_mAP50",
        "val_mAP50_95",
        "val_precision",
        "val_recall",
    ]
    def _fmt(v: Optional[float]) -> str:
        return "-" if v is None else f"{v:.4f}"

    str_rows = []
    for r in rows:
        str_rows.append({
            "conf": f"{float(r['conf']):.4f}",
            "tta_mAP50": _fmt(r.get("tta_mAP50")),
            "tta_mAP50_95": _fmt(r.get("tta_mAP50_95")),
            "tta_recall": _fmt(r.get("tta_recall")),
            "val_mAP50": _fmt(r.get("val_mAP50")),
            "val_mAP50_95": _fmt(r.get("val_mAP50_95")),
            "val_precision": _fmt(r.get("val_precision")),
            "val_recall": _fmt(r.get("val_recall")),
        })

    widths = {h: len(h) for h in headers}
    for r in str_rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r[h])))

    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "  ".join("-" * widths[h] for h in headers)
    print("\n[CONF SUMMARY]")
    print(header_line)
    print(sep_line)
    for r in str_rows:
        line = "  ".join(str(r[h]).ljust(widths[h]) for h in headers)
        print(line)

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

# ---------------------- AABB helpers (for optional AABB IoU eval) ----------------------

def poly_to_aabb(poly: np.ndarray) -> tuple[float,float,float,float]:
    x1, y1 = np.min(poly, axis=0)
    x2, y2 = np.max(poly, axis=0)
    return float(x1), float(y1), float(x2), float(y2)

def iou_aabb(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (b1[2]-b1[0])) * max(0.0, (b1[3]-b1[1]))
    a2 = max(0.0, (b2[2]-b2[0])) * max(0.0, (b2[3]-b2[1]))
    return float(inter / max(a1 + a2 - inter, 1e-9))

# ---------------------- Metrics core ----------------------

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

def eval_tta_detections(
    records,
    class_names: dict | None,
    iou_thrs: list[float],
    iou_mode: str = "obb",
    recall_iou: float = 0.50,
    conf_thresh: float = 0.25,
):
    """
    Evaluate mAP and Recall at given IoU/conf thresholds.
    records: [{'image_id':..., 'gt':[{'cls','poly'}], 'pred':[{'cls','conf','poly'}]}]
    iou_thrs: e.g. [0.50,0.55,...,0.95]
    iou_mode: 'obb' or 'aabb'
    recall_iou: IoU threshold for recall counting (default=0.50)
    conf_thresh: confidence threshold to count a detection as valid (default=0.25)
    """
    if iou_mode == "aabb":
        def _iou(p1, p2):
            return iou_aabb(poly_to_aabb(p1), poly_to_aabb(p2))
    else:
        def _iou(p1, p2):
            return poly_iou(p1, p2)

    # ---- クラス一覧 ----
    if class_names:
        cls_ids = sorted(class_names.keys())
    else:
        s = set()
        for r in records:
            for g in r['gt']:   s.add(int(g['cls']))
            for p in r['pred']: s.add(int(p['cls']))
        cls_ids = sorted(int(c) for c in s if c is not None and c >= 0)

    # ---- GT索引 ----
    gt_by_img_cls = {}
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

    aps_50, aps_range = {}, {}
    recall_hits_by_cls = {cid: 0 for cid in cls_ids}

    # ---- 各クラスごとに評価 ----
    for cid in cls_ids:
        preds = []
        for r in records:
            img_id = r['image_id']
            for p in r['pred']:
                if int(p['cls']) == cid:
                    preds.append((img_id, float(p['conf']), p['poly']))
        preds.sort(key=lambda x: -x[1])

        ap_per_thr = []
        # ===== mAP =====
        for thr in iou_thrs:
            tp = np.zeros(len(preds), dtype=np.float32)
            fp = np.zeros(len(preds), dtype=np.float32)
            for key, arr in gt_by_img_cls.items():
                if key[1] == cid:
                    for it in arr:
                        it['matched'] = False

            for i, (img_id, conf, ppoly) in enumerate(preds):
                gts = gt_by_img_cls.get((img_id, cid), [])
                iou_max = 0.0; jmax = -1
                for j, g in enumerate(gts):
                    if g['matched']:
                        continue
                    iou = _iou(ppoly, g['poly'])
                    if iou > iou_max:
                        iou_max, jmax = iou, j
                if iou_max >= thr and jmax >= 0:
                    tp[i] = 1.0
                    gts[jmax]['matched'] = True
                else:
                    fp[i] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            denom = np.maximum(cum_tp + cum_fp, 1e-12)
            prec = cum_tp / denom
            rec  = cum_tp / max(npos_by_cls[cid], 1e-12)
            ap = _compute_ap(rec, prec) if npos_by_cls[cid] > 0 else 0.0
            ap_per_thr.append(ap)
            if abs(thr - 0.50) < 1e-9:
                aps_50[cid] = ap
        aps_range[cid] = float(np.mean(ap_per_thr)) if ap_per_thr else 0.0

        # ===== Recall(conf ≥ conf_thresh & IoU ≥ recall_iou) =====
        for key, arr in gt_by_img_cls.items():
            if key[1] == cid:
                for it in arr:
                    it['matched'] = False
        matched_cnt = 0
        for img_id, conf, ppoly in preds:
            if conf < conf_thresh:
                continue  # 低信頼予測は除外
            gts = gt_by_img_cls.get((img_id, cid), [])
            iou_max, jmax = 0.0, -1
            for j, g in enumerate(gts):
                if g['matched']:
                    continue
                iou = _iou(ppoly, g['poly'])
                if iou > iou_max:
                    iou_max, jmax = iou, j
            if iou_max >= recall_iou and jmax >= 0:
                gts[jmax]['matched'] = True
                matched_cnt += 1
        recall_hits_by_cls[cid] = matched_cnt

    # ---- 集計 ----
    mAP50 = float(np.mean(list(aps_50.values()))) if aps_50 else 0.0
    mAP50_95 = float(np.mean(list(aps_range.values()))) if aps_range else 0.0
    total_pos = sum(npos_by_cls.values())
    total_hits = sum(recall_hits_by_cls.values())
    recall_overall = float(total_hits / max(total_pos, 1e-12)) if total_pos > 0 else 0.0

    per_class = {}
    for cid in cls_ids:
        name = class_names.get(cid, str(cid)) if class_names else str(cid)
        rc = float(recall_hits_by_cls[cid] / max(npos_by_cls[cid], 1e-12)) if npos_by_cls[cid] > 0 else 0.0
        per_class[name] = {
            "AP50": float(aps_50.get(cid, 0.0)),
            "AP50_95": float(aps_range.get(cid, 0.0)),
            f"Recall(conf>={conf_thresh})": rc,
        }

    return {
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        f"recall_overall(conf>={conf_thresh})": recall_overall,
        "per_class": per_class,
        "num_images": len(records),
        "num_classes": len(cls_ids),
    }

# ---------------------- GT loader & drawing ----------------------

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

            # YOLO OBB (cls cx cy w h ang[rad or deg])
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

            # Polygon (cls x1 y1 x2 y2 x3 y3 x4 y4) possibly in [0,1] or abs
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

# ---------------------- Ultralytics metrics harvest ----------------------

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

# ---------------------- SAHI sliced inference for OBB ----------------------
def predict_obb_with_sahi_if_enabled(
    use_sahi: bool,
    weights_path: str,
    img_bgr: np.ndarray,
    conf: float,
    device: Optional[str],
    slice_w: int,
    slice_h: int,
    ovw: float,
    ovh: float,
    postprocess_type: str,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """
    SAHIのスライス推論でOBB検出を実行し、(poly(4x2), cls, conf) を返す。
    SAHIが無効 or import不可のときは (None, None, None) を返す。
    """
    if not use_sahi or AutoDetectionModel is None or get_sliced_prediction is None:
        return None, None, None

    # SAHIモデルの準備（Ultralytics OBBに対応）
    try:
        det_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_path,          # *.pt パス
            confidence_threshold=conf,
            device=device if device is not None else None,  # "0" / "0,1" でもOK
        )
    except Exception as e:
        print(f"[WARN] SAHI setup failed: {e}")
        return None, None, None

    try:
        # SAHI スライス推論
        result = get_sliced_prediction(
            img_bgr,
            det_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=ovh,
            overlap_width_ratio=ovw,
            postprocess_type=postprocess_type,   # "NMS" 推奨
            # perform_standard_pred=False,  # 明示したい場合は有効化
        )
    except Exception as e:
        print(f"[WARN] SAHI prediction failed: {e}")
        return None, None, None

    polys, classes, confs = [], [], []

    # --- できるだけ堅牢に、多角形(4点)を取り出す ---
    # SAHIはCOCO形式エクスポートができるので、まずはそこから polygon を優先取得
    try:
        # segmentation が OBBの4点になる実装に追従（無ければ後続のフォールバックで処理）
        annos = result.to_coco_annotations()
    except Exception:
        annos = []

    used = 0
    for a in annos:
        seg = a.get("segmentation", None)
        if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, tuple)) and len(seg[0]) >= 8:
            flat = seg[0][:8]  # 4点分
            poly = np.array(flat, dtype=np.float32).reshape(4, 2)
            polys.append(poly)
            classes.append(int(a.get("category_id", 0)))
            confs.append(float(a.get("score", a.get("confidence", conf))))
            used += 1

    # COCO経由で拾えなかった場合は、object_prediction_list から OBB を推測して回収
    # （将来のSAHI変更に備えて複数の属性名を試す）
    if used == 0:
        opl = getattr(result, "object_prediction_list", []) or []
        for op in opl:
            # クラスID / スコア（無ければ推測）
            try:
                c = getattr(op.category, "id", None)
                if c is None:
                    c = getattr(op.category, "category_id", 0)
                cid = int(c)
            except Exception:
                cid = 0

            try:
                sc = getattr(op.score, "value", None)
                confv = float(sc) if sc is not None else float(conf)
            except Exception:
                confv = float(conf)

            # OBBの4点を取得（属性の揺れに対応）
            poly = None
            # 1) 典型: op.obb / op.rotated_bbox / op.quad / op.points 等
            for attr in ["obb", "rotated_bbox", "oriented_bbox", "quad", "points", "polygon"]:
                obj = getattr(op, attr, None)
                if obj is None:
                    continue
                # list/tupleで座標が入っているケース
                if isinstance(obj, (list, tuple, np.ndarray)):
                    arr = np.array(obj, dtype=np.float32).reshape(-1)
                    if arr.size >= 8:
                        poly = arr[:8].reshape(4, 2)
                        break
                # オブジェクトに to_xyxyxyxy / to_points などのメソッドがあるケース
                for meth in ["to_xyxyxyxy", "to_points", "to_polygon", "to_list"]:
                    fn = getattr(obj, meth, None)
                    if callable(fn):
                        arr = np.array(fn(), dtype=np.float32).reshape(-1)
                        if arr.size >= 8:
                            poly = arr[:8].reshape(4, 2)
                            break
                if poly is not None:
                    break

            # 2) 最後の手段: AABBしか無い場合はAABB→四角形に変換
            if poly is None:
                try:
                    bb = getattr(op.bbox, "to_xyxy", None)
                    bb = bb() if callable(bb) else bb
                except Exception:
                    bb = None
                if bb is None:
                    # bbox dict/tuple などから推測
                    bb = getattr(op, "bbox", None)
                try:
                    if bb is not None:
                        bb = np.array(bb, dtype=np.float32).reshape(-1)
                        if bb.size >= 4:
                            x1, y1, x2, y2 = bb[:4]
                            poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
                except Exception:
                    poly = None

            if poly is not None and poly.shape == (4, 2):
                polys.append(poly)
                classes.append(cid)
                confs.append(confv)

    return polys, classes, confs

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", '-d', required=True, help="data.yaml (must contain the split to visualize)")
    ap.add_argument("--weights", '-w', required=True, help="path to best.pt")
    ap.add_argument("--out", '-o', required=True, help="output directory")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"], help="which split")
    ap.add_argument("--conf", type=str, default="0.25", help="confidence threshold(s) for predictions (comma-separated)")
    ap.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold (per-call)")
    ap.add_argument("--angle_unit", choices=["deg", "rad"], default="deg", help="GT angle unit")
    ap.add_argument("--thick", type=int, default=4, help="line thickness for overlay")
    ap.add_argument("--max_images", type=int, default=0, help="limit number of visualized images (0=all)")
    ap.add_argument("--device", default=None, help="device for Ultralytics (e.g., 0, '0,1', 'cpu')")
    ap.add_argument("--skip_vis", action="store_true", help="skip overlays, compute metrics only")
    # --- TTA options ---
    ap.add_argument("--tta", default="none", choices=["none","flips","rot90","all"], help="flip/rotate test-time augmentation")
    ap.add_argument("--tta_merge_iou", type=float, default=0.50, help="merge IoU for rotated-NMS across TTA views")
    # --- SAHI slice options ---
    ap.add_argument("--use_sahi_slice", action="store_true",
                help="Use SAHI sliced inference when --tta=none")
    ap.add_argument("--slice_width",  type=int, default=512)
    ap.add_argument("--slice_height", type=int, default=512)
    ap.add_argument("--overlap_width_ratio",  type=float, default=0.20)
    ap.add_argument("--overlap_height_ratio", type=float, default=0.20)
    ap.add_argument("--sahi_postprocess", choices=["NMS","UNION","NONE"], default="NMS",
                    help="SAHI postprocess type (NMS recommended)")
    
    ap.add_argument("--cfg", type=str, default=None, help="YAML file containing train section (with imgsz)")
    ap.add_argument("--map_iou_range", default="0.50:0.95:0.05",
                help="IoU range for mAP (start:end:step), e.g., 0.50:0.95:0.05")
    ap.add_argument("--eval_iou_mode", default="obb", choices=["obb","aabb"],
                help="IoU type for TTA metrics (obb=rotated polygon IoU, aabb=axis-aligned IoU)")
    ap.add_argument("--recall_iou", type=float, default=0.50, help="IoU threshold for recall calculation")
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

    if args.cfg is None:
        raise RuntimeError("--cfg is required to obtain train.imgsz")
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

    conf_list = parse_conf_list(args.conf)
    summaries = []

    def run_eval_for_conf(conf: float) -> dict:
        summary = {
            "conf": conf,
            "tta_mAP50": None,
            "tta_mAP50_95": None,
            "tta_recall": None,
            "val_mAP50": None,
            "val_mAP50_95": None,
            "val_precision": None,
            "val_recall": None,
        }

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

                # Predictions
                if args.tta != "none":
                    # 既存のTTAマージ
                    polys, pred_classes, pred_confs, _xywhr = predict_obb_with_tta(
                        model=model, img_bgr=as_bgr(img), conf=conf, iou=args.iou,
                        imgsz=imgsz, tta_mode=args.tta, tta_merge_iou=args.tta_merge_iou
                    )
                    pred_legend = "Pred (TTA)"
                else:
                    # TTAがnoneのときだけ SAHI のスライス推論を試す
                    sahi_polys, sahi_classes, sahi_confs = predict_obb_with_sahi_if_enabled(
                        use_sahi=args.use_sahi_slice,
                        weights_path=args.weights,
                        img_bgr=as_bgr(img),
                        conf=conf,
                        device=args.device,
                        slice_w=args.slice_width,
                        slice_h=args.slice_height,
                        ovw=args.overlap_width_ratio,
                        ovh=args.overlap_height_ratio,
                        postprocess_type=args.sahi_postprocess,
                    )

                    if sahi_polys is not None:
                        polys, pred_classes, pred_confs = sahi_polys, sahi_classes, sahi_confs
                        _xywhr = None
                        pred_legend = "Pred (SAHI)"
                    else:
                        # SAHIが無効/失敗 → 通常の単画像推論（TTAなし）
                        res = model.predict(
                            source=as_bgr(img),
                            conf=conf,
                            iou=args.iou,
                            imgsz=imgsz,
                            verbose=False,
                            device=args.device if args.device is not None else None,
                        )
                        # Ultralytics OBB → polygon へ（あなたの tta_obb.py の obb_to_polygon 等が流用できるならそれでもOK）
                        polys, pred_classes, pred_confs = [], [], []
                        for r in (res or []):
                            # r.obb または r.boxes.rbox など、環境によって保持場所が異なるのでいくつか試す
                            obb = getattr(r, "obb", None) or getattr(getattr(r, "boxes", None), "rbox", None)
                            if obb is not None and hasattr(obb, "xywhr"):
                                xywhr = obb.xywhr.cpu().numpy()
                                confs = getattr(obb, "conf", getattr(getattr(r, "boxes", None), "conf", None))
                                confs = confs.cpu().numpy() if confs is not None else np.ones(len(xywhr), dtype=np.float32)
                                clsids = getattr(obb, "cls", getattr(getattr(r, "boxes", None), "cls", None))
                                clsids = clsids.cpu().numpy().astype(int) if clsids is not None else np.zeros(len(xywhr), dtype=int)
                                for (cx,cy,w,h,ang), c, sc in zip(xywhr, clsids, confs):
                                    poly = obb_to_polygon(float(cx), float(cy), float(w), float(h), float(np.degrees(ang)))
                                    polys.append(poly.astype(np.float32))
                                    pred_classes.append(int(c))
                                    pred_confs.append(float(sc))
                            else:
                                # AABBのみ → AABBを四角形に
                                boxes = getattr(r, "boxes", None)
                                if boxes is None:
                                    continue
                                xyxy = getattr(boxes, "xyxy", None)
                                clsids = getattr(boxes, "cls", None)
                                confs = getattr(boxes, "conf", None)
                                if xyxy is None:
                                    continue
                                xyxy = xyxy.cpu().numpy()
                                clsids = clsids.cpu().numpy().astype(int) if clsids is not None else np.zeros(len(xyxy), dtype=int)
                                confs = confs.cpu().numpy() if confs is not None else np.ones(len(xyxy), dtype=np.float32)
                                for (x1,y1,x2,y2), c, sc in zip(xyxy, clsids, confs):
                                    poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
                                    polys.append(poly)
                                    pred_classes.append(int(c))
                                    pred_confs.append(float(sc))
                        pred_legend = "Pred"

                # Draw
                for poly, cls, pred_conf in zip(polys, pred_classes, pred_confs):
                    if class_names is not None and cls is not None and cls >= 0:
                        name = class_names.get(int(cls), str(int(cls)))
                        txt = f"PR:{name}"
                    else:
                        txt = "PR"
                    if pred_conf is not None:
                        txt = f"{txt} {pred_conf:.2f}"
                    draw_poly(canvas, poly, C_PR, thick, label=txt)

                # legend
                y0 = 30
                cv2.putText(canvas, "GT", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GT, 2, cv2.LINE_AA)
                cv2.line(canvas, (60, y0-8), (120, y0-8), C_GT, thick, cv2.LINE_AA)
                cv2.putText(canvas, pred_legend, (140, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_PR, 2, cv2.LINE_AA)
                cv2.line(canvas, (270, y0-8), (430, y0-8), C_PR, thick, cv2.LINE_AA)

                out_path = out_dir / f"{img_path.stem}_gt_pred.png"
                cv2.imwrite(str(out_path), canvas)
                print(f"[{i}/{len(img_paths)}] saved -> {out_path}")

                pred_items = []
                for poly, cls, pred_conf in zip(polys, pred_classes, pred_confs):
                    if cls is None or cls < 0:
                        continue
                    pred_items.append({"cls": int(cls), "conf": float(pred_conf), "poly": poly.astype(np.float32)})

                tta_records.append({
                    "image_id": str(img_path),
                    "gt": gt_items,
                    "pred": pred_items,
                })

            print("\n[TTA EVAL] Computing TTA-based metrics (mAP@0.50 and mAP@0.50:0.95) ...")
            tta_metrics = eval_tta_detections(
                tta_records,
                class_names,
                iou_thrs=map_iou_thrs,
                iou_mode=args.eval_iou_mode,
                conf_thresh=conf,
                recall_iou=args.recall_iou,
            )
            summary["tta_mAP50"] = tta_metrics.get("mAP50")
            summary["tta_mAP50_95"] = tta_metrics.get("mAP50_95")
            summary["tta_recall"] = extract_recall_overall(tta_metrics)

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
                for k in tta_metrics.keys():
                    if k.startswith("recall_overall"):
                        w.writerow([k, tta_metrics[k]])

            print(f"[TTA EVAL] mAP50={tta_metrics['mAP50']:.4f}  mAP50_95={tta_metrics['mAP50_95']:.4f}")
            print(f"[TTA EVAL] Saved: {tta_json}")
            for k in tta_metrics.keys():
                if k.startswith("recall_overall"):
                    print(f"[TTA EVAL] {k}={tta_metrics[k]:.4f}")

        # ---------- Metrics via Ultralytics val ----------
        print("\n[VAL] Running Ultralytics validation to compute metrics (no TTA) ...")
        val_ret = model.val(
            data=str(data_yaml),
            split=split_key,
            conf=conf,
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

        if metrics:
            summary["val_mAP50"] = metrics.get("mAP50")
            summary["val_mAP50_95"] = metrics.get("mAP50_95")
            summary["val_precision"] = metrics.get("precision")
            summary["val_recall"] = metrics.get("recall")

        return summary

    for conf in conf_list:
        if len(conf_list) > 1:
            print(f"\n[CONF] conf={conf}")
        summaries.append(run_eval_for_conf(conf))

    if len(conf_list) > 1:
        print_conf_summary(summaries)

if __name__ == "__main__":
    main()
