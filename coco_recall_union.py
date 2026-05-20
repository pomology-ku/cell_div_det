#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute recall from COCO JSONs exported by eval.py (tta_gt_conf*.coco.json and tta_pred_conf*.coco.json).

- GT input: COCO dataset-style JSON with images/annotations/categories (exported by eval.py).
- Prediction input: COCO results-style JSON (list of detections), potentially from two different models/methods.
- Output: recall for each prediction set separately and for their union (concatenated predictions).

Recall definition (matches eval.py / eval_tta_detections):
- consider predictions with score >= conf_thresh
- a GT instance is matched if an unmatched prediction of the same class in the same image achieves IoU >= recall_iou
- greedy matching by descending prediction score (1-to-1)

IoU modes:
- obb: polygon IoU (requires tta_obb.poly_iou)
- aabb: axis-aligned bbox IoU derived from the polygon
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from tta_obb import poly_iou


def poly_to_aabb(poly: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1 = np.min(poly, axis=0)
    x2, y2 = np.max(poly, axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def iou_aabb(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
    return float(inter / max(a1 + a2 - inter, 1e-9))


def coco_det_to_poly(det: dict) -> Optional[np.ndarray]:
    seg = det.get("segmentation", None)
    # eval.py exports segmentation as [[x1,y1,x2,y2,x3,y3,x4,y4]]
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, tuple)) and len(seg[0]) >= 8:
        flat = seg[0][:8]
        return np.array(flat, dtype=np.float32).reshape(4, 2)
    # fallback: bbox -> rectangle polygon
    bb = det.get("bbox", None)
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        x, y, w, h = map(float, bb[:4])
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    return None


def load_gt_coco(gt_json: Path):
    with open(gt_json, "r", encoding="utf-8") as f:
        gt = json.load(f)
    images = gt.get("images", []) or []
    anns = gt.get("annotations", []) or []
    cats = gt.get("categories", []) or []
    cat_name = {int(c["id"]): str(c.get("name", c["id"])) for c in cats if "id" in c}
    img_ids = set(int(im["id"]) for im in images if "id" in im)
    id_to_file = {
        int(im["id"]): str(im.get("file_name", ""))
        for im in images
        if "id" in im and im.get("file_name", "") != ""
    }
    file_to_id = {v: k for k, v in id_to_file.items()}
    return img_ids, anns, cat_name, id_to_file, file_to_id


def load_pred_coco(pred_json: Path) -> List[dict]:
    with open(pred_json, "r", encoding="utf-8") as f:
        dets = json.load(f)
    if not isinstance(dets, list):
        raise ValueError(f"Prediction JSON must be a list (COCO results format): {pred_json}")
    return dets


def build_unique_basename_map(file_to_id: Dict[str, int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for fname in file_to_id.keys():
        base = Path(fname).name
        counts[base] = counts.get(base, 0) + 1
    return {
        Path(fname).name: img_id
        for fname, img_id in file_to_id.items()
        if counts.get(Path(fname).name, 0) == 1
    }


def remap_dets_to_canonical_gt(
    dets: List[dict],
    src_gt_json: Optional[Path],
    canonical_file_to_id: Dict[str, int],
) -> List[dict]:
    """Map detections from another eval.py export onto the canonical GT image_id space."""
    if src_gt_json is None:
        return dets

    _, _, _, src_id_to_file, _ = load_gt_coco(src_gt_json)
    canonical_basename_to_id = build_unique_basename_map(canonical_file_to_id)
    remapped = []
    dropped = 0

    for d in dets:
        try:
            src_img_id = int(d["image_id"])
        except Exception:
            dropped += 1
            continue
        fname = src_id_to_file.get(src_img_id)
        if fname is None:
            dropped += 1
            continue

        new_img_id = canonical_file_to_id.get(fname)
        if new_img_id is None:
            new_img_id = canonical_basename_to_id.get(Path(fname).name)
        if new_img_id is None:
            dropped += 1
            continue

        dd = dict(d)
        dd["image_id"] = int(new_img_id)
        remapped.append(dd)

    if dropped:
        print(f"[WARN] remap from {src_gt_json}: dropped {dropped} detections with no canonical image match.")
    return remapped


def build_gt_signature(
    gt_img_ids: set,
    gt_anns: list,
    id_to_file: Dict[int, str],
) -> Dict[Tuple[str, int], int]:
    sig: Dict[Tuple[str, int], int] = {}
    for a in gt_anns:
        try:
            img_id = int(a["image_id"])
            cid = int(a["category_id"])
        except Exception:
            continue
        if gt_img_ids and img_id not in gt_img_ids:
            continue
        fname = id_to_file.get(img_id, str(img_id))
        sig[(fname, cid)] = sig.get((fname, cid), 0) + 1
    return sig


def print_gt_diff(
    canonical_gt_json: Path,
    other_gt_json: Path,
    canonical_id_to_file: Dict[int, str],
    canonical_img_ids: set,
    canonical_anns: list,
):
    other_img_ids, other_anns, _, other_id_to_file, _ = load_gt_coco(other_gt_json)
    sig_a = build_gt_signature(canonical_img_ids, canonical_anns, canonical_id_to_file)
    sig_b = build_gt_signature(other_img_ids, other_anns, other_id_to_file)
    keys = sorted(set(sig_a.keys()) | set(sig_b.keys()))
    diffs = [(k, sig_a.get(k, 0), sig_b.get(k, 0)) for k in keys if sig_a.get(k, 0) != sig_b.get(k, 0)]

    if not diffs:
        print(f"[GT CHECK] {canonical_gt_json.name} and {other_gt_json.name}: same GT counts by file_name/category_id.")
        return

    print(f"[GT CHECK] {canonical_gt_json.name} vs {other_gt_json.name}: {len(diffs)} file/category count differences.")
    for (fname, cid), ca, cb in diffs[:20]:
        print(f"  category={cid}  canonical={ca}  other={cb}  file={fname}")
    if len(diffs) > 20:
        print(f"  ... {len(diffs) - 20} more differences")


def compute_recall_from_coco(
    gt_img_ids: set,
    gt_anns: list,
    pred_dets: list,
    conf_thresh: float,
    recall_iou: float,
    iou_mode: str,
    class_name_map: Optional[Dict[int, str]] = None,
):
    if iou_mode == "aabb":
        def _iou(p1, p2):
            return iou_aabb(poly_to_aabb(p1), poly_to_aabb(p2))
    else:
        def _iou(p1, p2):
            return poly_iou(p1, p2)

    # GT index
    gt_by_img_cls: Dict[Tuple[int, int], List[dict]] = {}
    npos_by_cls: Dict[int, int] = {}
    for a in gt_anns:
        try:
            img_id = int(a["image_id"])
            cid = int(a["category_id"])
        except Exception:
            continue
        if gt_img_ids and img_id not in gt_img_ids:
            continue
        poly = coco_det_to_poly(a)
        if poly is None or poly.shape != (4, 2):
            continue
        gt_by_img_cls.setdefault((img_id, cid), []).append({"poly": poly, "matched": False})
        npos_by_cls[cid] = npos_by_cls.get(cid, 0) + 1

    # Pred list (filter invalid, keep only those referencing GT images)
    preds: List[Tuple[int, float, int, np.ndarray]] = []
    for d in pred_dets:
        try:
            img_id = int(d["image_id"])
            cid = int(d["category_id"])
            score = float(d.get("score", d.get("confidence", 0.0)))
        except Exception:
            continue
        if gt_img_ids and img_id not in gt_img_ids:
            continue
        poly = coco_det_to_poly(d)
        if poly is None or poly.shape != (4, 2):
            continue
        preds.append((img_id, score, cid, poly))

    # Compute recall with greedy matching per class (like eval.py)
    recall_hits_by_cls: Dict[int, int] = {cid: 0 for cid in npos_by_cls.keys()}

    for cid in sorted(npos_by_cls.keys()):
        # reset matched flags for this class
        for (img_id, c), arr in gt_by_img_cls.items():
            if c == cid:
                for it in arr:
                    it["matched"] = False

        cls_preds = [(img_id, score, poly) for (img_id, score, c, poly) in preds if c == cid]
        cls_preds.sort(key=lambda x: -x[1])

        matched_cnt = 0
        for img_id, score, ppoly in cls_preds:
            if score < conf_thresh:
                continue
            gts = gt_by_img_cls.get((img_id, cid), [])
            iou_max, jmax = 0.0, -1
            for j, g in enumerate(gts):
                if g["matched"]:
                    continue
                iou = _iou(ppoly, g["poly"])
                if iou > iou_max:
                    iou_max, jmax = iou, j
            if iou_max >= recall_iou and jmax >= 0:
                gts[jmax]["matched"] = True
                matched_cnt += 1

        recall_hits_by_cls[cid] = matched_cnt

    total_pos = sum(npos_by_cls.values())
    total_hits = sum(recall_hits_by_cls.values())
    recall_overall = float(total_hits / max(total_pos, 1e-12)) if total_pos > 0 else 0.0

    per_class = {}
    for cid in sorted(npos_by_cls.keys()):
        name = class_name_map.get(cid, str(cid)) if class_name_map else str(cid)
        per_class[name] = float(recall_hits_by_cls.get(cid, 0) / max(npos_by_cls.get(cid, 0), 1e-12))

    return {
        "recall_overall": recall_overall,
        "total_gt": int(total_pos),
        "matched_gt": int(total_hits),
        "per_class_recall": per_class,
        "num_pred_total": int(len(preds)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT COCO dataset JSON exported by eval.py (tta_gt_conf*.coco.json)")
    ap.add_argument("--gt_a", default=None, help="Optional GT JSON that defines pred_a image_id mapping; defaults to --gt")
    ap.add_argument("--gt_b", default=None, help="Optional GT JSON that defines pred_b image_id mapping")
    ap.add_argument("--pred_a", required=True, help="Prediction COCO results JSON A (tta_pred_conf*.coco.json)")
    ap.add_argument("--pred_b", required=True, help="Prediction COCO results JSON B (tta_pred_conf*.coco.json)")
    ap.add_argument("--conf_thresh", type=float, default=0.10, help="confidence threshold for recall")
    ap.add_argument("--recall_iou", type=float, default=0.50, help="IoU threshold for recall matching")
    ap.add_argument("--iou_mode", choices=["obb", "aabb"], default="obb", help="IoU type (obb=polygon IoU, aabb=axis-aligned)")
    ap.add_argument("--print_per_class", action="store_true", help="print per-class recall")
    ap.add_argument("--check_gt_diff", action="store_true", help="compare --gt against --gt_a/--gt_b by file_name/category counts")
    args = ap.parse_args()

    gt_path = Path(args.gt).resolve()
    gt_a_path = Path(args.gt_a).resolve() if args.gt_a else None
    gt_b_path = Path(args.gt_b).resolve() if args.gt_b else None
    pred_a_path = Path(args.pred_a).resolve()
    pred_b_path = Path(args.pred_b).resolve()

    gt_img_ids, gt_anns, cat_name, gt_id_to_file, gt_file_to_id = load_gt_coco(gt_path)
    det_a = load_pred_coco(pred_a_path)
    det_b = load_pred_coco(pred_b_path)
    det_a = remap_dets_to_canonical_gt(det_a, gt_a_path, gt_file_to_id)
    det_b = remap_dets_to_canonical_gt(det_b, gt_b_path, gt_file_to_id)

    # Basic sanity checks
    def _warn_bad_ids(dets: list, label: str):
        bad = 0
        for d in dets:
            try:
                img_id = int(d["image_id"])
            except Exception:
                continue
            if gt_img_ids and img_id not in gt_img_ids:
                bad += 1
        if bad > 0:
            print(f"[WARN] {label}: {bad} detections refer to image_id not in GT; they will be ignored.")

    _warn_bad_ids(det_a, "pred_a")
    _warn_bad_ids(det_b, "pred_b")

    res_a = compute_recall_from_coco(
        gt_img_ids, gt_anns, det_a,
        conf_thresh=args.conf_thresh,
        recall_iou=args.recall_iou,
        iou_mode=args.iou_mode,
        class_name_map=cat_name,
    )
    res_b = compute_recall_from_coco(
        gt_img_ids, gt_anns, det_b,
        conf_thresh=args.conf_thresh,
        recall_iou=args.recall_iou,
        iou_mode=args.iou_mode,
        class_name_map=cat_name,
    )
    res_u = compute_recall_from_coco(
        gt_img_ids, gt_anns, list(det_a) + list(det_b),
        conf_thresh=args.conf_thresh,
        recall_iou=args.recall_iou,
        iou_mode=args.iou_mode,
        class_name_map=cat_name,
    )

    print("\n[RECALL SUMMARY]")
    print(f"GT: {gt_path}")
    if gt_a_path:
        print(f"gt_a map: {gt_a_path}")
    if gt_b_path:
        print(f"gt_b map: {gt_b_path}")
    print(f"pred_a: {pred_a_path}")
    print(f"pred_b: {pred_b_path}")
    print(f"conf_thresh={args.conf_thresh}  recall_iou={args.recall_iou}  iou_mode={args.iou_mode}")
    print("")
    print(f"recall_a:     {res_a['recall_overall']:.4f}  (matched {res_a['matched_gt']}/{res_a['total_gt']}, preds={res_a['num_pred_total']})")
    print(f"recall_b:     {res_b['recall_overall']:.4f}  (matched {res_b['matched_gt']}/{res_b['total_gt']}, preds={res_b['num_pred_total']})")
    print(f"recall_union: {res_u['recall_overall']:.4f}  (matched {res_u['matched_gt']}/{res_u['total_gt']}, preds={res_u['num_pred_total']})")

    if args.print_per_class:
        print("\n[PER-CLASS RECALL] (union)")
        for k, v in res_u["per_class_recall"].items():
            print(f"{k}\t{v:.4f}")

    if args.check_gt_diff:
        if gt_a_path:
            print("")
            print_gt_diff(gt_path, gt_a_path, gt_id_to_file, gt_img_ids, gt_anns)
        if gt_b_path:
            print("")
            print_gt_diff(gt_path, gt_b_path, gt_id_to_file, gt_img_ids, gt_anns)


if __name__ == "__main__":
    main()

