#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBBヒット/ミスのサイズ・向き統計を出すツール
- 入力: GTラベルdir, 予測ラベルdir（YOLO OBB: cls x1 y1 x2 y2 x3 y3 x4 y4 [conf]、0..1正規化）
- 画像dirを渡すと実寸(px)で評価（推奨）。未指定なら正規化座標のまま評価。
- IoU>=thr（デフォ 0.2）でクラス一致（--ignore-classで無視）かつ貪欲にマッチングしてTP/FP/FN判定
- 出力:
    - 標準出力: 全体のTP/FP/FN、面積/アスペクト/角度ビンごとのRecall/Precision
    - CSV: per_instance.csv（各GT/Pred行に特徴量とTP/FP/FNフラグ）
"""

import argparse, csv
from pathlib import Path
import numpy as np
import cv2

# ---------- 幾何ユーティリティ ----------
def poly_area(P):
    return 0.5 * abs(np.dot(P[:,0], np.roll(P[:,1],-1)) - np.dot(P[:,1], np.roll(P[:,0],-1)))

def suth_hodg_clip(subject, clipper):
    """Sutherland–Hodgman polygon clipping (intersection of subject ∩ clipper)."""
    def edge_clip(S, E, Pts):
        out = []
        S, E = np.asarray(S, float), np.asarray(E, float)
        for i in range(len(Pts)):
            C = np.asarray(Pts[i-1], float)
            D = np.asarray(Pts[i], float)
            inD = np.cross(E - S, D - S) >= 0
            inC = np.cross(E - S, C - S) >= 0
            if inD:
                if not inC:
                    out.append(line_inter(C, D, S, E))
                out.append(D.tolist())
            elif inC:
                out.append(line_inter(C, D, S, E).tolist())
        return out

    def line_inter(A, B, S, E):
        """線分ABと線分SEの交点を返す"""
        A, B, S, E = np.asarray(A, float), np.asarray(B, float), np.asarray(S, float), np.asarray(E, float)
        BA = B - A
        ES = E - S
        denom = (BA[0]*ES[1] - BA[1]*ES[0])
        if abs(denom) < 1e-12:
            return B
        t = ((S[0]-A[0])*ES[1] - (S[1]-A[1])*ES[0]) / denom
        return A + t * BA

    out = [list(map(float, p)) for p in subject]
    if not out:
        return np.empty((0,2))
    for i in range(len(clipper)):
        S = np.asarray(clipper[i-1], float)
        E = np.asarray(clipper[i], float)
        out = edge_clip(S, E, out)
        if not out:
            break
    return np.array(out, dtype=float)

def poly_iou(P, Q):
    inter = suth_hodg_clip(P, Q)
    if inter.size == 0: return 0.0
    ai = poly_area(inter); aP = poly_area(P); aQ = poly_area(Q)
    if aP <= 0 or aQ <= 0: return 0.0
    return float(ai / (aP + aQ - ai + 1e-12))

def yolo_poly_to_pixels(parts, W=1.0, H=1.0):
    # parts: [cls, x1,y1,...,x4,y4, (opt conf)]
    pts = np.array(list(map(float, parts[1:9])), dtype=float).reshape(4,2)
    pts[:,0] *= W; pts[:,1] *= H
    return pts

def ensure_ccw(P):
    return P if np.cross(P[1]-P[0], P[2]-P[1]) >= 0 else P[::-1]

def obb_features(P):
    """面積, アスペクト比(長/短), 角度[deg](長辺の角度: 0..180)"""
    P = ensure_ccw(P)
    e = [P[(i+1)%4]-P[i] for i in range(4)]
    L = [np.linalg.norm(v) for v in e]
    # 長辺ベクトルを e[idx] とする
    idx = int(np.argmax([L[0], L[1]]))  # e0(=上辺) vs e1(=右辺)だけで十分
    long_vec = e[idx]
    short_len = L[1] if idx==0 else L[0]
    long_len  = L[0] if idx==0 else L[1]
    aspect = (long_len+1e-12)/(short_len+1e-12)
    ang = np.degrees(np.arctan2(long_vec[1], long_vec[0])) % 180.0
    area = poly_area(P)
    return area, aspect, ang

# ---------- I/O ----------
def read_label_file(txt_path):
    if not txt_path.exists(): return []
    out = []
    for ln, s in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), 1):
        if not s.strip(): continue
        parts = s.split()
        if len(parts) < 9: continue
        cls = int(float(parts[0]))
        conf = float(parts[9]) if len(parts) >= 10 else None
        out.append((cls, parts, conf))
    return out

# ---------- マッチング ----------
def match_greedy(gt_list, pr_list, iou_thr=0.2, classwise=True):
    """gt/pr: list of dict {poly, cls, conf?} をIoUで貪欲マッチング"""
    pairs = []
    for gi, g in enumerate(gt_list):
        for pi, p in enumerate(pr_list):
            if classwise and g["cls"] != p["cls"]: continue
            iou = poly_iou(g["poly"], p["poly"])
            if iou >= iou_thr:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g, used_p = set(), set()
    matches = []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p: continue
        used_g.add(gi); used_p.add(pi)
        matches.append((gi, pi, iou))
    unmatched_g = [i for i in range(len(gt_list)) if i not in used_g]
    unmatched_p = [i for i in range(len(pr_list)) if i not in used_p]
    return matches, unmatched_g, unmatched_p

# ---------- 集計 ----------
def bin_id(val, bins):
    return int(np.digitize([val], bins, right=False)[0])  # 1..len(bins)
def pretty_bin_edges(bins):
    return ["(-inf,{:.3g}]".format(bins[0])] + \
           ["({:.3g},{:.3g}]".format(bins[i-1], bins[i]) for i in range(1,len(bins))] + \
           ["({:.3g},inf)".format(bins[-1])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", '-g', required=True, type=Path, help="GT labels dir")
    ap.add_argument("--pred", '-p', required=True, type=Path, help="Pred labels dir")
    ap.add_argument("--images", '-i', type=Path, default=None, help="Images dir (推奨: W,H取得)")
    ap.add_argument("--exts", type=str, default="jpg,jpeg,png,bmp,tif,tiff,webp")
    ap.add_argument("--iou-thr", type=float, default=0.2)
    ap.add_argument("--ignore-class", action="store_true")
    ap.add_argument("--out-csv", '-o', type=Path, default=Path("per_instance.csv"))
    ap.add_argument("--area-bins", type=str, default="0.0,200,800,3200")  # px^2基準（画像dirなしなら正規化^2）
    ap.add_argument("--aspect-bins", type=str, default="1.5,3,6,12")
    ap.add_argument("--angle-bins", type=str, default="15,30,45,60,75,90")
    args = ap.parse_args()

    exts = set("."+e.strip().lower() for e in args.exts.split(",") if e.strip())
    area_bins   = np.array([float(x) for x in args.area_bins.split(",") if x], dtype=float)
    aspect_bins = np.array([float(x) for x in args.aspect_bins.split(",") if x], dtype=float)
    angle_bins  = np.array([float(x) for x in args.angle_bins.split(",")  if x], dtype=float)

    # 画像サイズ取得
    size_cache = {}
    def img_size(stem):
        if args.images is None:
            return (1.0, 1.0)
        if stem in size_cache:
            return size_cache[stem]
        # 探索
        for p in args.images.rglob(stem + ".*"):
            if p.suffix.lower() in exts:
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None: continue
                H, W = img.shape[:2]
                size_cache[stem] = (W, H)
                return W, H
        size_cache[stem] = (1.0, 1.0)  # fallback
        return (1.0, 1.0)

    rows = []
    stems = sorted(set([p.stem for p in args.gt.rglob("*.txt")]) |
                   set([p.stem for p in args.pred.rglob("*.txt")]))

    TP = FP = FN = 0

    for stem in stems:
        gt_path  = args.gt   / f"{stem}.txt"
        pr_path  = args.pred / f"{stem}.txt"
        W, H = img_size(stem)

        gtl = read_label_file(gt_path)
        prl = read_label_file(pr_path)

        gt_objs = [{"cls":cls, "poly":ensure_ccw(yolo_poly_to_pixels(parts, W, H))}
                   for (cls, parts, _) in gtl]
        pr_objs = [{"cls":cls, "poly":ensure_ccw(yolo_poly_to_pixels(parts, W, H)), "conf":conf}
                   for (cls, parts, conf) in prl]

        matches, um_g, um_p = match_greedy(gt_objs, pr_objs, iou_thr=args.iou_thr,
                                           classwise=(not args.ignore_class))

        # TP
        for gi, pi, iou in matches:
            area, aspect, ang = obb_features(gt_objs[gi]["poly"])
            rows.append({
                "image": stem, "role": "TP", "iou": iou, "cls": gt_objs[gi]["cls"],
                "area": area, "aspect": aspect, "angle": ang,
                "conf": pr_objs[pi].get("conf", None)
            })
        TP += len(matches)

        # FN（GT未検出）
        for gi in um_g:
            area, aspect, ang = obb_features(gt_objs[gi]["poly"])
            rows.append({
                "image": stem, "role": "FN", "iou": 0.0, "cls": gt_objs[gi]["cls"],
                "area": area, "aspect": aspect, "angle": ang, "conf": None
            })
        FN += len(um_g)

        # FP（余計な検出）
        for pi in um_p:
            area, aspect, ang = obb_features(pr_objs[pi]["poly"])
            rows.append({
                "image": stem, "role": "FP", "iou": 0.0, "cls": pr_objs[pi]["cls"],
                "area": area, "aspect": aspect, "angle": ang,
                "conf": pr_objs[pi].get("conf", None)
            })
        FP += len(um_p)

    # CSV出力
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image","role","cls","iou","conf","area","aspect","angle"])
        w.writeheader(); w.writerows(rows)

    # 集計（リコール=TP/(TP+FN), 適合率=TP/(TP+FP) をビン別で）
    rows_np = {k: np.array([r[k] for r in rows if r[k] is not None]) for k in ["area","aspect","angle"]}
    roles = np.array([r["role"] for r in rows])

    def summarize_by_bins(var, bins, name):
        print(f"\n== {name} bins ==")
        edges = pretty_bin_edges(bins)
        for b in range(len(edges)):
            # bin index: 0..len(bins) gives len(bins)+1 bins (±infを含む)
            mask = np.array([bin_id(v, bins)==b for v in rows_np[var]])
            tp = np.sum((roles=="TP") & mask)
            fn = np.sum((roles=="FN") & mask)
            fp = np.sum((roles=="FP") & mask)
            rec = tp / (tp + fn) if (tp+fn)>0 else np.nan
            prec = tp / (tp + fp) if (tp+fp)>0 else np.nan
            print(f"{edges[b]:>16s} | count={tp+fn+fp:4d}  recall={rec:6.3f}  precision={prec:6.3f}  (TP={tp}, FN={fn}, FP={fp})")

    print(f"\n=== Overall ===\nTP={TP}  FP={FP}  FN={FN}")
    rec_all = TP/(TP+FN) if (TP+FN)>0 else float("nan")
    prec_all= TP/(TP+FP) if (TP+FP)>0 else float("nan")
    print(f"Recall={rec_all:.3f}  Precision={prec_all:.3f}")
    summarize_by_bins("area",   area_bins,   "Area")
    summarize_by_bins("aspect", aspect_bins, "Aspect ratio (long/short)")
    summarize_by_bins("angle",  angle_bins,  "Angle (deg)")

if __name__ == "__main__":
    main()
