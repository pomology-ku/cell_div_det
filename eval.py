#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay GT and inference OBBs on images and save.
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
    # まずは images -> labels の置換（標準ケース）
    if "images" in parts:
        idx = parts.index("images")
        lbl_parts = parts.copy()
        lbl_parts[idx] = "labels"
        lbl_dir = Path(*lbl_parts[:-1])
        cand = lbl_dir / f"{img_path.stem}.txt"
        if cand.exists():
            return cand
        # 余分な階層があっても拾えるように再帰検索
        found = list(lbl_dir.rglob(f"{img_path.stem}.txt"))
        if found:
            return found[0]

    # 兄弟階層: <parent_of_images>/labels/**/<stem>.txt を再帰探索
    root = img_path.parents[2] if len(img_path.parents) >= 3 else img_path.parent
    for lbl_root in [root / "labels", img_path.parent.parent / "labels"]:
        if lbl_root.exists():
            found = list(lbl_root.rglob(f"{img_path.stem}.txt"))
            if found:
                return found[0]

    # 同ディレクトリにあるケース
    cand2 = img_path.with_suffix(".txt")
    return cand2 if cand2.exists() else None

# ---------------------- Geometry ----------------------

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

def load_gt_polys_yolo_obb(lbl_path: Path, img_w: int, img_h: int,
                           angle_unit: str = "deg") -> List[Tuple[np.ndarray, int]]:
    polys: List[Tuple[np.ndarray, int]] = []
    if not lbl_path or not lbl_path.exists():
        return polys
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            cls = int(float(parts[0]))
            x, y, w, h, ang = map(float, parts[1:6])
            cx = x * img_w
            cy = y * img_h
            ww = w * img_w
            hh = h * img_h
            angle_deg = np.rad2deg(ang) if angle_unit == "rad" else ang
            poly = obb_to_polygon(cx, cy, ww, hh, angle_deg)
            polys.append((poly, cls))
    return polys

def as_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

def draw_poly(img: np.ndarray, poly: np.ndarray, color: Tuple[int,int,int], thick: int, label: Optional[str] = None):
    pts = poly.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
    if label:
        p0 = tuple(pts[0, 0].tolist())
        cv2.putText(img, label, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def load_gt_polys_auto(lbl_path: Path, img_w: int, img_h: int) -> list[tuple[np.ndarray, int]]:
    """対応:
       (A) YOLO-OBB: <cls> cx cy w h ang(deg)
       (B) 四角形:   <cls> x1 y1 x2 y2 x3 y3 x4 y4  (+任意の付加列)
       値が 0..1 なら正規化として画素に変換、>1 を含めば画素値として扱う。
    """
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
            # 先頭はクラスIDを想定
            if not is_float(toks[0]):
                # DOTA形式 (クラス名が末尾) などに備える（例: 8座標 + class + diff）
                # → 最後の8個が座標ならそれを使う
                nums = [t for t in toks if is_float(t)]
            else:
                nums = toks

            # YOLO-OBB: cls + 5 = 6
            if len(nums) >= 6 and len(nums) < 9:
                try:
                    cls = int(float(nums[0]))
                    cx, cy, w, h, ang = map(float, nums[1:6])
                except Exception:
                    continue
                # 正規化→画素
                cx, cy = cx * img_w, cy * img_h
                w, h = w * img_w, h * img_h
                poly = obb_to_polygon(cx, cy, w, h, ang)  # 角度は度を想定
                polys.append((poly, cls))
                continue

            # 四角形: cls + 8 = 9 以上（付加列あってもOK）
            if len(nums) >= 9:
                try:
                    cls = int(float(nums[0]))
                except Exception:
                    cls = 0
                coords = nums[1:9]  # 最初の8数値を座標として使う
                if not all(is_float(x) for x in coords):
                    continue
                vals = list(map(float, coords))
                xs = vals[0::2]
                ys = vals[1::2]
                # 正規化 or 画素値の判定（0..1 の割合が高ければ正規化とみなす）
                frac_01 = sum(0.0 <= v <= 1.0 for v in vals) / len(vals)
                if frac_01 >= 0.75:
                    xs = [x * img_w for x in xs]
                    ys = [y * img_h for y in ys]
                poly = np.array(list(zip(xs, ys)), dtype=np.float32)
                # 必ず4点になるよう整形
                if poly.shape == (4, 2):
                    polys.append((poly, cls))
                continue
    return polys

# ---------------------- Metrics collection ----------------------

def try_collect_metrics_from_api(val_ret) -> Dict[str, float]:
    """
    Try to read metrics from Ultralytics API return.
    Supports both HBB('box') and OBB('obb') fields. Returns dict with keys:
      mAP50, mAP50_95, precision, recall, nc (num classes), n_images
    Missing fields are omitted.
    """
    out = {}
    try:
        m = getattr(val_ret, "metrics", None) or val_ret
        # Prefer OBB if present
        target = None
        for key in ["obb", "box", "segment"]:
            if hasattr(m, key) and getattr(m, key) is not None:
                target = getattr(m, key)
                break
        if target is not None:
            # Ultralytics exposes .map50 and .map (for 0.5:0.95)
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
        # global counts if available
        if hasattr(m, "speed") and hasattr(m.speed, "images"):
            out["n_images"] = int(m.speed.images)  # best effort
    except Exception:
        pass
    return out

def try_collect_metrics_from_csv(metrics_dir: Path) -> Dict[str, float]:
    """
    Parse Ultralytics results.csv if present, extracting top-line metrics.
    We try a few column name variants.
    """
    out = {}
    csv_path = metrics_dir / "results.csv"
    if not csv_path.exists():
        # search fallback
        cands = list(metrics_dir.glob("**/results.csv"))
        if cands:
            csv_path = cands[0]
        else:
            return out
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return out
    row = rows[-1]
    # candidate keys
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
                    out[k] = float(row[ck])
                    break
                except Exception:
                    continue
    return out

def copy_if_exists(src: Path, dst_dir: Path):
    if src.exists():
        ensure_dir(dst_dir)
        shutil.copy2(src, dst_dir / src.name)

def harvest_val_artifacts(val_proj_dir: Path, dest_dir: Path):
    """
    Copy common Ultralytics artifacts (PR curves, confusion matrix, etc.) into out/metrics/.
    """
    ensure_dir(dest_dir)
    # typical files
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", '-d', required=True, help="data.yaml (must contain the split to visualize)")
    ap.add_argument("--weights", '-w', required=True, help="path to best.pt")
    ap.add_argument("--out", '-o', required=True, help="output directory")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"], help="which split")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for predictions")
    ap.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold (inference)")
    ap.add_argument("--angle_unit", choices=["deg", "rad"], default="deg", help="GT angle unit")
    ap.add_argument("--thick", type=int, default=4, help="line thickness for overlay")
    ap.add_argument("--max_images", type=int, default=0, help="limit number of visualized images (0=all)")
    ap.add_argument("--device", default=None, help="device for Ultralytics (e.g., 0, '0,1', 'cpu')")
    ap.add_argument("--skip_vis", action="store_true", help="skip overlays, compute metrics only")
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    cfg = load_yaml(data_yaml)
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

        for i, img_path in enumerate(img_paths, 1):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            H, W = img.shape[:2]
            canvas = as_bgr(img.copy())

            # GT
            lbl_path = map_image_to_label(img_path)
            if lbl_path:
                for poly, cls in load_gt_polys_auto(lbl_path, W, H):
                    txt = f"GT:{class_names.get(cls, str(cls))}" if class_names else None
                    draw_poly(canvas, poly, C_GT, thick, label=txt) 

            # Predictions
            res = model.predict(source=str(img_path), conf=args.conf, iou=args.iou, verbose=False)[0]
            # Prefer oriented polygons
            pred_polys: List[np.ndarray] = []
            pred_classes: List[int] = []
            pred_confs: List[Optional[float]] = []

            if hasattr(res, "obb") and res.obb is not None:
                if hasattr(res.obb, "xyxyxyxy") and res.obb.xyxyxyxy is not None:
                    polys = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
                    pred_polys.extend(polys)
                elif hasattr(res.obb, "xywhr") and res.obb.xywhr is not None:
                    xywhr = res.obb.xywhr.cpu().numpy()
                    for cx, cy, ww, hh, ang in xywhr:
                        angle_deg = np.rad2deg(ang) if abs(ang) <= 3.2 else ang
                        pred_polys.append(obb_to_polygon(cx, cy, ww, hh, angle_deg))
            else:
                if res.boxes is not None and res.boxes.xyxy is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    for x1, y1, x2, y2 in xyxy:
                        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                        pred_polys.append(poly)

            if getattr(res, "boxes", None) is not None and res.boxes is not None:
                if res.boxes.cls is not None:
                    pred_classes = res.boxes.cls.cpu().numpy().astype(int).tolist()
                if res.boxes.conf is not None:
                    pred_confs = res.boxes.conf.cpu().numpy().tolist()
            # pad
            if len(pred_classes) < len(pred_polys):
                pred_classes += [-1] * (len(pred_polys) - len(pred_classes))
            if len(pred_confs) < len(pred_polys):
                pred_confs += [None] * (len(pred_polys) - len(pred_confs))

            for poly, cls, conf in zip(pred_polys, pred_classes, pred_confs):
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
            cv2.putText(canvas, "Pred", (140, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_PR, 2, cv2.LINE_AA)
            cv2.line(canvas, (210, y0-8), (270, y0-8), C_PR, thick, cv2.LINE_AA)

            out_path = out_dir / f"{img_path.stem}_gt_pred.png"
            cv2.imwrite(str(out_path), canvas)
            print(f"[{i}/{len(img_paths)}] saved -> {out_path}")

    # ---------- Metrics via Ultralytics val ----------
    print("\n[VAL] Running Ultralytics validation to compute metrics ...")
    val_ret = model.val(
        data=str(data_yaml),
        split=split_key,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        plots=True,
        project=str(metrics_dir),  # write artifacts under out/metrics/
        name=".",
        verbose=False,
    )

    # Prefer API, fallback to CSV
    metrics = try_collect_metrics_from_api(val_ret)
    if not metrics:
        metrics = try_collect_metrics_from_csv(metrics_dir)

    # Save summaries
    sum_csv = metrics_dir / "metrics_summary.csv"
    sum_json = metrics_dir / "metrics_summary.json"
    if metrics:
        with open(sum_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in metrics.items():
                w.writerow([k, v])
        with open(sum_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[VAL] Summary saved: {sum_csv}")
    else:
        print("[VAL] Warning: could not parse metrics automatically (check Ultralytics CSV/JSON).")

    # Harvest common plots/files to metrics/
    harvest_val_artifacts(metrics_dir, metrics_dir)
    print("[DONE] Visuals in:", out_dir)
    print("[DONE] Metrics in:", metrics_dir)

if __name__ == "__main__":
    main()

