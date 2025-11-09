#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dir直下の .tif を逐次GPU推論し、以下のいずれか/両方を出力するツール
  (A) CVATにインポート可能な COCO 1.0 (instances) JSON（画像は含めない）
      - annotations[].bbox は回転を無視した AABB [x, y, w, h]
      - annotations[].segmentation は polygon(画素座標) を保持
  (B) Ultralytics YOLO OBB 形式 (labels/*.txt) と対応画像 (images/)
      - 1行: <cls> <xc> <yc> <w> <h> <angle_rad>   ※座標は画像サイズで正規化、角度はラジアン

使い方例:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # (A) COCO JSON 出力
  python export_coco_from_yolo_obb.py -d /path/to/dir -m best.pt \
      --out-json coco.json -s 1920 -g cuda:0 --half

  # (B) YOLO-OBB 出力（画像も保存）
  python export_coco_from_yolo_obb.py -d /path/to/dir -m best.pt \
      --yolo-obb-dir out_obb --save-images-mode copy

  # (A)+(B) 両方
  python export_coco_from_yolo_obb.py -d /path/to/dir -m best.pt \
      --out-json coco.json --yolo-obb-dir out_obb

注意:
- YOLO-OBB の学習/検証に使う場合は images/ と labels/ の両方が必要です。
- 角度はラジアン。モデルから (x,y,w,h,rad) が得られない時は四隅から推定します。
"""

import argparse, gc, json, math, os, shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

import cv2
from tta_obb import predict_obb_with_tta

IMG_EXT = ".tif"

# ---------- 幾何ユーティリティ ----------

def polygon_area(xs: np.ndarray, ys: np.ndarray) -> float:
    # shoelace formula
    return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))

def clamp_polygon_to_image(poly_xyxy, W, H):
    arr = np.asarray(poly_xyxy, dtype=np.float64).reshape(-1)
    xs, ys = arr[0::2], arr[1::2]
    xs = np.clip(xs, 0.0, max(float(W - 1), 0.0))
    ys = np.clip(ys, 0.0, max(float(H - 1), 0.0))
    out = np.empty_like(arr)
    out[0::2], out[1::2] = xs, ys
    return out.tolist()

def poly_valid(poly_xyxy, min_area_px=1.0):
    if poly_xyxy is None:
        return False
    arr = np.asarray(poly_xyxy, dtype=np.float64).reshape(-1)
    if arr.size < 6:  # 少なくとも3頂点(=6値)必要
        return False
    if not np.all(np.isfinite(arr)):
        return False
    xs, ys = arr[0::2], arr[1::2]
    area = polygon_area(xs, ys)
    return area > float(min_area_px)

def xyrwha_to_poly(xc, yc, w, h, ang_rad):
    """(cx,cy,w,h,angle[rad]) -> 4点多角形 (x1,y1,...,x4,y4) 画素座標"""
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    dx, dy = w / 2.0, h / 2.0
    rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]], dtype=np.float64)
    R = np.array([[c, -s], [s,  c]], dtype=np.float64)
    pts = (rect @ R.T) + np.array([xc, yc], dtype=np.float64)
    return pts.reshape(-1).tolist()  # [x1,y1,...,x4,y4]

def poly_to_bbox(poly_xyxy):
    arr = np.asarray(poly_xyxy, dtype=np.float64).reshape(-1)
    xs, ys = arr[0::2], arr[1::2]
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

def _normalize_angle_wh(a, w, h):
    """
    角度を (-pi/2, pi/2] に正規化。必要なら w<->h を入替え、角度±pi/2 調整。
    """
    # (-pi, pi] にまず正規化
    a = (a + math.pi) % (2 * math.pi) - math.pi
    # (-pi/2, pi/2] に畳み込み
    if a <= -math.pi/2:
        a += math.pi
        w, h = h, w
    elif a > math.pi/2:
        a -= math.pi
        w, h = h, w
    return a, w, h

def poly4_to_xyrwha(poly):
    """
    四隅 (x1,y1,...,x4,y4) から (xc,yc,w,h,a[rad]) を推定。
    頂点順序は矩形の巡回順を仮定（Ultralyticsの xyxyxyxy を想定）。
    """
    arr = np.asarray(poly, dtype=np.float64).reshape(-1)
    pts = arr.reshape(4, 2)
    xc, yc = pts[:,0].mean(), pts[:,1].mean()
    # 辺長を算出し、長辺/短辺を決定
    edges = np.roll(pts, -1, axis=0) - pts
    lens = np.linalg.norm(edges, axis=1)
    # 最初の辺ベクトルから角度を取る（長辺・短辺に依らず代表一本）
    # 幅=隣接2辺のうち長い方、高さ=短い方 とする
    i_long = int(np.argmax(lens))
    i_short = (i_long + 1) % 4
    w = lens[i_long]
    h = lens[i_short]
    ex, ey = edges[i_long]
    a = math.atan2(ey, ex)  # ラジアン
    a, w, h = _normalize_angle_wh(a, w, h)
    return float(xc), float(yc), float(w), float(h), float(a)

# ---------- 画像保存ユーティリティ ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str = "copy"):
    """
    mode: copy | symlink | hardlink | auto
    """
    ensure_dir(dst.parent)
    if mode == "auto":
        # 同一ファイルシステムならハードリンク、無理ならコピー
        try:
            os.link(src, dst)
            return "hardlink"
        except Exception:
            shutil.copy2(src, dst)
            return "copy"
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
        except OSError:
            # Windows等で失敗したらコピーにフォールバック
            shutil.copy2(src, dst)
            return "copy"
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown save-images-mode: {mode}")
    return mode

# ---------- メイン ----------

def main():
    ap = argparse.ArgumentParser(description="Export COCO JSON and/or Ultralytics YOLO-OBB labels from YOLO OBB inference")
    ap.add_argument("-d", "--dir", required=True, help="Directory containing .tif images (non-recursive)")
    ap.add_argument("-m", "--model", required=True, help="Path to YOLO11-OBB model (.pt)")
    # (A) COCO JSON
    ap.add_argument("--out-json", default=None, help="Output COCO JSON path (if set, export COCO)")
    ap.add_argument("--cvat-root", type=str, default="", help="COCO images[].file_name の先頭に付ける相対ルート")
    ap.add_argument("--no-strip-ext", action="store_true", help="拡張子を保持したまま file_name を書く（既定は拡張子を外す）")
    # (B) YOLO-OBB
    ap.add_argument("--yolo-obb-dir", default=None, help="Export Ultralytics OBB dataset here if set (creates images/ and labels/)")
    ap.add_argument("--save-images-mode", choices=["copy", "symlink", "hardlink", "auto", "none"], default="copy",
                    help="YOLO-OBB用に画像を保存する方法（noneだと保存しない※非推奨）")
    ap.add_argument("--yolo-images-subdir", default="images", help="YOLO-OBB 画像サブディレクトリ名")
    ap.add_argument("--yolo-labels-subdir", default="labels", help="YOLO-OBB ラベルサブディレクトリ名")
    ap.add_argument("--yolo-write-classes", action="store_true", help="names.txt を書き出す（1行1クラス名）")
    # 推論設定
    ap.add_argument("-c", "--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("-s", "--imgsz", type=int, default=1920, help="Inference image size")
    ap.add_argument("-g", "--device", default=None, help="Device (e.g., cuda:0). Default auto")
    ap.add_argument("--half", action="store_true", help="Use fp16 on CUDA")
    ap.add_argument("--min-area", type=float, default=1.0, help="Minimum polygon area in pixels to keep")
    ap.add_argument("--tta", default="none", choices=["none","flips","rot90","all"],
                help="flip/rotate test-time augmentation")
    ap.add_argument("--tta-merge-iou", type=float, default=0.50,
                    help="rotated-NMS IoU for merging TTA views")
    ap.add_argument("--nms-iou", type=float, default=0.70,
                    help="per-call NMS IoU passed to Ultralytics model.predict")
    args = ap.parse_args()

    # どちらも指定なしはエラー
    if args.out_json is None and args.yolo_obb_dir is None:
        raise SystemExit("Specify at least one of --out-json or --yolo-obb-dir")

    def build_cvat_file_name(p: Path, root: str, base_dir: Path, keep_ext: bool) -> str:
        rel = str(p.relative_to(base_dir)).replace("\\", "/")
        rel_comp = rel if keep_ext else rel.rsplit(".", 1)[0]
        root = (root or "").strip().replace("\\", "/").strip("/")
        return f"{root}/{rel_comp}" if root else rel_comp

    # device
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 画像列挙（直下のみ）
    img_paths = sorted([p for p in Path(args.dir).glob(f"*{IMG_EXT}")])
    if not img_paths:
        raise SystemExit(f"No {IMG_EXT} images found in the top-level of the directory.")

    # モデル・クラス名
    model = YOLO(args.model)
    names = getattr(model, "names", {})
    if isinstance(names, dict) and names:
        class_names = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, (list, tuple)) and len(names) > 0:
        class_names = list(names)
    else:
        class_names = ["object"]

    # (A) COCO JSON 準備
    if args.out_json:
        categories = [{"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(class_names)]
        coco = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1
    else:
        coco, ann_id = None, None

    # (B) YOLO-OBB 準備
    if args.yolo_obb_dir:
        yolo_root = Path(args.yolo_obb_dir)
        img_dir = yolo_root / args.yolo_images_subdir
        lbl_dir = yolo_root / args.yolo_labels_subdir
        ensure_dir(img_dir)
        ensure_dir(lbl_dir)
        if args.yolo_write_classes:
            with (yolo_root / "names.txt").open("w", encoding="utf-8") as f:
                for n in class_names:
                    f.write(f"{n}\n")

    # 推論ループ
    for img_id, p in enumerate(tqdm(img_paths, total=len(img_paths)), start=1):
        # 画像サイズ
        img_np = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img_np is None:
            raise RuntimeError(f"Failed to read image: {p}")
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        H, W = img_np.shape[:2] 

        # (A) COCO: images[]
        if coco is not None:
            file_name_for_cvat = build_cvat_file_name(
                p=p, root=args.cvat_root, base_dir=Path(args.dir), keep_ext=args.no_strip_ext
            )
            coco["images"].append({
                "id": img_id,
                "file_name": file_name_for_cvat,  # CVATの name と一致させる
                "width": int(W),
                "height": int(H),
            })

        # (B) YOLO-OBB: 画像保存
        if args.yolo_obb_dir:
            # 出力先ファイル名は元ファイル名をそのまま使う（サブディレクトリなし）
            dst_img = (Path(args.yolo_obb_dir) / args.yolo_images_subdir / p.name)
            if args.save_images_mode != "none":
                link_or_copy(p, dst_img, args.save_images_mode)

        # TTA推論（複数ビュー→逆変換→集約→回転NMS）
        polys_np, pred_classes, pred_confs, xywhr_merged = predict_obb_with_tta(
            model=model,
            img_bgr=img_np,
            conf=args.conf,
            iou=args.nms_iou,          # ← Ultralytics側のNMS IoU
            imgsz=args.imgsz,
            tta_mode=args.tta,
            tta_merge_iou=args.tta_merge_iou  # ← TTAビュー統合用の回転NMS IoU
        )

        # 1画像分: 出力用の構造に詰め替え
        polys = []       # [(poly(list[8]), cls(int))]
        obb_params = []  # [(xc,yc,w,h,a(rad),cls)]

        for (poly_np, cls_idx, score, xywhr) in zip(polys_np, pred_classes, pred_confs, xywhr_merged):
            # polygon（画素座標; 8値）をCOCO用に
            poly_pix = poly_np.reshape(-1).astype(float).tolist()
            polys.append((poly_pix, int(cls_idx)))

            # xywhr_merged は (cx,cy,w,h,deg)。YOLO-OBBはラジアンで書く
            cx, cy, ww, hh, a_deg = xywhr
            a_rad = np.deg2rad(a_deg)
            # 軽く (-pi/2, pi/2] に畳み込み（eval側と規約一致）
            if a_rad <= -np.pi/2:
                a_rad += np.pi
            elif a_rad > np.pi/2:
                a_rad -= np.pi
            obb_params.append((float(cx), float(cy), float(ww), float(hh), float(a_rad), int(cls_idx)))

        # (A) COCO JSON: annotations 追加（AABB bbox）
        if coco is not None:
            for poly_pix, cls_idx in polys:
                # 画像境界にクリップ & 面積チェック
                poly_pix = clamp_polygon_to_image(poly_pix, W, H)
                if not poly_valid(poly_pix, min_area_px=args.min_area):
                    continue
                arr = np.asarray(poly_pix, dtype=np.float64).reshape(-1)
                xs, ys = arr[0::2], arr[1::2]
                area = float(polygon_area(xs, ys))
                if area <= 0.0:
                    continue
                bbox = poly_to_bbox(arr)  # AABB
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_idx) + 1,  # 1始まり
                    #"segmentation": [arr.tolist()],   # COCOは list[list[...]]
                    #"area": area,
                    "bbox": [float(b) for b in bbox],
                    "iscrowd": 0,
                })
                ann_id += 1

        # (B) YOLO-OBB: labels/*.txt 書き出し（正規化）
        if args.yolo_obb_dir:
            if len(obb_params) > 0:
                lbl_path = Path(args.yolo_obb_dir) / args.yolo_labels_subdir / (p.stem + ".txt")
                with lbl_path.open("w", encoding="utf-8") as f:
                    for xc, yc, ww, hh, a, cls_idx in obb_params:
                        # 画素 -> 正規化
                        xc_n = xc / W
                        yc_n = yc / H
                        w_n  = max(0.0, ww / W)
                        h_n  = max(0.0, hh / H)
                        # 角度はそのままラジアン（正規化済み）
                        f.write(f"{int(cls_idx)} {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f} {a:.6f}\n")
            else:
                # 検出なしでも空ファイルを置くのが無難
                lbl_path = Path(args.yolo_obb_dir) / args.yolo_labels_subdir / (p.stem + ".txt")
                ensure_dir(lbl_path.parent)
                lbl_path.touch(exist_ok=True)

        # 後処理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # JSON 保存
    if coco is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"[OK] Wrote COCO annotations: {out_path}")
        print("CVAT: Task -> Actions -> Import annotations -> COCO 1.0 (instances)")
        print("※ 既存タスクのフレーム名と images[].file_name が完全一致している必要があります。")

    if args.yolo_obb_dir:
        print(f"[OK] Wrote Ultralytics OBB dataset to: {args.yolo_obb_dir}")
        print("構成: images/（画像）, labels/（.txt: cls xc yc w h angle[rad], 正規化）。")

if __name__ == "__main__":
    main()

