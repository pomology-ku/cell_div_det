import numpy as np
import cv2
import torch
from torchvision.ops import nms as tv_nms
from ultralytics import YOLO

def normalize_angle_deg(a: float) -> float:
    while a <= -90.0:
        a += 180.0
    while a > 90.0:
        a -= 180.0
    return a

def obb_to_polygon(cx, cy, w, h, angle_deg):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    hw, hh = w / 2.0, h / 2.0
    rect = np.array([[-hw, -hh],[ hw, -hh],[ hw,  hh],[-hw,  hh]], dtype=np.float32)
    R = np.array([[c, -s],[s,  c]], dtype=np.float32)
    pts = rect @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts

def _signed_area(poly: np.ndarray) -> float:
    x = poly[:,0]; y = poly[:,1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly[::-1].copy() if _signed_area(poly) < 0 else poly

def poly_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    p1 = _ensure_ccw(poly1.astype(np.float32))
    p2 = _ensure_ccw(poly2.astype(np.float32))
    a1 = abs(cv2.contourArea(p1)); a2 = abs(cv2.contourArea(p2))
    if a1 <= 0.0 or a2 <= 0.0: return 0.0
    inter, _ = cv2.intersectConvexConvex(p1, p2)
    if inter <= 0.0: return 0.0
    return float(inter / max(a1 + a2 - inter, 1e-9))

def canonicalize_xywhr(cx, cy, w, h, ang_deg):
    if w < h:
        w, h = h, w
        ang_deg = normalize_angle_deg(ang_deg + 90.0)
    ang_deg = normalize_angle_deg(ang_deg)
    return float(cx), float(cy), float(w), float(h), float(ang_deg)

def alt_rotate_xywhr(b):
    cx, cy, w, h, a = b
    return (cx, cy, h, w, normalize_angle_deg(a + 90.0))

def equiv_iou(bi, bj):
    Pi = obb_to_polygon(*bi)
    Pj = obb_to_polygon(*bj)
    i0 = poly_iou(Pi, Pj)
    Pj_alt = obb_to_polygon(*alt_rotate_xywhr(bj))
    i1 = poly_iou(Pi, Pj_alt)
    Pi_alt = obb_to_polygon(*alt_rotate_xywhr(bi))
    i2 = poly_iou(Pi_alt, Pj)
    return max(i0, i1, i2)

def near_orthogonal_gate(bi, bj, center_k=0.5, area_tol=0.40, ang_tol=20.0):
    (cxi, cyi, wi, hi, ai) = bi
    (cxj, cyj, wj, hj, aj) = bj
    si = max(1e-6, np.sqrt(wi * hi))
    sj = max(1e-6, np.sqrt(wj * hj))
    s  = 0.5 * (si + sj)
    d  = np.hypot(cxi - cxj, cyi - cyj) / s
    if d > center_k: return False
    ai_area = wi * hi; aj_area = wj * hj
    if ai_area <= 0 or aj_area <= 0: return False
    rel = abs(ai_area - aj_area) / max(ai_area, aj_area)
    if rel > area_tol: return False
    dang = abs((ai - aj + 180.0) % 180.0 - 90.0)
    if dang > ang_tol: return False
    return True

def _xywhr_to_aabb(xywhr_list):
    aabbs = []
    for (cx, cy, w, h, ang) in xywhr_list:
        poly = obb_to_polygon(cx, cy, w, h, ang)
        x1, y1 = np.min(poly, axis=0)
        x2, y2 = np.max(poly, axis=0)
        aabbs.append([x1, y1, x2, y2])
    return torch.tensor(aabbs, dtype=torch.float32)

def merge_rotated_nms_fallback(xywhr_list, confs, classes, iou_thr=0.5, per_class=True,
                               aabb_stage=True, aabb_iou=0.9,
                               center_k=0.5, area_tol=0.40, ang_tol=20.0):
    n = len(xywhr_list)
    if n == 0: return [], [], []
    # 規約化
    xywhr_norm = []
    for (cx, cy, w, h, a) in xywhr_list:
        xywhr_norm.append(canonicalize_xywhr(cx, cy, w, h, a))
    xywhr_list = xywhr_norm
    keep = list(range(n))
    if aabb_stage:
        boxes_aabb = _xywhr_to_aabb(xywhr_list)
        scores = torch.tensor(confs, dtype=torch.float32)
        keep_idx = tv_nms(boxes_aabb, scores, aabb_iou)
        keep = keep_idx.cpu().numpy().tolist()
    xywhr = [xywhr_list[i] for i in keep]
    confs_k = [confs[i] for i in keep]
    cls_k   = [classes[i] for i in keep]
    order = np.argsort(-np.asarray(confs_k))
    used = np.zeros(len(order), dtype=bool)
    final = []
    for oi in order:
        if used[oi]: continue
        final.append(oi); used[oi] = True
        bi = xywhr[oi]
        for oj in order:
            if used[oj]: continue
            if per_class and (cls_k[oi] != cls_k[oj]): continue
            bj = xywhr[oj]
            if (equiv_iou(bi, bj) >= iou_thr) or near_orthogonal_gate(bi, bj, center_k, area_tol, ang_tol):
                used[oj] = True
    out_boxes  = [xywhr[i] for i in final]
    out_scores = [confs_k[i] for i in final]
    out_cls    = [cls_k[i]   for i in final]
    return out_boxes, out_scores, out_cls

def merge_rotated_nms(xywhr_list, confs, classes, iou_thr=0.5):
    return merge_rotated_nms_fallback(xywhr_list, confs, classes, iou_thr=iou_thr, per_class=True)

class TTATransform:
    def __init__(self, name: str): self.name = name
    def apply_img(self, img: np.ndarray) -> np.ndarray:
        if self.name == "identity": return img
        if self.name == "hflip":    return cv2.flip(img, 1)
        if self.name == "vflip":    return cv2.flip(img, 0)
        if self.name == "rot90":    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.name == "rot180":   return cv2.rotate(img, cv2.ROTATE_180)
        if self.name == "rot270":   return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        raise ValueError(self.name)
    def invert_box(self, cx, cy, w, h, ang_deg, orig_w, orig_h):
        if self.name == "identity":
            pass
        elif self.name == "hflip":
            cx = orig_w - cx; ang_deg = normalize_angle_deg(180.0 - ang_deg)
        elif self.name == "vflip":
            cy = orig_h - cy; ang_deg = normalize_angle_deg(-ang_deg)
        elif self.name == "rot90":
            x_p, y_p = cx, cy
            cx, cy = orig_w - y_p, x_p; w, h = h, w
            ang_deg = normalize_angle_deg(ang_deg - 90.0)
        elif self.name == "rot180":
            cx = orig_w - cx; cy = orig_h - cy
            ang_deg = normalize_angle_deg(ang_deg - 180.0)
        elif self.name == "rot270":
            x_p, y_p = cx, cy
            cx, cy = y_p, orig_h - x_p; w, h = h, w
            ang_deg = normalize_angle_deg(ang_deg + 90.0)
        return cx, cy, w, h, ang_deg

def get_tta_list(mode: str):
    if mode == "none":  return [TTATransform("identity")]
    if mode == "flips": return [TTATransform(n) for n in ["identity", "hflip", "vflip"]]
    if mode == "rot90": return [TTATransform(n) for n in ["identity", "rot90", "rot180", "rot270"]]
    if mode == "all":   return [TTATransform(n) for n in ["identity", "hflip", "vflip", "rot90", "rot270"]]
    raise ValueError(f"Unknown TTA mode: {mode}")

# ---- ここだけ小改造：xywhr と poly の両方を返す ----
def predict_obb_with_tta(model: YOLO, img_bgr: np.ndarray, conf: float, iou: float,
                         imgsz: int | None, tta_mode: str, tta_merge_iou: float):
    H0, W0 = img_bgr.shape[:2]
    tta_list = get_tta_list(tta_mode)
    xywhr_all, cls_all, conf_all = [], [], []

    for t in tta_list:
        img_t = t.apply_img(img_bgr)
        res = model.predict(source=img_t, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        if getattr(res, "obb", None) is not None and res.obb is not None and res.obb.xywhr is not None:
            xywhr = res.obb.xywhr.cpu().numpy()  # (N,5) angle is rad in Ultralytics
            if getattr(res.obb, "cls", None) is not None:
                cls_arr = res.obb.cls.cpu().numpy().astype(int)
            elif getattr(res, "boxes", None) is not None and getattr(res.boxes, "cls", None) is not None:
                cls_arr = res.boxes.cls.cpu().numpy().astype(int)
            else:
                cls_arr = np.full(len(xywhr), -1, dtype=int)
            if getattr(res.obb, "conf", None) is not None:
                conf_arr = res.obb.conf.cpu().numpy().astype(float)
            elif getattr(res, "boxes", None) is not None and getattr(res.boxes, "conf", None) is not None:
                conf_arr = res.boxes.conf.cpu().numpy().astype(float)
            else:
                conf_arr = np.zeros(len(xywhr), dtype=float)

            n = len(xywhr)
            if len(cls_arr) != n:
                cls_arr = np.pad(cls_arr, (0, max(0, n-len(cls_arr))), constant_values=-1)[:n]
            if len(conf_arr) != n:
                conf_arr = np.pad(conf_arr, (0, max(0, n-len(conf_arr))), constant_values=0.0)[:n]

            for k in range(n):
                cx, cy, ww, hh, ang = xywhr[k]
                ang_deg = np.degrees(ang) if abs(ang) <= 3.2 else ang
                cx2, cy2, w2, h2, a2 = t.invert_box(cx, cy, ww, hh, ang_deg, W0, H0)
                cx2, cy2, w2, h2, a2 = canonicalize_xywhr(cx2, cy2, w2, h2, a2)
                w2 = max(w2, 1e-3); h2 = max(h2, 1e-3)
                xywhr_all.append((float(cx2), float(cy2), float(w2), float(h2), float(a2)))
                cls_all.append(int(cls_arr[k])); conf_all.append(float(conf_arr[k]))
        else:
            if getattr(res, "boxes", None) is not None and res.boxes is not None and res.boxes.xyxy is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.full(len(xyxy), -1)
                confs = res.boxes.conf.cpu().numpy().tolist() if res.boxes.conf is not None else [0.0]*len(xyxy)
                for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w, h   = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
                    cx2, cy2, w2, h2, a2 = t.invert_box(cx, cy, w, h, 0.0, W0, H0)
                    cx2, cy2, w2, h2, a2 = canonicalize_xywhr(cx2, cy2, w2, h2, a2)
                    w2 = max(w2, 1e-3); h2 = max(h2, 1e-3)
                    xywhr_all.append((float(cx2), float(cy2), float(w2), float(h2), float(a2)))
                    cls_all.append(int(c)); conf_all.append(float(s))

    # マージ（回転NMS）
    xywhr_merged, conf_merged, cls_merged = merge_rotated_nms(xywhr_all, conf_all, cls_all, iou_thr=tta_merge_iou)
    # 便利用に polygon も返す
    polys = [obb_to_polygon(cx, cy, w, h, ang) for (cx, cy, w, h, ang) in xywhr_merged]
    return polys, cls_merged, conf_merged, xywhr_merged
