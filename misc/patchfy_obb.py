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

Key rules:
  * A polygon is assigned to a patch if its centroid lies within the patch’s
    inner area (the patch shrunk by --inner_margin on all sides).
  * Polygons are shifted to patch-local coords, clipped to [0, patch_size],
    normalized back to [0,1], and dropped if their post-clip area < --min_area.
  * Optionally keep negative patches (no objects) with --keep_neg and control
    their ratio with --max_neg_ratio.

Example:
  python patchify_yolo_obb_polygons.py \
      -i /data/images -l /data/labels -o /out/obb_patches \
      -p 1280 -s 384 --inner_margin 0.05 --min_area 32 --keep_neg --max_neg_ratio 2.0

Notes:
  - Supported image suffixes: .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp
  - Grayscale images are auto-converted to 3 channels to keep training consistent.
  - If a label file is missing, that image is treated as all-negative unless
    --skip_unlabeled is set.
"""

import argparse
from pathlib import Path
import os, sys, math
from typing import List, Tuple
import cv2, numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------- geometry helpers ----------

def _poly_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _ensure_ccw(P: np.ndarray) -> np.ndarray:
    cross = np.cross(P[1]-P[0], P[2]-P[1])
    return P if cross >= 0 else P[::-1]

def _edge_lengths(P):
    return np.array([
        np.linalg.norm(P[(i+1)%4]-P[i]) for i in range(4)
    ],float)

def _is_oob_point(p):
    return (p[0]<0 or p[0]>1 or p[1]<0 or p[1]>1)

def _feasible_t_interval_for_point(q,d):
    t_low,t_high=-np.inf,np.inf
    for i in range(2):
        if abs(d[i])<1e-12:
            if q[i]<0 or q[i]>1:return None
        else:
            t0=(0-q[i])/d[i];t1=(1-q[i])/d[i]
            lo,hi=(t0,t1) if t0<=t1 else (t1,t0)
            t_low=max(t_low,lo);t_high=min(t_high,hi)
    return None if t_low>t_high else (t_low,t_high)

def _candidate_t_to_touch_boundary(p,d):
    c=[]
    for i in range(2):
        if abs(d[i])>1e-12:
            c+=[(0-p[i])/d[i],(1-p[i])/d[i]]
    return [t for t in c if np.isfinite(t)]

def _move_pair_along_long_edge(P,i,area_min=1e-6):
    L=_edge_lengths(P)
    e_prev,e_next=L[(i-1)%4],L[i%4]
    if e_prev>=e_next:
        d=P[i]-P[(i-1)%4];j=(i+1)%4
    else:
        d=P[(i+1)%4]-P[i];j=(i-1)%4
    n=np.linalg.norm(d)
    if n<1e-12:return False,P
    d=d/n;p=P[i].copy();q=P[j].copy()
    Ip=_feasible_t_interval_for_point(p,d);Iq=_feasible_t_interval_for_point(q,d)
    if Ip is None or Iq is None:return False,P
    t_low=max(Ip[0],Iq[0]);t_high=min(Ip[1],Iq[1])
    if t_low>t_high:return False,P
    cand=[t for t in _candidate_t_to_touch_boundary(p,d) if t_low-1e-12<=t<=t_high+1e-12]
    t=min(cand,key=abs) if cand else (t_low if abs(t_low)<abs(t_high) else t_high)
    v=t*d;P2=P.copy();P2[i]+=v;P2[j]+=v
    P2=np.clip(P2,0,1)
    if _poly_area(P2)<area_min:return False,P
    return True,P2

def slide_clip_normalized(P01,area_min01=1e-6):
    P=_ensure_ccw(P01.copy())
    if not any(_is_oob_point(P[k]) for k in range(4)):
        return P if _poly_area(P)>=area_min01 else None
    for _ in range(4):
        oob=next((k for k in range(4) if _is_oob_point(P[k])),None)
        if oob is None:break
        ok,Pn=_move_pair_along_long_edge(P,oob,area_min01)
        if not ok:return None
        P=Pn
    return _ensure_ccw(P) if _poly_area(P)>=area_min01 else None

def clip_quad_inside_patch(pts_px,P:int,mode="slide",area_min_px=16.):
    if mode=="clip":
        Q=np.clip(pts_px,0,float(P));a=_poly_area(Q/P)
        return Q if a*(P**2)>=area_min_px else None
    Q01=slide_clip_normalized(pts_px/float(P),area_min_px/(P**2))
    return None if Q01 is None else Q01*float(P)

# ---------- IO and patch main ----------

def find_images(root:Path)->List[Path]:
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS])

def read_label(lbl,W,H):
    out=[]
    if not lbl.exists():return out
    for line in lbl.read_text().splitlines():
        sp=line.split();
        if len(sp)!=9:continue
        cls=int(sp[0]);pts=np.array(list(map(float,sp[1:])),float).reshape(4,2)
        abs_=np.stack([pts[:,0]*W,pts[:,1]*H],1)
        out.append((cls,abs_))
    return out

def poly_centroid(pts):
    return float(pts[:,0].mean()),float(pts[:,1].mean())

def save_label(path,items,P):
    with open(path,'w') as f:
        for cls,pts in items:
            n=np.clip(pts/P,0,1)
            f.write(str(cls)+' '+' '.join(f"{v:.6f}" for v in n.reshape(-1))+"\n")

def process_one(img_path,rel,lbl_root,out_root,P,S,inner_margin,min_area,keep_neg,max_neg,force_ext,dry_run,clip_mode):
    lbl_path=lbl_root/rel.with_suffix('.txt')
    img=cv2.imdecode(np.fromfile(str(img_path),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if img is None:return(0,0)
    H,W=img.shape[:2]
    if img.ndim==2:img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    objs=read_label(lbl_path,W,H)
    cent=[(c,p,*poly_centroid(p)) for c,p in objs]
    pos=neg=0
    n_rows=max(1,math.ceil((H-P)/S)+1)
    n_cols=max(1,math.ceil((W-P)/S)+1)
    max_negs=math.ceil(max_neg*max(1,len(objs))) if keep_neg else 0
    ext=(force_ext or img_path.suffix[1:])
    for r in range(n_rows):
        y0=min(r*S,max(0,H-P));y1=y0+P
        for c in range(n_cols):
            x0=min(c*S,max(0,W-P));x1=x0+P
            margin=int(inner_margin*P)
            ix0,iy0,ix1,iy1=x0+margin,y0+margin,x1-margin,y1-margin
            kept=[]
            for cls,pts,cx,cy in cent:
                if not(ix0<=cx<ix1 and iy0<=cy<iy1):continue
                local=pts.copy();local[:,0]-=x0;local[:,1]-=y0
                local=clip_quad_inside_patch(local,P,mode=clip_mode,area_min_px=min_area)
                if local is None:continue
                kept.append((cls,local))
            relname=f"{rel.as_posix()}__r{r}_c{c}"
            out_img=out_root/'images'/f"{relname}.{ext}"
            out_lbl=out_root/'labels'/f"{relname}.txt"
            if not kept:
                if keep_neg and neg<max_negs and not dry_run:
                    out_img.parent.mkdir(parents=True,exist_ok=True);out_lbl.parent.mkdir(parents=True,exist_ok=True)
                    tile=img[y0:y1,x0:x1];cv2.imencode(f'.{ext}',tile)[1].tofile(str(out_img));open(out_lbl,'w').close()
                neg+=1;continue
            if not dry_run:
                out_img.parent.mkdir(parents=True,exist_ok=True);out_lbl.parent.mkdir(parents=True,exist_ok=True)
                tile=img[y0:y1,x0:x1];cv2.imencode(f'.{ext}',tile)[1].tofile(str(out_img));save_label(out_lbl,kept,P)
            pos+=1
    return pos,neg

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('-i','--img_dir',required=True)
    ap.add_argument('-l','--label_dir',required=True)
    ap.add_argument('-o','--out_dir',required=True)
    ap.add_argument('-p','--patch_size',type=int,default=1280)
    ap.add_argument('-s','--stride',type=int,default=384)
    ap.add_argument('--inner_margin',type=float,default=0.1)
    ap.add_argument('--min_area',type=float,default=16.)
    ap.add_argument('--keep_neg',action='store_true')
    ap.add_argument('--max_neg_ratio',type=float,default=1.)
    ap.add_argument('--ext',default=None)
    ap.add_argument('--clip_mode',choices=['slide','clip'],default='slide')
    ap.add_argument('--dry_run',action='store_true')
    args=ap.parse_args()
    img_root, lbl_root, out_root = Path(args.img_dir), Path(args.label_dir), Path(args.out_dir)
    (out_root/'images').mkdir(parents=True,exist_ok=True)
    (out_root/'labels').mkdir(parents=True,exist_ok=True)
    total_pos=total_neg=0
    for img_path in find_images(img_root):
        rel=img_path.relative_to(img_root).with_suffix('')
        pos,neg=process_one(img_path,rel,lbl_root,out_root,args.patch_size,args.stride,args.inner_margin,args.min_area,args.keep_neg,args.max_neg_ratio,args.ext,args.dry_run,args.clip_mode)
        total_pos+=pos;total_neg+=neg
    print(f"[DONE] Pos={total_pos} Neg={total_neg}")

if __name__=='__main__':
    main()
