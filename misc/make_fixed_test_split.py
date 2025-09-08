#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-OBB データを images/, labels/ ディレクトリから直接探索し、
クラス比を保って test/cv_pool に分割する。

前提:
- 画像: root/images/**/xxx.ext
- ラベル: root/labels/**/xxx.txt   (画像と同じサブフォルダ・同じ stem)
- ラベル内容は YOLO-OBB 形式 (class_id x1 y1 x2 y2 x3 y3 x4 y4, 0–1正規化)

出力:
- out_root/test/{images,labels}
- out_root/cv_pool/{images,labels}
- test.txt, cv_pool.txt （出力先の画像相対パス）
- （--gen_yaml 指定時）data.yaml も生成
"""

import argparse, random, shutil, re, sys, os
from pathlib import Path
from collections import Counter, defaultdict
import yaml

WS_RE = re.compile(r"\s+")
IMG_EXTS = [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"]

def norm_space(s: str) -> str:
    return WS_RE.sub("_", s)

def normalize_rel_path(p: Path, root: Path) -> Path:
    rel = p.relative_to(root)
    return Path(*[norm_space(x) for x in rel.parts])

def get_primary_class_from_yolo_obb(label_path: Path, mode="first"):
    if (not label_path.exists()) or label_path.stat().st_size == 0:
        return "NEG"
    first, freq, order = None, Counter(), []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts=line.strip().split()
            if len(parts) < 9:
                continue
            cid = parts[0]
            if first is None: first = cid
            freq[cid]+=1; order.append(cid)
    if not freq: return "NEG"
    if mode=="first": return first
    m=max(freq.values()); cands={c for c,v in freq.items() if v==m}
    for cid in order:
        if cid in cands: return cid

def copy_with_norm(root: Path, src: Path, dst_root: Path):
    rel = src.relative_to(root)
    rel_norm = Path(*[norm_space(x) for x in rel.parts])
    dst = dst_root/rel_norm
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--ratio_test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primary", choices=["first","maxfreq"], default="first")
    ap.add_argument("--gen_yaml", action="store_true")
    args=ap.parse_args()

    random.seed(args.seed)
    root=Path(args.root).resolve(); out=Path(args.out_root).resolve()
    img_dir=root/"images"; lab_dir=root/"labels"
    assert img_dir.exists() and lab_dir.exists(), "images/ と labels/ が必要"

    # 画像探索
    img_list=[p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    items=[]
    for ip in img_list:
        # labels/.../same_stem.txt を探す
        rel=ip.relative_to(img_dir)
        lp=lab_dir/rel.with_suffix(".txt")
        cls=get_primary_class_from_yolo_obb(lp,args.primary)
        items.append((ip,lp,cls))

    # クラス比を保って分割
    by_cls=defaultdict(list)
    for it in items: by_cls[it[2]].append(it)

    test,pool=[],[]
    for cls,lst in by_cls.items():
        random.shuffle(lst)
        k=round(len(lst)*args.ratio_test)
        test+=lst[:k]; pool+=lst[k:]

    target=int(round(len(items)*args.ratio_test))
    if len(test)<target:
        movable=sorted(pool,key=lambda it:len(by_cls[it[2]]),reverse=True)
        need=min(target-len(test),len(movable))
        test+=movable[:need]; pool=movable[need:]+[]
    elif len(test)>target:
        movable=sorted(test,key=lambda it:len(by_cls[it[2]]),reverse=True)
        need=len(test)-target
        keep=movable[need:]; back=movable[:need]
        test=keep; pool+=back

    # 出力ディレクトリ
    test_img=out/"test/"; test_lab=out/"test/"
    pool_img=out/"cv_pool/"; pool_lab=out/"cv_pool/"

    test_list,pool_list=[],[]

    for ip,lp,cls in test:
        ip_out=copy_with_norm(root,ip,test_img)
        if lp.exists(): copy_with_norm(root,lp,test_lab)
        else:
            outp=test_lab/normalize_rel_path(lp,root); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text("")
        test_list.append(str(ip_out.relative_to(out)).replace(os.sep,"/"))

    for ip,lp,cls in pool:
        ip_out=copy_with_norm(root,ip,pool_img)
        if lp.exists(): copy_with_norm(root,lp,pool_lab)
        else:
            outp=pool_lab/normalize_rel_path(lp,root); outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text("")
        pool_list.append(str(ip_out.relative_to(out)).replace(os.sep,"/"))

    (out/"test.txt").write_text("\n".join(test_list)+"\n",encoding="utf-8")
    (out/"cv_pool.txt").write_text("\n".join(pool_list)+"\n",encoding="utf-8")

    if args.gen_yaml:
        data={"path":str(out),
              "train":"cv_pool/images",
              "val":"test/images"}
        (out/"data.yaml").write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding="utf-8")

    print(f"Done: test={len(test)} cv_pool={len(pool)} total={len(items)}")
    print("Test class dist:",Counter([it[2] for it in test]))
    print("Pool class dist:",Counter([it[2] for it in pool]))
    if "NEG" in by_cls: print(f"[Note] NEG images: {len(by_cls['NEG'])}")

if __name__=="__main__":
    main()

