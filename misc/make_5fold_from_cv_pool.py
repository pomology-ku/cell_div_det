#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cv_pool を K-fold にクラス比維持で分割し、各 fold に train/val/test と data.yaml を生成（YOLO-OBB）

- cv_pool/images/ の再帰探索時に、先頭ディレクトリが 'train'/'val'/'test' なら自動的に1段剥がして配置（★）
  例) cv_pool/images/train/sub/img.jpg -> foldK/train/images/sub/img.jpg
"""

import argparse, random, shutil, os, sys
from pathlib import Path
from collections import defaultdict, Counter
import yaml

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
STRIP_CANDIDATES = {"train", "val", "test"}  # ★ 先頭要素として現れたら剥がす候補

def link_or_copy(src: Path, dst: Path, use_symlink: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def get_primary_class_from_yolo_obb(label_path: Path, mode: str = "first") -> str:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "NEG"
    first = None
    from collections import Counter
    freq = Counter(); order = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cid = parts[0]
            if first is None: first = cid
            freq[cid] += 1; order.append(cid)
    if not freq: return "NEG"
    if mode == "first": return first
    m = max(freq.values()); cands = {k for k,v in freq.items() if v == m}
    for cid in order:
        if cid in cands: return cid
    return first or "NEG"

def load_meta_from_yaml(yaml_path: Path):
    """source_yaml から names, nc, channels を読み取る（存在するものだけ返す）"""
    if not yaml_path.exists():
        return None, None, None
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[WARN] data.yaml を読み取れませんでした: {e}", file=sys.stderr)
        return None, None, None
    names = data.get("names", None)
    nc = data.get("nc", None)
    channels = data.get("channels", None)  # ★ 追加
    return names, nc, channels

def write_list_txt(base_dir: Path, file_path: Path, image_paths_abs: list[Path]):
    rels = [str(p.relative_to(base_dir)).replace(os.sep, "/") for p in image_paths_abs]
    file_path.write_text("\n".join(rels) + "\n", encoding="utf-8")

def collect_images_recursive(img_root: Path):
    return sorted([p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def maybe_strip_leading(rel: Path, enable: bool, extra_drop: int) -> Path:
    """★ rel の先頭を必要に応じて剥がす。"""
    parts = list(rel.parts)
    # 追加で明示的に N 層剥がす設定
    drop = max(0, extra_drop)
    if enable and parts and parts[0] in STRIP_CANDIDATES:
        drop = max(drop, 1)
    if drop > 0 and len(parts) > drop:
        parts = parts[drop:]
    return Path(*parts) if parts else Path(rel.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="データルート（cv_pool/ と test/ を含む）")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primary", choices=["first","maxfreq"], default="first", help="主ラベル決定方法")
    ap.add_argument("--use_symlink", action="store_true", help="コピーではなくシンボリックリンク（失敗時はコピー）")
    ap.add_argument("--source_yaml", type=str, default="data.yaml", help="names/nc を継承する YAML（root 直下を想定）")
    # ★ 先頭ディレクトリ剥がしの制御
    ap.add_argument("--strip_prefix", action="store_true", default=True,
                    help="cv_pool/images 下の先頭が train/val/test の場合に自動で1段剥がす（既定: 有効）")
    ap.add_argument("--strip_levels", type=int, default=0,
                    help="先頭から追加で剥がすディレクトリ層の数（デフォルト0）")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root).resolve()
    cv_img_dir = root/"cv_pool"/"images"
    cv_lab_dir = root/"cv_pool"/"labels"
    test_img_src = root/"test"/"images"
    test_lab_src = root/"test"/"labels"
    assert cv_img_dir.exists() and cv_lab_dir.exists(), "cv_pool/images と cv_pool/labels が必要です"
    assert test_img_src.exists() and test_lab_src.exists(), "test/images と test/labels が必要です"

    names, nc, channels = load_meta_from_yaml(Path(args.source_yaml))

    # cv_pool を収集
    cv_imgs = collect_images_recursive(cv_img_dir)

    # items = [(cls, img_abs, lab_abs, rel_from_images_stripped)]
    items = []
    for ip in cv_imgs:
        rel = ip.relative_to(cv_img_dir)
        # ★ ここで剥がす
        rel_out = maybe_strip_leading(rel, enable=args.strip_prefix, extra_drop=args.strip_levels)
        lp = cv_lab_dir / rel.with_suffix(".txt")  # 対応ラベルは元の rel で参照する（存在確認用）
        cls = get_primary_class_from_yolo_obb(lp, mode=args.primary) if lp.exists() else "NEG"
        items.append((cls, ip, lp, rel_out))

    # クラス別にシャッフル → ラウンドロビン
    by_cls = defaultdict(list)
    for rec in items: by_cls[rec[0]].append(rec)
    folds = [[] for _ in range(args.folds)]
    for cls, lst in by_cls.items():
        random.shuffle(lst)
        for i, rec in enumerate(lst):
            folds[i % args.folds].append(rec)

    # 各 fold を作成
    for k in range(args.folds):
        fold_dir = root / f"fold{k}"
        tr_img = fold_dir/"train"/"images"; tr_lab = fold_dir/"train"/"labels"
        va_img = fold_dir/"val"/"images";   va_lab = fold_dir/"val"/"labels"
        te_img = fold_dir/"test"/"images";  te_lab = fold_dir/"test"/"labels"

        val_set = folds[k]
        train_set = [rec for j, fold in enumerate(folds) if j != k for rec in fold]

        # train/val の書き出し（剥がした相対パス rel_out で配置）
        for cls, ip, lp, rel_out in train_set:
            link_or_copy(ip, tr_img/rel_out, use_symlink=args.use_symlink)
            # ラベルの書き出しは、元の lp を読み、対応する出力先も rel_out に合わせる
            dst_lab = tr_lab/rel_out.with_suffix(".txt")
            if lp.exists():
                link_or_copy(lp, dst_lab, use_symlink=args.use_symlink)
            else:
                dst_lab.parent.mkdir(parents=True, exist_ok=True)
                if not dst_lab.exists(): dst_lab.write_text("", encoding="utf-8")

        for cls, ip, lp, rel_out in val_set:
            link_or_copy(ip, va_img/rel_out, use_symlink=args.use_symlink)
            dst_lab = va_lab/rel_out.with_suffix(".txt")
            if lp.exists():
                link_or_copy(lp, dst_lab, use_symlink=args.use_symlink)
            else:
                dst_lab.parent.mkdir(parents=True, exist_ok=True)
                if not dst_lab.exists(): dst_lab.write_text("", encoding="utf-8")

        # test は共通ソースを fold 内にコピー/リンク
        test_imgs = collect_images_recursive(test_img_src)
        for ip in test_imgs:
            rel_t = ip.relative_to(test_img_src)
            # ★ test 側も、もし先頭が train/val/test なら同様に剥がす（整合のため）
            rel_t_out = maybe_strip_leading(rel_t, enable=args.strip_prefix, extra_drop=args.strip_levels)
            lp = test_lab_src / rel_t.with_suffix(".txt")
            link_or_copy(ip, te_img/rel_t_out, use_symlink=args.use_symlink)
            dst_lab = te_lab/rel_t_out.with_suffix(".txt")
            if lp.exists():
                link_or_copy(lp, dst_lab, use_symlink=args.use_symlink)
            else:
                dst_lab.parent.mkdir(parents=True, exist_ok=True)
                if not dst_lab.exists(): dst_lab.write_text("", encoding="utf-8")

        # list 作成（foldK からの相対パス）
        def collect(img_root: Path): return sorted([p for p in img_root.rglob("*") if p.suffix.lower() in IMG_EXTS])
        train_imgs_out = collect(tr_img); val_imgs_out = collect(va_img); test_imgs_out = collect(te_img)
        write_list_txt(fold_dir, fold_dir/"train.txt", train_imgs_out)
        write_list_txt(fold_dir, fold_dir/"val.txt",   val_imgs_out)
        write_list_txt(fold_dir, fold_dir/"test.txt",  test_imgs_out)

        # data.yaml
        data_yaml = {"path": str(fold_dir.resolve()), 
                     "train": str((fold_dir/"train").resolve()), 
                     "val": str((fold_dir/"val").resolve()), 
                     "test": str((fold_dir/"test").resolve())}
        if names is not None: data_yaml["names"] = names
        if nc is not None:    data_yaml["nc"] = nc
        if channels is not None: data_yaml["channels"] = channels
        (fold_dir/"data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

        print(f"Fold{k}: train={len(train_imgs_out)}  val={len(val_imgs_out)}  test={len(test_imgs_out)}")
        print("  train class dist:", Counter([c for c, *_ in train_set]))
        print("  val   class dist:", Counter([c for c, *_ in val_set]))

if __name__ == "__main__":
    main()
