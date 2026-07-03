#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 OBB: ハイパラをYAMLで管理し、foldごとに train/val/test を実行するランチャー
- train: セクションに「リスト値」があれば自動で全組合せスイープ（imgsz など）
- models: に複数書けば各モデルもスイープ

使い方例:
  python train_obb_cv.py \
    --root dat/v250903 \
    --folds 5 \
    --cfg hyperparams.yaml
"""

import argparse, csv, subprocess, sys
from pathlib import Path
import yaml
from itertools import product
import importlib
from ultralytics import YOLO 

def to_cli_kv(key, val):
    # YOLO CLI に渡す "key=value"
    if isinstance(val, bool):
        sval = "True" if val else "False"
    elif isinstance(val, (int, float)):
        sval = str(val)
    elif isinstance(val, (list, tuple)):
        # ここには来ない（スイープでスカラー化する）が、安全のため文字列化
        sval = ",".join(str(x) for x in val)
    else:
        sval = str(val)
    if any(c.isspace() for c in sval):
        return f'{key}="{sval}"'
    return f"{key}={sval}"

def run_cmd(cmd):
    print("\n[CMD]", cmd)
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def scalar_or_default(d, key, default):
    v = d.get(key, default)
    if isinstance(v, (list, tuple)):
        # リストの場合は最初の要素を表示用途のデフォルトに
        return v[0]
    return v

def make_sweep_combinations(train_hp: dict):
    """
    train_hp 内のリスト/タプルを全てスイープ軸にする。
    例) {"imgsz":[1024,1280], "epochs":150, "cos_lr":True}
        -> [{"imgsz":1024}, {"imgsz":1280}]
    複数キーがリストなら直積を取る。
    """
    sweep_keys = []
    sweep_vals = []
    for k, v in train_hp.items():
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            sweep_keys.append(k)
            sweep_vals.append(list(v))
    if not sweep_keys:
        return [dict()]  # スイープ無し
    combos = []
    for values in product(*sweep_vals):
        combos.append({k: v for k, v in zip(sweep_keys, values)})
    return combos

def name_suffix_from_combo(combo: dict, default_imgsz=None, default_epochs=None):
    """
    既存の e{epochs}_i{imgsz} に加えて、
    他にもスイープキーがあれば _key{val} を付与。
    """
    # 基本
    imgsz = combo.get("imgsz", default_imgsz)
    epochs = combo.get("epochs", default_epochs)
    parts = []
    if epochs is not None:
        parts.append(f"e{epochs}")
    if imgsz is not None:
        parts.append(f"i{imgsz}")
    # その他
    for k, v in sorted(combo.items()):
        if k in ("imgsz", "epochs"):
            continue
        parts.append(f"{k}{v}")
    return "_".join(parts) if parts else "exp"

def latest_epoch_from_results(run_dir: Path):
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    last_epoch = None
    try:
        with open(results_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if not row:
                    continue
                try:
                    last_epoch = int(float(row[0].strip()))
                except (ValueError, IndexError):
                    continue
    except OSError:
        return None
    return last_epoch

def train_reached_target_epochs(run_dir: Path, epochs):
    try:
        target_epochs = int(float(epochs))
    except (TypeError, ValueError):
        return False

    last_epoch = latest_epoch_from_results(run_dir)
    return last_epoch is not None and (last_epoch + 1) >= target_epochs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="foldK が並ぶ親ディレクトリ (例: dat/v250903)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cfg", required=True, help="ハイパラYAML（下のテンプレ参照）")
    ap.add_argument("--start_fold", type=int, default=0, help="このfoldから開始")
    ap.add_argument("--end_fold", type=int, default=None, help="このfoldまで（含む）。未指定なら folds-1")
    ap.add_argument("--augmod", default=None,
                    help="外部Albumentationsモジュール（build_train_aug() を持つ）")
    ap.add_argument("--resume", action="store_true",
                    help="既存runのlast.ptから再開し、完了済みrunはスキップ")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cfg_path = Path(args.cfg).resolve()
    assert cfg_path.exists(), f"cfg not found: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 期待フィールド
    task        = cfg.get("task", "obb")
    train_hp    = cfg.get("train", {})            # yolo train に渡す key:val
    val_hp      = cfg.get("val",   {})            # yolo val   に渡す key:val
    test_hp     = cfg.get("test",  {"split":"test"})
    # ループ制御
    models      = cfg.get("models", [])
    project     = cfg.get("project", "runs_obb")
    name_prefix = cfg.get("name_prefix", "fold")
    device      = cfg.get("device", 0)

    # デフォ名用：epochs/imgsz の既定（スカラー or リスト先頭）
    default_epochs = scalar_or_default(train_hp, "epochs", 100)
    default_imgsz  = scalar_or_default(train_hp, "imgsz", 1024)
    seed_default   = scalar_or_default(train_hp, "seed", 0)

    # fold 範囲
    start_k = args.start_fold
    end_k   = args.end_fold if args.end_fold is not None else args.folds - 1

    # モデルが単数なら配列化
    if not models:
        models = [cfg.get("model", "yolo11n-obb.pt")]

    # スイープ組合せ（train_hp 内のリスト全てが対象）
    combos = make_sweep_combinations(train_hp)  # 例: [{"imgsz":1024}, {"imgsz":1280}, ...] など

    aug_list = None
    if args.augmod:
        try:
            augmod = importlib.import_module(args.augmod)
            aug_list = augmod.build_train_aug()
            print(f"[AUG] Loaded Albumentations from {args.augmod}")
        except Exception as e:
            print(f"[AUG] WARNING: failed to load {args.augmod}: {e}\n"
                  f"      -> continue WITHOUT custom Albumentations.")

    for k in range(start_k, end_k + 1):
        fold_dir = root / f"fold{k}"
        data_yaml = fold_dir / "data.yaml"
        assert data_yaml.exists(), f"missing: {data_yaml}"

        for model in models:
            for combo in combos:
                # この実験用の train 引数を確定（スイープ値で上書き）
                this_train = dict(train_hp)  # ベース
                this_train.update(combo)     # スイープ値で上書き

                # 名称にスイープ内容を反映
                epochs = scalar_or_default(this_train, "epochs", default_epochs)
                imgsz  = scalar_or_default(this_train, "imgsz",  default_imgsz)
                seed   = scalar_or_default(this_train, "seed",   seed_default)
                suffix = name_suffix_from_combo(combo, default_imgsz=imgsz, default_epochs=epochs)
                exp_name = f"{name_prefix}{k}_{Path(model).stem}_{suffix}"
                run_dir = Path(project) / exp_name
                weights_dir = run_dir / "weights"
                best = weights_dir / "best.pt"
                last = weights_dir / "last.pt"
                done_marker = run_dir / "train_cv.done"

                print(f"\n[PY] yolo {task} train  (with augmentations={aug_list is not None})")

                if (args.resume and done_marker.exists()
                        and train_reached_target_epochs(run_dir, epochs)):
                    print(f"[RESUME] skip completed run: {run_dir}")
                    continue

                resume_from_last = False
                skip_train = False
                if args.resume:
                    if last.exists() and not train_reached_target_epochs(run_dir, epochs):
                        resume_from_last = True
                        print(f"[RESUME] resume training from: {last}")
                    elif best.exists():
                        skip_train = True
                        print(f"[RESUME] training already has weights; run validation only: {best}")

                if not skip_train:
                    y = YOLO(str(last) if resume_from_last else model)  # OBB対応の重み/モデルをロード

                    # train() に重複引数を渡さないように除去
                    safe_train = dict(this_train)
                    for k_rm in ["imgsz", "data", "project", "name", "device", "model", "resume"]:
                        safe_train.pop(k_rm, None)

                    train_kwargs = dict(
                        data=str(data_yaml),
                        project=project,
                        name=exp_name,
                        device=device,
                        imgsz=imgsz,
                        augmentations=aug_list,
                        **safe_train,
                    )
                    if resume_from_last:
                        train_kwargs["resume"] = True

                    y.train(**train_kwargs)

                # val on val split
                print(f"[PY] yolo {task} val (val split)")
                eval_weights = best if best.exists() else last
                assert eval_weights.exists(), f"missing weights for validation: {eval_weights}"
                y_best = YOLO(str(eval_weights))
                y_best.val(
                    data=str(data_yaml),
                    imgsz=imgsz,
                    **val_hp
                )
                done_marker.write_text("done\n", encoding="utf-8")

if __name__ == "__main__":
    main()

