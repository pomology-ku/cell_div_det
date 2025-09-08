#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 OBB: 最終モデル学習ランチャー（cvなし）
- train: セクションに「リスト値」があれば全組合せスイープ（例: imgsz など）
- models: に複数書けば各モデルもスイープ
- 学習は data.yaml の train を使用（valは内部split）。評価は test のみ実施。

使い方例:
  python train.py \
    --data dat/final_data.yaml \
    --cfg hyperparams.yaml

cfg (例):
task: obb                  # yolo サブコマンド（obb/detect等）
project: runs_obb          # YOLOの出力プロジェクト
name_prefix: final         # 実験名プレフィックス
device: 0
models:
  - yolo11n-obb.pt
  - yolo11m-obb.pt
train:
  epochs: [150, 200]
  imgsz: [1024, 1280]
  seed: 0
  cos_lr: True
  batch: 16
test:
  split: test              # ここは基本固定でOK（data.yamlにtest記載が必要）
  iou: 0.50:0.95
  save_json: True
  conf: 0.001
"""

import argparse, subprocess, sys
from pathlib import Path
import yaml
from itertools import product

def to_cli_kv(key, val):
    # YOLO CLI に渡す "key=value"
    if isinstance(val, bool):
        sval = "True" if val else "False"
    elif isinstance(val, (int, float)):
        sval = str(val)
    elif isinstance(val, (list, tuple)):
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
        return v[0]
    return v

def make_sweep_combinations(train_hp: dict):
    """
    train_hp 内のリスト/タプルを全てスイープ軸にする。
    例) {"imgsz":[1024,1280], "epochs":150} -> [{"imgsz":1024}, {"imgsz":1280}]
    複数キーがリストなら直積。
    """
    sweep_keys, sweep_vals = [], []
    for k, v in train_hp.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
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
    既存の e{epochs}_i{imgsz} に加えて、他にもスイープキーがあれば _key{val} を付与。
    """
    imgsz = combo.get("imgsz", default_imgsz)
    epochs = combo.get("epochs", default_epochs)
    parts = []
    if epochs is not None:
        parts.append(f"e{epochs}")
    if imgsz is not None:
        parts.append(f"i{imgsz}")
    for k, v in sorted(combo.items()):
        if k in ("imgsz", "epochs"):
            continue
        parts.append(f"{k}{v}")
    return "_".join(parts) if parts else "exp"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="最終学習用 data.yaml（train=cv_pool, test=test。valは未指定）")
    ap.add_argument("--cfg", required=True, help="ハイパラYAML（上のテンプレ参照）")
    args = ap.parse_args()

    data_yaml = Path(args.data).resolve()
    assert data_yaml.exists(), f"data.yaml not found: {data_yaml}"

    cfg_path = Path(args.cfg).resolve()
    assert cfg_path.exists(), f"cfg not found: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 設定の取得
    task        = cfg.get("task", "obb")
    train_hp    = cfg.get("train", {})           # yolo train に渡す key:val
    test_hp     = cfg.get("test",  {"split": "test"})
    models      = cfg.get("models", [])
    project     = cfg.get("project", "runs_obb")
    name_prefix = cfg.get("name_prefix", "final")
    device      = cfg.get("device", 0)

    # モデルが単数なら配列化
    if not models:
        models = [cfg.get("model", "yolo11n-obb.pt")]

    # デフォ名用
    default_epochs = scalar_or_default(train_hp, "epochs", 100)
    default_imgsz  = scalar_or_default(train_hp, "imgsz", 1024)
    seed_default   = scalar_or_default(train_hp, "seed", 0)

    # スイープ組合せ
    combos = make_sweep_combinations(train_hp)

    for model in models:
        for combo in combos:
            # この実験用の train 引数（スイープ値で上書き）
            this_train = dict(train_hp)
            this_train.update(combo)

            # 名称
            epochs = scalar_or_default(this_train, "epochs", default_epochs)
            imgsz  = scalar_or_default(this_train, "imgsz",  default_imgsz)
            _seed  = scalar_or_default(this_train, "seed",   seed_default)
            suffix = name_suffix_from_combo(combo, default_imgsz=imgsz, default_epochs=epochs)
            exp_name = f"{name_prefix}_{Path(model).stem}_{suffix}"

            # === train ===
            base_args = {
                "data": str(data_yaml),
                "model": model,
                "project": project,
                "name": exp_name,
                "device": device,
            }
            merged_train = {**base_args, **this_train}
            cli_train = " ".join([to_cli_kv(k, v) for k, v in merged_train.items()])
            cmd_train = f"yolo {task} train {cli_train}"
            run_cmd(cmd_train)

if __name__ == "__main__":
    main()
