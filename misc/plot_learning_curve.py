#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultralytics YOLO(OBB) の learning curve 可視化スクリプト

- project 配下の run ディレクトリを走査
- 各 run の results.csv を読み、指定メトリクスをプロット
- 列名は候補から自動検出（Ultralytics 版差に対応）
- run名の prefix / 正規表現で対象を絞り込み
- 出力: PNG（オーバーレイ図・run個別図）、CSV（抽出メトリクスの縦持ちまとめ）

使い方例:
  python plot_learning_curves.py \
    --project runs_obb \
    --out_dir reports_curves \
    --name_prefix exp005_hsv0 \
    --metrics mAP50-95 mAP50 train/box_loss train/cls_loss lr

  # 正規表現で絞る場合
  python plot_learning_curves.py -p runs_obb -o reports_curves \
    --name_regex "^exp005_hsv[01]_yolo11m-obb_e200_i1280_hsv_v(0\.5|0\.6|0\.65)$" \
    --metrics mAP50-95 mAP50
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# ===== 列名候補（自動検出） =====
CANDIDATES = {
    "epoch": ["epoch", "Epoch", "epochs"],
    "mAP50-95": [
        "metrics/mAP50-95(OBB)", "metrics/mAP50-95(B)", "metrics/mAP50-95",
        "mAP50-95(OBB)", "mAP50-95(B)", "mAP50-95"
    ],
    "mAP50": [
        "metrics/mAP50(OBB)", "metrics/mAP50(B)", "metrics/mAP50",
        "mAP50(OBB)", "mAP50(B)", "mAP50"
    ],
    # 代表的な学習損失（名前は版で差あり）
    "train/box_loss": ["train/box_loss", "train/box", "box_loss", "obb_loss", "train/obb_loss"],
    "train/cls_loss": ["train/cls_loss", "train/cls", "cls_loss"],
    "train/dfl_loss": ["train/dfl_loss", "train/dfl", "dfl_loss"],
    "val/box_loss":   ["val/box_loss", "val/box", "val_obb_loss"],
    "val/cls_loss":   ["val/cls_loss", "val/cls"],
    "val/dfl_loss":   ["val/dfl_loss", "val/dfl"],
    "lr":             ["lr", "lr0", "lr_0"],
}

def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    # 小文字対応
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def resolve_metric_column(df: pd.DataFrame, user_key: str) -> str | None:
    """
    user_key が candidates 辞書のキーなら候補から探す。
    そうでなければ、そのまま列名として探す（カスタム列に対応）。
    """
    if user_key in CANDIDATES:
        return find_col(df, CANDIDATES[user_key])
    # そのまま列名で（大文字小文字ゆらぎも吸収）
    if user_key in df.columns:
        return user_key
    lower = {c.lower(): c for c in df.columns}
    if user_key.lower() in lower:
        return lower[user_key.lower()]
    return None

def load_args_hints(run_dir: Path) -> Dict:
    """タイトルに入れるために args.yaml から少し拾う（あれば）"""
    out = {}
    for fn in ["args.yaml", "hyp.yaml"]:
        p = run_dir / fn
        if p.exists():
            try:
                y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                out.update(y)
            except Exception:
                pass
    hints = {}
    for k in ["model", "imgsz", "hsv_v", "hsv_s", "epochs", "seed"]:
        hints[k] = out.get(k)
    return hints

def plot_overlay(metric: str, runs: List[Dict], out_png: Path):
    plt.figure()  # 1図1プロット
    for r in runs:
        df = r["df"]
        ep = r["epoch_col"]
        mc = r["metric_cols"][metric]
        if mc is None or ep is None:
            continue
        try:
            plt.plot(df[ep].values, df[mc].values, label=r["label"])
        except Exception:
            continue
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_single(metric: str, run: Dict, out_png: Path):
    plt.figure()
    df = run["df"]
    ep = run["epoch_col"]
    mc = run["metric_cols"][metric]
    if mc is None or ep is None:
        plt.close(); return
    plt.plot(df[ep].values, df[mc].values)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    label = run["label"]
    plt.title(f"{metric} - {label}")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", "-p", required=True)
    ap.add_argument("--out_dir", "-o", default="reports_curves")
    ap.add_argument("--name_prefix", "-n", default="", help="前方一致で run を絞る（空なら全件）")
    ap.add_argument("--name_regex",  default="", help="正規表現で run を絞る（prefixより優先）")
    ap.add_argument("--metrics", nargs="+", default=["mAP50-95", "mAP50", "train/box_loss", "train/cls_loss", "train/dfl_loss", "lr"])
    ap.add_argument("--overlay_only", action="store_true", help="個別図の保存を省略して、オーバーレイ図だけ保存")
    args = ap.parse_args()

    project = Path(args.project)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 対象 run を集める
    runs = []
    for d in sorted(project.glob("*")):
        if not d.is_dir(): continue
        if args.name_regex:
            if not re.search(args.name_regex, d.name): continue
        elif args.name_prefix:
            if not d.name.startswith(args.name_prefix): continue

        csvp = d / "results.csv"
        if not csvp.exists(): continue
        try:
            df = pd.read_csv(csvp)
        except Exception:
            continue
        if df.empty: continue

        # epoch 列
        ep = None
        for c in CANDIDATES["epoch"]:
            if c in df.columns: ep = c; break
        if ep is None:
            # epochが無ければ0..N-1を疑似epochとする
            df = df.copy()
            df["epoch_auto"] = range(len(df))
            ep = "epoch_auto"

        # 各メトリクスの実列名を解決
        metric_cols = {m: resolve_metric_column(df, m) for m in args.metrics}

        # runの見出し用ラベル（短く）
        hints = load_args_hints(d)
        hint_txt = []
        for k in ["model","imgsz","hsv_v","hsv_s","epochs","seed"]:
            if hints.get(k) is not None:
                hint_txt.append(f"{k}={hints[k]}")
        label = d.name if not hint_txt else f"{d.name} ({', '.join(hint_txt)})"

        runs.append({
            "dir": d, "name": d.name, "df": df, "epoch_col": ep,
            "metric_cols": metric_cols, "label": label
        })

    if not runs:
        print("[WARN] 対象 run が見つかりませんでした。--name_prefix / --name_regex を確認してください。")
        return

    # 縦持ちのまとめCSVも出す（あとで表計算に便利）
    recs = []
    for r in runs:
        df = r["df"]; ep = r["epoch_col"]
        for m, col in r["metric_cols"].items():
            if col is None: continue
            sub = df[[ep, col]].copy()
            sub.columns = ["epoch", "value"]
            sub["metric"] = m
            sub["run"] = r["name"]
            recs.append(sub)
    if recs:
        cat = pd.concat(recs, ignore_index=True)
        cat.to_csv(out_dir / "learning_curves_long.csv", index=False)
        print(f"[OK] 保存: {out_dir/'learning_curves_long.csv'}")

    # メトリクスごとのオーバーレイ図
    for m in args.metrics:
        safe_m = sanitize_name(m)
        out = out_dir / f"overlay_{safe_m}.png"
        plot_overlay(m, runs, out)
        print(f"[OK] 保存: {out}")

    # run 個別図
    if not args.overlay_only:
        for r in runs:
            for m in args.metrics:
                safe_m = sanitize_name(m)
                out = out_dir / f"{r['name']}_{safe_m}.png"
                plot_single(m, r, out)
        print("[OK] 個別図も保存しました。")

if __name__ == "__main__":
    main()
