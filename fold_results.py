#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習に使った YAML (train cfg) を読み、[] で指定されたスイープ軸に基づき自動集計するスクリプト

・やること
  - --train_cfg で与えたYAMLから、trainセクション内の「値がリスト」のキーをスイープ軸として抽出（例: imgsz, hsv_v, hsv_s）
  - models（または model）もスイープ軸として扱う（models がリストなら models 軸、なければ単一 model を固定値）
  - project 配下の run ディレクトリを走査し、各 run の args.yaml/hyp.yaml を読み、
      軸キー（例: model/imgsz/hsv_v/hsv_s）を取得（不足時はexp名からフォールバック抽出）
  - 各 run の results.csv から mAP50-95/mAP50 のベスト値を取得
  - 軸ごとに集計CSVを自動生成（「軸=1個」「全軸の組合せ」）
  - 期待グリッド（全組合せ）に対して、見つかった本数/不足（カバレッジ）もCSVに出す

・出力
  - summary_per_run.csv                 : 走査した run 一覧（軸の値＋指標）
  - summary_by_<axis>.csv               : 各軸ごとの平均・標準偏差（例: summary_by_model.csv, summary_by_imgsz.csv）
  - summary_by_all_axes.csv             : 全軸の組み合わせ別平均（クロス集計）
  - expected_grid_coverage.csv          : 期待グリッドと実際のカバレッジ（何本見つかったか、不足の有無）

使い方:
  python aggregate_from_train_cfg.py \
    --project runs_obb \
    --out_dir reports_sem \
    --train_cfg hyperparams.yaml \
    --name_prefix exp004_hsv    # 任意。prefixで対象実験を限定したい場合

必要: pandas, pyyaml
"""

import argparse
import re
from pathlib import Path
from itertools import product
import yaml
import pandas as pd

# ====== Ultralyticsの列候補（版差に対応） ======
MAP5095_CANDIDATES = [
    "metrics/mAP50-95(OBB)", "metrics/mAP50-95(B)", "metrics/mAP50-95",
    "mAP50-95(OBB)", "mAP50-95(B)", "mAP50-95"
]
MAP50_CANDIDATES = [
    "metrics/mAP50(OBB)", "metrics/mAP50(B)", "metrics/mAP50",
    "mAP50(OBB)", "mAP50(B)", "mAP50"
]
EPOCH_COLS = ["epoch", "Epoch", "epochs"]

# ====== 名称フォールバック（足りないときだけ使用） ======
NAME_PATTERNS = [
    re.compile(r".*_(?P<model>yolo\d+[nslmx]-obb).*_i(?P<imgsz>\d+)", re.IGNORECASE),
    re.compile(r"fold(?P<fold>\d+)[_\-]+(?P<model>[^_]+)[_\-]+e(?P<epochs>\d+)[_\-]+i(?P<imgsz>\d+)", re.IGNORECASE),
]

def load_yaml_safe(p: Path):
    try:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def get_best_metrics(df: pd.DataFrame) -> dict:
    m95_col = find_col(df, MAP5095_CANDIDATES)
    m50_col = find_col(df, MAP50_CANDIDATES)
    ep_col  = find_col(df, EPOCH_COLS)

    out = {"metric_col": m95_col, "metric50_col": m50_col, "epoch_col": ep_col}
    if m95_col is None and m50_col is None:
        return {**out,
                "best_mAP50-95": None, "best_epoch_mAP50-95": None,
                "best_mAP50": None, "best_epoch_mAP50": None}
    if m95_col is not None:
        idx = df[m95_col].astype(float).idxmax()
        out["best_mAP50-95"] = float(df.loc[idx, m95_col])
        out["best_epoch_mAP50-95"] = int(df.loc[idx, ep_col]) if ep_col else int(idx)
    else:
        out["best_mAP50-95"] = None
        out["best_epoch_mAP50-95"] = None
    if m50_col is not None:
        idx2 = df[m50_col].astype(float).idxmax()
        out["best_mAP50"] = float(df.loc[idx2, m50_col])
        out["best_epoch_mAP50"] = int(df.loc[idx2, ep_col]) if ep_col else int(idx2)
    else:
        out["best_mAP50"] = None
        out["best_epoch_mAP50"] = None
    return out

def parse_name_fallback(name: str) -> dict:
    info = {"model": None, "imgsz": None}
    for pat in NAME_PATTERNS:
        m = pat.match(name)
        if m:
            gd = m.groupdict()
            if gd.get("model"): info["model"] = gd["model"]
            if gd.get("imgsz"):
                try: info["imgsz"] = int(gd["imgsz"])
                except: pass
            break
    return info

def load_run_axes(run_dir: Path) -> dict:
    """args.yaml/hyp.yaml から軸候補を拾い、無ければ名前から補完"""
    argsy = load_yaml_safe(run_dir / "args.yaml")
    hypy  = load_yaml_safe(run_dir / "hyp.yaml")
    merged = {**hypy, **argsy}
    out = {}
    # よく使うキー
    if "model" in merged: out["model"] = merged.get("model")
    if "imgsz" in merged: out["imgsz"] = merged.get("imgsz")
    if "img_size" in merged and "imgsz" not in out: out["imgsz"] = merged.get("img_size")
    if "hsv_v" in merged: out["hsv_v"] = merged.get("hsv_v")
    if "hsv_s" in merged: out["hsv_s"] = merged.get("hsv_s")
    if "degrees" in merged: out["degrees"] = merged.get("degrees")
    # 型整形
    for k in ["imgsz","degrees"]:
        try:
            if k in out and out[k] is not None:
                out[k] = int(out[k])
        except Exception:
            pass
    for k in ["hsv_v","hsv_s"]:
        try:
            if k in out and out[k] is not None:
                out[k] = float(out[k])
        except Exception:
            pass
    # フォールバック
    fb = parse_name_fallback(run_dir.name)
    if "model" not in out or out["model"] is None:
        out["model"] = fb.get("model")
    if "imgsz" not in out or out["imgsz"] is None:
        out["imgsz"] = fb.get("imgsz")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", "-p", required=True, help="Ultralyticsのプロジェクトディレクトリ（例: runs_obb）")
    ap.add_argument("--out_dir", "-o", default="reports", help="出力ディレクトリ")
    ap.add_argument("--train_cfg", "-c", required=True, help="学習に使った YAML（スイープ軸はこの中のリスト）")
    ap.add_argument("--name_prefix", "-n", default="", help="対象runの名前プレフィックス（空なら全件）")
    args = ap.parse_args()

    project = Path(args.project)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ==== 1) 学習YAMLからスイープ軸を抽出 ====
    cfg = yaml.safe_load(Path(args.train_cfg).read_text(encoding="utf-8")) or {}
    train_hp = cfg.get("train", {}) or {}
    # 軸候補: train内で「値がリスト」のキー
    sweep_axes = [k for k, v in train_hp.items() if isinstance(v, (list, tuple)) and len(v) > 0]
    # モデル軸（modelsがあればそれ、無ければ単一modelを固定値）
    models = cfg.get("models", [])
    if models:
        sweep_axes = ["model"] + sweep_axes
        expected_values = {"model": list(models)}
    else:
        # model が単一なら固定値として扱う（軸にはしない）
        expected_values = {}
    # 各軸の期待値集合
    for k in train_hp:
        v = train_hp[k]
        if isinstance(v, (list, tuple)) and len(v) > 0:
            # 数値は int/float 正規化（比較ブレ回避）
            vs = []
            for e in v:
                try:
                    if isinstance(e, str) and e.isdigit():
                        vs.append(int(e))
                    else:
                        vs.append(float(e) if isinstance(e, (int,float)) or "." in str(e) else int(e))
                except Exception:
                    vs.append(e)
            expected_values[k] = vs

    # ==== 2) run を走査して、軸値＋metrics を収集 ====
    rows = []
    for run_dir in sorted(project.glob("*")):
        if not run_dir.is_dir():
            continue
        if args.name_prefix and not run_dir.name.startswith(args.name_prefix):
            continue
        if not (run_dir / "weights" / "best.pt").exists():
            continue

        df = None
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                pass
        if df is None or df.empty:
            continue

        axes_in_run = load_run_axes(run_dir)
        best = get_best_metrics(df)

        row = {"exp_name": run_dir.name, "exp_path": str(run_dir)}
        # まずは全軸を None で用意
        for ax in sweep_axes:
            row[ax] = None

        # 値を埋める（args優先）
        for ax in sweep_axes:
            if ax in axes_in_run and axes_in_run[ax] is not None:
                row[ax] = axes_in_run[ax]
            # trainのキー名と run 側のキー名が違う場合の素直マッピング
            if row[ax] is None and ax == "imgsz" and "imgsz" in axes_in_run:
                row[ax] = axes_in_run["imgsz"]

        # metrics
        row.update({
            "best_mAP50-95": best.get("best_mAP50-95"),
            "best_epoch_mAP50-95": best.get("best_epoch_mAP50-95"),
            "best_mAP50": best.get("best_mAP50"),
            "best_epoch_mAP50": best.get("best_epoch_mAP50"),
        })
        rows.append(row)

    if not rows:
        print("[WARN] 対象runが見つかりませんでした。--name_prefix や --project を確認してください。")
        return

    df = pd.DataFrame(rows)

    # 型の揺れを整える（比較しやすく）
    for ax in sweep_axes:
        if ax in df.columns:
            # 数値っぽければ数値化
            try:
                df[ax] = pd.to_numeric(df[ax], errors="ignore")
            except Exception:
                pass

    # 保存：全runの一覧
    per_run_csv = out_dir / "summary_per_run.csv"
    df.to_csv(per_run_csv, index=False)
    print(f"[OK] 保存: {per_run_csv}")

    # スコア列（mAP50-95優先、無ければmAP50）
    df["score_for_group"] = df["best_mAP50-95"].fillna(df["best_mAP50"])

    # ==== 3) 軸ごとの自動集計（単軸） ====
    for ax in sweep_axes:
        if ax not in df.columns:
            continue
        g = df.groupby(ax, dropna=False).agg(
            n=("score_for_group","count"),
            mean=("score_for_group","mean"),
            std=("score_for_group","std"),
            mean_mAP50_95=("best_mAP50-95","mean"),
            std_mAP50_95=("best_mAP50-95","std"),
            mean_mAP50=("best_mAP50","mean"),
            std_mAP50=("best_mAP50","std"),
        ).reset_index().sort_values(ax)
        out = out_dir / f"summary_by_{ax}.csv"
        g.to_csv(out, index=False)
        print(f"[OK] 保存: {out}")

    # ==== 4) 全軸の組合せ集計 ====
    if sweep_axes and all(ax in df.columns for ax in sweep_axes):
        g = df.groupby(sweep_axes, dropna=False).agg(
            n=("score_for_group","count"),
            mean=("score_for_group","mean"),
            std=("score_for_group","std"),
            mean_mAP50_95=("best_mAP50-95","mean"),
            std_mAP50_95=("best_mAP50-95","std"),
            mean_mAP50=("best_mAP50","mean"),
            std_mAP50=("best_mAP50","std"),
        ).reset_index().sort_values(sweep_axes)
        out = out_dir / "summary_by_all_axes.csv"
        g.to_csv(out, index=False)
        print(f"[OK] 保存: {out}")

    # ==== 5) 期待グリッドに対するカバレッジ ====
    # expected_values が空ならスキップ（全軸にリストが無い場合）
    if expected_values:
        keys = list(expected_values.keys())
        grid = list(product(*[expected_values[k] for k in keys]))
        recs = []
        for vals in grid:
            mask = pd.Series([True] * len(df))
            for k, v in zip(keys, vals):
                if k in df.columns:
                    mask &= (df[k] == v)
            found = int(mask.sum())
            recs.append({**{k: v for k, v in zip(keys, vals)}, "found_runs": found})
        cov = pd.DataFrame(recs)
        out_cov = out_dir / "expected_grid_coverage.csv"
        cov.to_csv(out_cov, index=False)
        print(f"[OK] 保存: {out_cov}")

    # 画面プレビュー
    print("\n=== (preview) per_run head ===")
    print(df.head(10).to_string(index=False))
    for ax in sweep_axes:
        p = out_dir / f"summary_by_{ax}.csv"
        if p.exists():
            print(f"\n=== (preview) by {ax} ===")
            print(pd.read_csv(p).to_string(index=False))
    if (out_dir / "summary_by_all_axes.csv").exists():
        print("\n=== (preview) by ALL axes ===")
        print(pd.read_csv(out_dir / "summary_by_all_axes.csv").head(20).to_string(index=False))
    if expected_values:
        print("\n=== (preview) expected grid coverage ===")
        print(pd.read_csv(out_dir / "expected_grid_coverage.csv").to_string(index=False))

if __name__ == "__main__":
    main()

