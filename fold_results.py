#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fold_results.py (v2)

- train.yaml のリスト軸に加え、各 run の args.yaml / hyp.yaml を総当りで走査し、
  実際に値が変わっているキーを “自動軸” として追加採用（例: mosaic, shear, translate, scale, mixup, flip系 等）
- 値が2種類以上のキーのみを “軸” にするため、勝手にノイズ列が増えない
- 浮動小数は丸めて group 化（例: 0.1500001 と 0.15 を同一視）

出力は従来と同じ:
  - summary_per_run.csv
  - summary_by_<axis>.csv （自動軸含む）
  - summary_by_all_axes.csv（採用軸が2つ以上ある時）
  - expected_grid_coverage.csv（train.yaml のリスト軸のみを対象）
"""

import argparse
import re
from pathlib import Path
from itertools import product
import yaml
import pandas as pd
import math

# ====== Ultralyticsの列候補（版差に対応） ======
MAP5095_CANDIDATES = [
    "metrics/mAP50-95(OBB)", "metrics/mAP50-95(B)", "metrics/mAP50-95",
    "mAP50-95(OBB)", "mAP50-95(B)", "mAP50-95", "map"  # map=0.5:0.95
]
MAP50_CANDIDATES = [
    "metrics/mAP50(OBB)", "metrics/mAP50(B)", "metrics/mAP50",
    "mAP50(OBB)", "mAP50(B)", "mAP50", "map50"
]
RECALL_CANDIDATES = [
    "metrics/recall(OBB)", "metrics/recall(B)", "metrics/recall",
    "recall(OBB)", "recall(B)", "recall", "mr"
]
PRECISION_CANDIDATES = [
    "metrics/precision(OBB)", "metrics/precision(B)", "metrics/precision",
    "precision(OBB)", "precision(B)", "precision", "mp"
]
EPOCH_COLS = ["epoch", "Epoch", "epochs"]

# ====== 自動で拾いたいキー候補（run 側から読む） ======
SCAN_KEYS = [
    # geometry aug
    "degrees", "shear", "translate", "scale", "perspective",
    "fliplr", "flipud",
    # mosaic/mixup 他
    "mosaic", "close_mosaic", "mixup", "copy_paste",
    # color aug（電顕なら0推奨でも、変えているなら拾う）
    "hsv_h", "hsv_s", "hsv_v",
    # size/epochsなど
    "imgsz", "img_size", "epochs", "batch",
    # nms系（args.yaml側）
    "iou", "conf", "agnostic_nms", "augment",
    # 学習率系（変えてたら拾う）
    "lr0", "lrf", "cos_lr", "momentum", "weight_decay",
    # focal loss系
    "fl_gamma", "fl_alpha",
]

ALIASES = {
    "fl_alpha": ["fl_alpha", "focal_alpha", "alpha", "loss.alpha", "loss.focal.alpha", "focal.alpha"],
    "fl_gamma": ["fl_gamma", "focal_gamma", "gamma", "loss.gamma", "loss.focal.gamma", "focal.gamma"],
}

NAME_PATTERNS = [
    re.compile(r".*_(?P<model>yolo\d+[nslmx]-obb).*_i(?P<imgsz>\d+)", re.IGNORECASE),
    re.compile(r"fold(?P<fold>\d+)[_\-]+(?P<model>[^_]+)[_\-]+e(?P<epochs>\d+)[_\-]+i(?P<imgsz>\d+)", re.IGNORECASE),
    re.compile(r".*_fl[_-]?alpha(?P<fl_alpha>\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r".*_fl[_-]?gamma(?P<fl_gamma>\d+(?:\.\d+)?)", re.IGNORECASE),
]


def load_yaml_safe(p: Path):
    try:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}

def find_col(df: pd.DataFrame, candidates):
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
    recall_col = find_col(df, RECALL_CANDIDATES)
    precision_col = find_col(df, PRECISION_CANDIDATES)
    ep_col  = find_col(df, EPOCH_COLS)

    out = {"metric_col": m95_col, "metric50_col": m50_col, "epoch_col": ep_col}
    if m95_col is None and m50_col is None:
        return {**out,
                "best_mAP50-95": None, "best_epoch_mAP50-95": None,
                "best_mAP50": None, "best_epoch_mAP50": None,
                "best_recall": None, "best_epoch_recall": None,
                "best_precision": None, "best_epoch_precision": None}
    if m95_col is not None:
        idx = df[m95_col].astype(float).idxmax()
        out["best_mAP50-95"] = float(df.loc[idx, m95_col])
        out["best_epoch_mAP50-95"] = int(df.loc[idx, ep_col]) if ep_col else int(idx)
        # best epoch での recall と precision
        best_epoch = int(df.loc[idx, ep_col]) if ep_col else int(idx)
        if recall_col is not None:
            try:
                out["best_recall"] = float(df.loc[idx, recall_col])
                out["best_epoch_recall"] = best_epoch
            except (ValueError, KeyError):
                out["best_recall"] = None
                out["best_epoch_recall"] = None
        else:
            out["best_recall"] = None
            out["best_epoch_recall"] = None
        if precision_col is not None:
            try:
                out["best_precision"] = float(df.loc[idx, precision_col])
                out["best_epoch_precision"] = best_epoch
            except (ValueError, KeyError):
                out["best_precision"] = None
                out["best_epoch_precision"] = None
        else:
            out["best_precision"] = None
            out["best_epoch_precision"] = None
    else:
        out["best_mAP50-95"] = None
        out["best_epoch_mAP50-95"] = None
        out["best_recall"] = None
        out["best_epoch_recall"] = None
        out["best_precision"] = None
        out["best_epoch_precision"] = None
    if m50_col is not None:
        idx2 = df[m50_col].astype(float).idxmax()
        out["best_mAP50"] = float(df.loc[idx2, m50_col])
        out["best_epoch_mAP50"] = int(df.loc[idx2, ep_col]) if ep_col else int(idx2)
    else:
        out["best_mAP50"] = None
        out["best_epoch_mAP50"] = None
    return out

def get_last_epoch_metrics(df: pd.DataFrame) -> dict:
    """最後のepochの結果を取得"""
    ep_col = find_col(df, EPOCH_COLS)
    m95_col = find_col(df, MAP5095_CANDIDATES)
    m50_col = find_col(df, MAP50_CANDIDATES)
    recall_col = find_col(df, RECALL_CANDIDATES)
    precision_col = find_col(df, PRECISION_CANDIDATES)
    
    if df.empty:
        return {
            "last_epoch": None,
            "last_mAP50-95": None,
            "last_mAP50": None,
            "last_recall": None,
            "last_precision": None,
        }
    
    # 最後の行を取得
    last_idx = df.index[-1]
    last_epoch = int(df.loc[last_idx, ep_col]) if ep_col else int(last_idx) + 1
    
    out = {"last_epoch": last_epoch}
    
    if m95_col is not None:
        try:
            out["last_mAP50-95"] = float(df.loc[last_idx, m95_col])
        except (ValueError, KeyError):
            out["last_mAP50-95"] = None
    else:
        out["last_mAP50-95"] = None
    
    if m50_col is not None:
        try:
            out["last_mAP50"] = float(df.loc[last_idx, m50_col])
        except (ValueError, KeyError):
            out["last_mAP50"] = None
    else:
        out["last_mAP50"] = None
    
    if recall_col is not None:
        try:
            out["last_recall"] = float(df.loc[last_idx, recall_col])
        except (ValueError, KeyError):
            out["last_recall"] = None
    else:
        out["last_recall"] = None
    
    if precision_col is not None:
        try:
            out["last_precision"] = float(df.loc[last_idx, precision_col])
        except (ValueError, KeyError):
            out["last_precision"] = None
    else:
        out["last_precision"] = None
    
    return out

def parse_name_fallback(name: str) -> dict:
    info = {"model": None, "imgsz": None, "fl_alpha": None, "fl_gamma": None}
    for pat in NAME_PATTERNS:
        m = pat.match(name)
        if m:
            gd = m.groupdict()
            if gd.get("model"): info["model"] = gd["model"]
            if gd.get("imgsz"):
                try: info["imgsz"] = int(gd["imgsz"])
                except: pass
            if gd.get("fl_alpha"):
                try: info["fl_alpha"] = float(gd["fl_alpha"])
                except: pass
            if gd.get("fl_gamma"):
                try: info["fl_gamma"] = float(gd["fl_gamma"])
                except: pass
    return info


def coerce_num(x):
    """可能なら数値へ。小数は丸めて安定化（4桁）。"""
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return int(x) if float(x).is_integer() else round(float(x), 4)
        # 文字列
        sx = str(x).strip()
        if sx.lower() in {"true","false"}:
            return 1 if sx.lower()=="true" else 0
        if re.match(r"^-?\d+$", sx):
            return int(sx)
        if re.match(r"^-?\d+\.\d+$", sx):
            return round(float(sx), 4)
        return x
    except Exception:
        return x

def load_run_axes(run_dir: Path) -> dict:
    """args.yaml/hyp.yaml から可能な限り拾い、数値に整形（ネスト対応）。"""
    argsy = load_yaml_safe(run_dir / "args.yaml")
    hypy  = load_yaml_safe(run_dir / "hyp.yaml")
    # args を優先（CLI > hyp）したいなら右側に hypy をマージ
    merged = {}
    # まず単純マージ（トップレベル）
    if isinstance(hypy, dict):
        merged.update(hypy)
    if isinstance(argsy, dict):
        merged.update(argsy)

    out = {}

    # model / imgsz はトップ・ネスト両方から拾う
    for key, aliases in {
        "model": ["model"],
        "imgsz": ["imgsz", "img_size", "img-size", "img", "img_size_train"],
    }.items():
        _, v = _deep_find_any(merged, aliases)
        if v is not None:
            out[key] = v

    # スキャンキー（トップでもネストでも）
    for k in SCAN_KEYS:
        v = _deep_find_key(merged, k)
        if v is not None:
            out[k] = v

    # 別名対応（focal/loss 下にあるケース）
    for norm_key, alias_keys in ALIASES.items():
        if norm_key not in out:
            _, v = _deep_find_any(merged, alias_keys)
            if v is not None:
                out[norm_key] = v

    # 型整形
    for k, v in list(out.items()):
        out[k] = coerce_num(v)

    # フォールバック（run名から）
    fb = parse_name_fallback(run_dir.name)
    for k in ("model", "imgsz", "fl_alpha", "fl_gamma"):
        if (k not in out) or (out[k] in (None, "")):
            if fb.get(k) not in (None, ""):
                out[k] = fb[k]

    return out

def _deep_find_key(d, key):
    """辞書/リストを再帰的に探索して最初に見つかった key の値を返す。見つからなければ None。"""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            res = _deep_find_key(v, key)
            if res is not None:
                return res
    elif isinstance(d, list):
        for e in d:
            res = _deep_find_key(e, key)
            if res is not None:
                return res
    return None

def _deep_find_any(d, keys):
    """keys のいずれかが見つかったら (見つかったキー名, 値) を返す。無ければ (None, None)。"""
    for k in keys:
        v = _deep_find_key(d, k)
        if v is not None:
            return k, v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", "-p", required=True, help="Ultralyticsのプロジェクトディレクトリ（例: runs_obb）")
    ap.add_argument("--out_dir", "-o", default="reports", help="出力ディレクトリ")
    ap.add_argument("--train_cfg", "-c", required=True, help="学習に使った YAML（スイープ軸はこの中のリスト）")
    ap.add_argument("--name_prefix", "-n", default="", help="対象runの名前プレフィックス（空なら全件）")
    ap.add_argument("--discover_all", action="store_true",
                    help="train.yamlに無いキーでも、runを走査して値が2種類以上あれば自動軸として加える")
    args = ap.parse_args()

    project = Path(args.project)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ==== 1) 学習YAMLから “期待軸” を抽出 ====
    cfg = yaml.safe_load(Path(args.train_cfg).read_text(encoding="utf-8")) or {}
    train_hp = cfg.get("train", {}) or {}
    expected_values = {}
    base_axes = []

    for k, v in train_hp.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            base_axes.append(k)
            # 値を数値に正規化して保存
            vs = [coerce_num(e) for e in v]
            expected_values[k] = vs

    models = cfg.get("models", [])
    if models:
        base_axes = ["model"] + base_axes
        expected_values["model"] = list(models)

    # ==== 2) runs を走査し、各 run の軸候補＋metrics を収集 ====
    rows = []
    runs_axes_values = {}  # 軸 -> 異なる値セット（自動軸の抽出用）

    for run_dir in sorted(project.glob("*")):
        if not run_dir.is_dir():
            continue
        if args.name_prefix and not run_dir.name.startswith(args.name_prefix):
            continue
        if not (run_dir / "weights" / "best.pt").exists():
            continue

        csv_path = run_dir / "results.csv"
        if not csv_path.exists():
            continue
        try:
            rdf = pd.read_csv(csv_path)
        except Exception:
            continue
        if rdf is None or rdf.empty:
            continue

        axes_in_run = load_run_axes(run_dir)
        best = get_best_metrics(rdf)
        last = get_last_epoch_metrics(rdf)

        row = {"exp_name": run_dir.name, "exp_path": str(run_dir)}
        # まずは基軸（train.yaml 由来）
        for ax in base_axes:
            row[ax] = axes_in_run.get(ax, None)

        # SCAN_KEYS も含め、run から拾えた全キーを候補に
        for k, v in axes_in_run.items():
            row[k] = v
            if k not in runs_axes_values:
                runs_axes_values[k] = set()
            runs_axes_values[k].add(v)

        # metrics (best)
        row.update({
            "best_mAP50-95": best.get("best_mAP50-95"),
            "best_epoch_mAP50-95": best.get("best_epoch_mAP50-95"),
            "best_mAP50": best.get("best_mAP50"),
            "best_epoch_mAP50": best.get("best_epoch_mAP50"),
            "best_recall": best.get("best_recall"),
            "best_epoch_recall": best.get("best_epoch_recall"),
            "best_precision": best.get("best_precision"),
            "best_epoch_precision": best.get("best_epoch_precision"),
        })
        # metrics (last epoch)
        row.update({
            "last_epoch": last.get("last_epoch"),
            "last_mAP50-95": last.get("last_mAP50-95"),
            "last_mAP50": last.get("last_mAP50"),
            "last_recall": last.get("last_recall"),
            "last_precision": last.get("last_precision"),
        })
        rows.append(row)

    if not rows:
        print("[WARN] 対象runが見つかりませんでした。--name_prefix や --project を確認してください。")
        return

    df = pd.DataFrame(rows)

    # ==== 3) “自動軸” を決定（discover_all 有効時） ====
    sweep_axes = list(dict.fromkeys(base_axes))  # 重複排除を保ったまま
    if args.discover_all:
        for k, vals in runs_axes_values.items():
            if k in sweep_axes:
                continue
            # None/NaN を除いたユニーク数
            uniq = {x for x in vals if x is not None and (not (isinstance(x, float) and math.isnan(x)))}
            if len(uniq) >= 2:
                sweep_axes.append(k)

    # 列の型を安定化（数値化）
    for ax in sweep_axes:
        if ax in df.columns:
            try:
                df[ax] = df[ax].apply(coerce_num)
            except Exception:
                pass

    # 小数は丸め直し（グルーピング安定化）
    def _round_if_float_series(s: pd.Series):
        try:
            return s.apply(lambda x: round(x, 4) if isinstance(x, float) else x)
        except Exception:
            return s
    for ax in sweep_axes:
        if ax in df.columns:
            df[ax] = _round_if_float_series(df[ax])

    # 保存：全runの一覧
    per_run_csv = out_dir / "summary_per_run.csv"
    df.to_csv(per_run_csv, index=False)
    print(f"[OK] 保存: {per_run_csv}")

    # スコア列（mAP50-95優先、無ければmAP50）
    df["score_for_group"] = df["best_mAP50-95"].fillna(df["best_mAP50"])

    # ==== 4) 軸ごとの自動集計（単軸） ====
    for ax in sweep_axes:
        if ax not in df.columns:
            continue
        agg_dict = {
            "n": ("score_for_group", "count"),
            "mean": ("score_for_group", "mean"),
            "std": ("score_for_group", "std"),
            "mean_mAP50_95": ("best_mAP50-95", "mean"),
            "std_mAP50_95": ("best_mAP50-95", "std"),
            "mean_mAP50": ("best_mAP50", "mean"),
            "std_mAP50": ("best_mAP50", "std"),
        }
        # recall と precision が存在する場合のみ追加
        if "best_recall" in df.columns:
            agg_dict["mean_best_recall"] = ("best_recall", "mean")
            agg_dict["std_best_recall"] = ("best_recall", "std")
        if "last_recall" in df.columns:
            agg_dict["mean_last_recall"] = ("last_recall", "mean")
            agg_dict["std_last_recall"] = ("last_recall", "std")
        if "best_precision" in df.columns:
            agg_dict["mean_best_precision"] = ("best_precision", "mean")
            agg_dict["std_best_precision"] = ("best_precision", "std")
        if "last_precision" in df.columns:
            agg_dict["mean_last_precision"] = ("last_precision", "mean")
            agg_dict["std_last_precision"] = ("last_precision", "std")
        if "last_mAP50-95" in df.columns:
            agg_dict["mean_last_mAP50_95"] = ("last_mAP50-95", "mean")
            agg_dict["std_last_mAP50_95"] = ("last_mAP50-95", "std")
        if "last_mAP50" in df.columns:
            agg_dict["mean_last_mAP50"] = ("last_mAP50", "mean")
            agg_dict["std_last_mAP50"] = ("last_mAP50", "std")
        
        g = df.groupby(ax, dropna=False).agg(**agg_dict).reset_index().sort_values(ax)
        out = out_dir / f"summary_by_{ax}.csv"
        g.to_csv(out, index=False)
        print(f"[OK] 保存: {out}")

    # ==== 5) 全軸の組合せ集計（採用軸が2つ以上あるとき） ====
    if len(sweep_axes) >= 2 and all(ax in df.columns for ax in sweep_axes):
        agg_dict = {
            "n": ("score_for_group", "count"),
            "mean": ("score_for_group", "mean"),
            "std": ("score_for_group", "std"),
            "mean_mAP50_95": ("best_mAP50-95", "mean"),
            "std_mAP50_95": ("best_mAP50-95", "std"),
            "mean_mAP50": ("best_mAP50", "mean"),
            "std_mAP50": ("best_mAP50", "std"),
        }
        # recall と precision が存在する場合のみ追加
        if "best_recall" in df.columns:
            agg_dict["mean_best_recall"] = ("best_recall", "mean")
            agg_dict["std_best_recall"] = ("best_recall", "std")
        if "last_recall" in df.columns:
            agg_dict["mean_last_recall"] = ("last_recall", "mean")
            agg_dict["std_last_recall"] = ("last_recall", "std")
        if "best_precision" in df.columns:
            agg_dict["mean_best_precision"] = ("best_precision", "mean")
            agg_dict["std_best_precision"] = ("best_precision", "std")
        if "last_precision" in df.columns:
            agg_dict["mean_last_precision"] = ("last_precision", "mean")
            agg_dict["std_last_precision"] = ("last_precision", "std")
        if "last_mAP50-95" in df.columns:
            agg_dict["mean_last_mAP50_95"] = ("last_mAP50-95", "mean")
            agg_dict["std_last_mAP50_95"] = ("last_mAP50-95", "std")
        if "last_mAP50" in df.columns:
            agg_dict["mean_last_mAP50"] = ("last_mAP50", "mean")
            agg_dict["std_last_mAP50"] = ("last_mAP50", "std")
        
        g = df.groupby(sweep_axes, dropna=False).agg(**agg_dict).reset_index().sort_values(sweep_axes)
        out = out_dir / "summary_by_all_axes.csv"
        g.to_csv(out, index=False)
        print(f"[OK] 保存: {out}")

    # ==== 6) 期待グリッドに対するカバレッジ（train.yamlのリスト軸のみ） ====
    if expected_values:
        keys = list(expected_values.keys())
        grid = list(product(*[expected_values[k] for k in keys]))
        recs = []
        for vals in grid:
            mask = pd.Series([True] * len(df))
            for k, v in zip(keys, vals):
                if k in df.columns:
                    mask &= (df[k] == v)
            recs.append({**{k: v for k, v in zip(keys, vals)}, "found_runs": int(mask.sum())})
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

if __name__ == "__main__":
    main()

