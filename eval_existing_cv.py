#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate one existing YOLO OBB model on each fold without training.

The output directory layout is intentionally compatible with fold_results.py:
each fold gets a run directory containing args.yaml and a single-row results.csv.
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def load_yaml_safe(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def scalar_or_default(d: dict, key: str, default: Any) -> Any:
    v = d.get(key, default)
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    return v


def collect_metrics_from_api(val_ret) -> dict:
    out = {}
    try:
        m = getattr(val_ret, "metrics", None) or val_ret
        target = None
        for key in ("obb", "box", "segment"):
            if hasattr(m, key) and getattr(m, key) is not None:
                target = getattr(m, key)
                break
        if target is not None:
            if hasattr(target, "map"):
                out["map"] = float(target.map)
            if hasattr(target, "map50"):
                out["map50"] = float(target.map50)
            if hasattr(target, "mp"):
                out["precision"] = float(target.mp)
            if hasattr(target, "mr"):
                out["recall"] = float(target.mr)
    except Exception:
        pass
    return out


def collect_metrics_from_csv(run_dir: Path) -> dict:
    csv_paths = [run_dir / "results.csv"] + list(run_dir.glob("**/results.csv"))
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            row = rows[-1]
        except Exception:
            continue

        out = {}
        key_map = {
            "map": [
                "metrics/mAP50-95(OBB)",
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
                "mAP50-95",
                "map",
            ],
            "map50": [
                "metrics/mAP50(OBB)",
                "metrics/mAP50(B)",
                "metrics/mAP50",
                "mAP50",
                "map50",
            ],
            "precision": [
                "metrics/precision(OBB)",
                "metrics/precision(B)",
                "metrics/precision",
                "precision",
                "mp",
            ],
            "recall": [
                "metrics/recall(OBB)",
                "metrics/recall(B)",
                "metrics/recall",
                "recall",
                "mr",
            ],
        }
        for out_key, candidates in key_map.items():
            for c in candidates:
                if c in row and row[c] not in (None, "", "nan"):
                    try:
                        out[out_key] = float(row[c])
                        break
                    except Exception:
                        continue
        if out:
            return out
    return {}


def write_results_csv(run_dir: Path, metrics: dict) -> None:
    row = {"epoch": 0}
    metric_columns = [
        ("metrics/precision(OBB)", "precision"),
        ("metrics/recall(OBB)", "recall"),
        ("metrics/mAP50(OBB)", "map50"),
        ("metrics/mAP50-95(OBB)", "map"),
    ]
    for col, key in metric_columns:
        if metrics.get(key) is not None:
            row[col] = metrics[key]
    with open(run_dir / "results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_args_yaml(run_dir: Path, payload: dict) -> None:
    with open(run_dir / "args.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate one existing model on foldK/data.yaml without training."
    )
    ap.add_argument("--root", required=True, help="Root containing fold0, fold1, ...")
    ap.add_argument("--weights", "-w", required=True, help="Existing .pt model to evaluate")
    ap.add_argument("--cfg", required=True, help="Training cfg used to obtain train.imgsz and val options")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--start_fold", type=int, default=0)
    ap.add_argument("--end_fold", type=int, default=None)
    ap.add_argument("--project", default=None, help="Output project dir; default uses cfg project or runs_obb")
    ap.add_argument("--name_prefix", "-n", default="eval_existing", help="Prefix for run names")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default=None, help="Override device from cfg")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    weights = Path(args.weights).resolve()
    cfg_path = Path(args.cfg).resolve()
    assert weights.exists(), f"weights not found: {weights}"
    assert cfg_path.exists(), f"cfg not found: {cfg_path}"

    cfg = load_yaml_safe(cfg_path)
    train_hp = cfg.get("train", {}) or {}
    val_hp = dict(cfg.get("val", {}) or {})
    project = Path(args.project or cfg.get("project", "runs_obb"))
    device = args.device if args.device is not None else cfg.get("device", None)
    imgsz = scalar_or_default(train_hp, "imgsz", 1024)
    end_fold = args.end_fold if args.end_fold is not None else args.folds - 1

    # The CLI split is the source of truth for this evaluation run.
    for k_rm in (
        "data",
        "imgsz",
        "project",
        "name",
        "device",
        "split",
        "model",
        "weights",
        "exist_ok",
    ):
        val_hp.pop(k_rm, None)

    for fold in range(args.start_fold, end_fold + 1):
        fold_dir = root / f"fold{fold}"
        data_yaml = fold_dir / "data.yaml"
        assert data_yaml.exists(), f"missing: {data_yaml}"

        run_name = f"{args.name_prefix}_fold{fold}_{weights.stem}_i{imgsz}_{args.split}"
        run_dir = project / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[PY] yolo val (fold={fold}, split={args.split}, weights={weights.name}, imgsz={imgsz})")
        model = YOLO(str(weights))
        val_kwargs = dict(
            data=str(data_yaml),
            split=args.split,
            imgsz=imgsz,
            project=str(project),
            name=run_name,
            exist_ok=True,
            **val_hp,
        )
        if device is not None:
            val_kwargs["device"] = device

        val_ret = model.val(**val_kwargs)
        metrics = collect_metrics_from_api(val_ret)
        if not metrics:
            metrics = collect_metrics_from_csv(run_dir)
        if not metrics:
            print(f"[WARN] metrics could not be parsed for fold{fold}; writing empty metric cells")

        write_results_csv(run_dir, metrics)
        write_args_yaml(
            run_dir,
            {
                "model": weights.stem,
                "weights": str(weights),
                "imgsz": imgsz,
                "split": args.split,
                "eval_only": True,
                "fold": fold,
                "data": str(data_yaml),
                "project": str(project),
                "name": run_name,
                "device": device,
            },
        )
        print(f"[OK] saved eval run: {run_dir}")


if __name__ == "__main__":
    main()
