#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run coco_recall_union.py across multiple folds and summarize recall results.

Usage example:
    python fold_recall_union.py \
        --exp_a kaki/reports/exp109 \
        --exp_b kaki/reports/exp110 \
        --n_folds 3 \
        --recall_iou 0 --iou_mode aabb --conf_thresh 0.1 --check_gt_diff

File naming convention (per fold):
    GT:     {exp_a}/fold{i}/metrics/tta_gt_{file_suffix}.coco.json
    pred_a: {exp_a}/fold{i}/metrics/tta_pred_{file_suffix}.coco.json
    pred_b: {exp_b}/fold{i}/metrics/tta_pred_{file_suffix}.coco.json
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_recall_summary(text: str) -> dict:
    """Extract recall_a, recall_b, recall_union from coco_recall_union.py stdout."""
    result = {}
    patterns = {
        "recall_a":     r"recall_a:\s+([\d.]+)\s+\(matched (\d+)/(\d+), preds=(\d+)\)",
        "recall_b":     r"recall_b:\s+([\d.]+)\s+\(matched (\d+)/(\d+), preds=(\d+)\)",
        "recall_union": r"recall_union:\s+([\d.]+)\s+\(matched (\d+)/(\d+), preds=(\d+)\)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[key] = float(m.group(1))
            result[f"{key}_matched"] = int(m.group(2))
            result[f"{key}_total"]   = int(m.group(3))
            result[f"{key}_preds"]   = int(m.group(4))
    return result


def run_fold(script: Path, gt_json: Path, pred_a_json: Path, pred_b_json: Path,
             recall_iou: float, iou_mode: str, conf_thresh: float,
             check_gt_diff: bool, print_per_class: bool) -> str:
    cmd = [
        sys.executable, str(script),
        "--gt",        str(gt_json),
        "--pred_a",    str(pred_a_json),
        "--pred_b",    str(pred_b_json),
        "--recall_iou",  str(recall_iou),
        "--iou_mode",    iou_mode,
        "--conf_thresh", str(conf_thresh),
    ]
    if check_gt_diff:
        cmd.append("--check_gt_diff")
    if print_per_class:
        cmd.append("--print_per_class")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.stdout


def print_summary_table(fold_results: list):
    col_w = 14  # column width for recall values

    header = f"{'fold':<8} {'recall_a':>{col_w}} {'recall_b':>{col_w}} {'recall_union':>{col_w}}"
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("RECALL SUMMARY TABLE")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in fold_results:
        ra  = r.get("recall_a",     float("nan"))
        rb  = r.get("recall_b",     float("nan"))
        ru  = r.get("recall_union", float("nan"))
        mta = f"({r.get('recall_a_matched','?')}/{r.get('recall_a_total','?')})"
        mtb = f"({r.get('recall_b_matched','?')}/{r.get('recall_b_total','?')})"
        mtu = f"({r.get('recall_union_matched','?')}/{r.get('recall_union_total','?')})"
        print(f"{r['fold']:<8} {ra:>7.4f} {mta:<6} {rb:>7.4f} {mtb:<6} {ru:>7.4f} {mtu}")

    print(sep)

    keys = ["recall_a", "recall_b", "recall_union"]
    arrays = {k: np.array([r.get(k, float("nan")) for r in fold_results]) for k in keys}

    def stat_line(label, fn):
        vals = [fn(arrays[k]) for k in keys]
        print(f"{label:<8} {vals[0]:>{col_w}.4f} {vals[1]:>{col_w}.4f} {vals[2]:>{col_w}.4f}")

    stat_line("mean", np.nanmean)
    stat_line("std",  np.nanstd)
    stat_line("best", np.nanmax)

    # Middle fold: recall_union closest to mean (representative fold for inference)
    union_arr  = arrays["recall_union"]
    union_mean = float(np.nanmean(union_arr))
    dists      = np.abs(union_arr - union_mean)
    best_idx   = int(np.nanargmax(union_arr))
    mid_idx    = int(np.nanargmin(dists))

    print(sep)
    print(f"\nBest fold  (recall_union = {union_arr[best_idx]:.4f}): {fold_results[best_idx]['fold']}")
    print(f"Middle fold (recall_union ≈ mean {union_mean:.4f}): {fold_results[mid_idx]['fold']}  "
          f"({union_arr[mid_idx]:.4f})")


def main():
    ap = argparse.ArgumentParser(
        description="Run coco_recall_union.py per fold and summarize recall into a table."
    )
    ap.add_argument("-a", "--exp_a",        required=True,
                    help="Path prefix for experiment A, e.g. kaki/reports/exp109")
    ap.add_argument("-b", "--exp_b",        required=True,
                    help="Path prefix for experiment B, e.g. kaki/reports/exp110")
    ap.add_argument("-n", "--n_folds",      type=int, default=5,
                    help="Number of folds (default: 5)")
    ap.add_argument("-s", "--file_suffix",  default="conf0p1",
                    help="Suffix used in JSON filenames: tta_gt_{suffix}.coco.json (default: conf0p1)")
    # pass-through args for coco_recall_union.py
    ap.add_argument("-r", "--recall_iou",   type=float, default=0.0)
    ap.add_argument("-m", "--iou_mode",     choices=["obb", "aabb"], default="aabb")
    ap.add_argument("-c", "--conf_thresh",  type=float, default=0.1)
    ap.add_argument("-d", "--check_gt_diff",   action="store_true")
    ap.add_argument("-p", "--print_per_class", action="store_true")
    args = ap.parse_args()

    script  = Path(__file__).parent / "coco_recall_union.py"
    exp_a   = Path(args.exp_a)
    exp_b   = Path(args.exp_b)
    suffix  = args.file_suffix

    fold_results = []

    for i in range(args.n_folds):
        fold_name   = f"fold{i}"
        gt_json     = exp_a / fold_name / "metrics" / f"tta_gt_{suffix}.coco.json"
        pred_a_json = exp_a / fold_name / "metrics" / f"tta_pred_{suffix}.coco.json"
        pred_b_json = exp_b / fold_name / "metrics" / f"tta_pred_{suffix}.coco.json"

        print(f"\n{'='*10} {fold_name} {'='*10}")
        stdout = run_fold(
            script, gt_json, pred_a_json, pred_b_json,
            recall_iou=args.recall_iou,
            iou_mode=args.iou_mode,
            conf_thresh=args.conf_thresh,
            check_gt_diff=args.check_gt_diff,
            print_per_class=args.print_per_class,
        )
        print(stdout, end="")

        recalls = parse_recall_summary(stdout)
        if not recalls:
            print(f"[WARN] Could not parse recall values for {fold_name}; check output above.",
                  file=sys.stderr)
        recalls["fold"] = fold_name
        fold_results.append(recalls)

    print_summary_table(fold_results)


if __name__ == "__main__":
    main()
