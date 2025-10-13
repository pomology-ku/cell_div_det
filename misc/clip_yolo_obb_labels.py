#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-OBB poly ラベル (.txt) を [0,1] に収める（長辺方向に沿った並行移動）

アルゴリズム（OOB 行のみ適用）:
  1) OOB 頂点を特定
  2) その頂点に接する 2 辺のうち長い方を「長辺」とする
  3) 長辺の向きベクトル d に沿って、その OOB 頂点 p を [0,1]^2 の境界まで最小移動でスライド
  4) p と「短辺」でつながる隣頂点 q に同じベクトル v = t*d を適用（長辺の平行性を保持）
  5) これで全頂点が内側に入るまで繰り返し（最大4回）

負が無い & 1超えが無い行は、そのまま（変更なし）。
"""

import argparse
from pathlib import Path
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(description="Clip YOLO-OBB (poly) label coordinates to [0,1] along long-edge direction.")
    ap.add_argument("-l", "--labels", required=True, type=Path, help="labels ディレクトリ（再帰的探索）")
    ap.add_argument("--area-min", type=float, default=1e-6, help="この面積未満のpolyは除外")
    ap.add_argument("--backup", action="store_true", help=".bak バックアップを残す")
    return ap.parse_args()

def poly_area(P):
    return 0.5 * abs(np.dot(P[:,0], np.roll(P[:,1], -1)) - np.dot(P[:,1], np.roll(P[:,0], -1)))

def ensure_ccw(P):
    cross = np.cross(P[1]-P[0], P[2]-P[1])
    return P if cross >= 0 else P[::-1]

def edge_lengths(P):
    e0 = np.linalg.norm(P[1]-P[0])
    e1 = np.linalg.norm(P[2]-P[1])
    e2 = np.linalg.norm(P[3]-P[2])
    e3 = np.linalg.norm(P[0]-P[3])
    return np.array([e0, e1, e2, e3], dtype=float)

def is_oob_point(p):
    return (p[0] < 0.0) or (p[0] > 1.0) or (p[1] < 0.0) or (p[1] > 1.0)

def feasible_t_interval_for_point(q, d):
    """q + t d が [0,1]^2 に入る t の区間を返す。存在しなければ None。"""
    t_low, t_high = -np.inf, np.inf
    # x
    if abs(d[0]) < 1e-12:
        if q[0] < 0.0 or q[0] > 1.0:
            return None
    else:
        tx0 = (0.0 - q[0]) / d[0]
        tx1 = (1.0 - q[0]) / d[0]
        lo, hi = (tx0, tx1) if tx0 <= tx1 else (tx1, tx0)
        t_low, t_high = max(t_low, lo), min(t_high, hi)
    # y
    if abs(d[1]) < 1e-12:
        if q[1] < 0.0 or q[1] > 1.0:
            return None
    else:
        ty0 = (0.0 - q[1]) / d[1]
        ty1 = (1.0 - q[1]) / d[1]
        lo, hi = (ty0, ty1) if ty0 <= ty1 else (ty1, ty0)
        t_low, t_high = max(t_low, lo), min(t_high, hi)
    if t_low > t_high:
        return None
    return (t_low, t_high)

def candidate_t_to_touch_boundary(p, d):
    """p + t d が境界 x=0,1,y=0,1 のいずれかに乗る t 候補（実数）を返す。"""
    cands = []
    if abs(d[0]) > 1e-12:
        cands += [(0.0 - p[0]) / d[0], (1.0 - p[0]) / d[0]]
    if abs(d[1]) > 1e-12:
        cands += [(0.0 - p[1]) / d[1], (1.0 - p[1]) / d[1]]
    # 重複/NaN除去
    cands = [t for t in cands if np.isfinite(t)]
    # 原点(=移動なし)は意味がないので残してもよいが後でフィルタ
    return cands

def move_pair_along_long_edge(P, i, area_min=1e-6):
    """
    頂点 i が OOB。i に接する2辺のうち長い方の方向 d に沿って i をスライド。
    「短辺」でつながる隣頂点 j にも同じベクトルを適用。
    成功すれば (True, 新P)、不可能なら (False, 旧P)。
    """
    n = 4
    # 辺長
    L = edge_lengths(P)
    # i に接する辺は (i-1)->i と i->(i+1)
    e_prev_len = L[(i-1) % n]
    e_next_len = L[i % n]
    # 長辺の向き d と、短辺でつながる相方 j を決定
    if e_prev_len >= e_next_len:
        # 長辺は (i-1)->i、短辺は i->(i+1) → 相方は j = i+1
        d = P[i] - P[(i-1) % n]
        j = (i+1) % n
    else:
        # 長辺は i->(i+1)、短辺は (i-1)->i → 相方は j = i-1
        d = P[(i+1) % n] - P[i]
        j = (i-1) % n

    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return False, P  # 退避：ゼロ長は無理
    d = d / norm

    p = P[i].copy()
    q = P[j].copy()

    # 両点が [0,1]^2 内に入る t の共通可行区間
    I_p = feasible_t_interval_for_point(p, d)
    I_q = feasible_t_interval_for_point(q, d)
    if I_p is None or I_q is None:
        return False, P
    t_low = max(I_p[0], I_q[0])
    t_high = min(I_p[1], I_q[1])
    if t_low > t_high:
        return False, P

    # p が境界に「触れる」t候補から、可行区間に入る最小 |t| を選ぶ
    cand = []
    for t in candidate_t_to_touch_boundary(p, d):
        if t_low - 1e-12 <= t <= t_high + 1e-12:
            # 過剰候補除去：移動後に確かに内側（境界上含む）か簡易チェック
            pp = p + t * d
            if -1e-9 <= pp[0] <= 1+1e-9 and -1e-9 <= pp[1] <= 1+1e-9:
                cand.append(t)
    if not cand:
        # 境界に触れる候補が無い場合、最近傍の端でクリップ（最小ノルム）
        # 0に最も近い t を可行区間端から選ぶ
        t = t_low if abs(t_low) < abs(t_high) else t_high
    else:
        # 最小 |t|
        t = min(cand, key=lambda x: abs(x))

    v = t * d
    P2 = P.copy()
    P2[i] = P2[i] + v
    P2[j] = P2[j] + v

    # 数値誤差の微小はゼロへ寄せる
    P2 = np.where(P2 < 0, 0.0, P2)
    P2 = np.where(P2 > 1, 1.0, P2)

    if poly_area(P2) < area_min:
        return False, P
    return True, P2

def sanitize_poly_line(parts, area_min=1e-6):
    """cls x1 y1 ... y4 [conf] を修正して返す。除外対象は None。"""
    cls = int(float(parts[0]))
    coords = list(map(float, parts[1:9]))
    conf = parts[9] if len(parts) >= 10 else None

    P = np.array(coords, dtype=float).reshape(4,2)
    P = ensure_ccw(P)

    # OOB がなければ何もしない（そのまま）
    if not any(is_oob_point(P[k]) for k in range(4)):
        if poly_area(P) < area_min:
            return None
        flat = [str(cls)] + [f"{v:.6f}" for v in P.reshape(-1).tolist()]
        if conf is not None:
            flat.append(conf)
        return " ".join(flat)

    # OOB がある場合：アルゴを最大 4 回まで適用（各回 1 頂点ずつ直す）
    for _ in range(4):
        # まだ OOB の頂点を探す
        oob_idx = next((k for k in range(4) if is_oob_point(P[k])), None)
        if oob_idx is None:
            break
        ok, P_new = move_pair_along_long_edge(P, oob_idx, area_min=area_min)
        if not ok:
            print(f"[WARN] cannot fix OOB poly, dropping: {' '.join(parts)}")
            # # どうしても直らない場合は、最後に安全クリップ（最小変更のため保険）
            # P[:,0] = np.clip(P[:,0], 0.0, 1.0)
            # P[:,1] = np.clip(P[:,1], 0.0, 1.0)
            # if poly_area(P) < area_min:
            #     return None
            # P = ensure_ccw(P)
            # flat = [str(cls)] + [f"{v:.6f}" for v in P.reshape(-1).tolist()]
            # if conf is not None:
            #     flat.append(conf)
            # return " ".join(flat)
        P = P_new

    # 最終チェック
    if poly_area(P) < area_min:
        return None
    P = ensure_ccw(P)
    flat = [str(cls)] + [f"{v:.6f}" for v in P.reshape(-1).tolist()]
    if conf is not None:
        flat.append(conf)
    return " ".join(flat)

def clip_dir(labels_dir: Path, area_min: float=1e-6, backup: bool=False):
    n_total = n_fixed = n_dropped = 0
    modified_files = []
    for txt in labels_dir.rglob("*.txt"):
        lines = txt.read_text(encoding="utf-8").splitlines()
        out_lines, changed = [], False
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 9:  # 他形式は無視
                out_lines.append(line)
                continue
            n_total += 1
            before = parts[1:9]
            fixed = sanitize_poly_line(parts, area_min)
            if fixed is None:
                n_dropped += 1
                changed = True
                continue
            after = fixed.split()[1:9]
            if any(abs(float(a)-float(b)) > 1e-9 for a,b in zip(before, after)):
                changed = True
                n_fixed += 1
            out_lines.append(fixed)
        if changed:
            if backup:
                txt.rename(txt.with_suffix(".txt.bak"))
            txt.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            modified_files.append(str(txt))
    print(f"[DONE] total={n_total}, fixed={n_fixed}, dropped={n_dropped}")
    if modified_files:
        print("\n[Modified files]")
        for f in modified_files:
            print(" -", f)
    else:
        print("\n[Modified files] None")

def main():
    args = parse_args()
    if not args.labels.exists():
        raise SystemExit(f"[ERROR] not found: {args.labels}")
    clip_dir(args.labels, args.area_min, args.backup)

if __name__ == "__main__":
    main()
