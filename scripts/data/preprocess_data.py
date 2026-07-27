#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "project_root" not in cfg:
        env_root = os.environ.get("PROJECT_ROOT", "")
        if not env_root:
            env_root = os.path.abspath(os.path.join(os.path.dirname(path), "../.."))
        cfg["project_root"] = env_root
    return cfg


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _copy_csv(src: str, dst: str) -> None:
    if not os.path.isfile(src):
        raise FileNotFoundError(f"missing source csv: {src}")
    _ensure_parent(dst)
    pd.read_csv(src).to_csv(dst, index=False)


def _musiceval_copy_splits(cfg: dict, source_root: str) -> None:
    root = cfg["project_root"]
    target = cfg["data"]["splits"]
    mapping = {"clean": {"train": "train.csv", "val": "val.csv", "test": "test.csv"}}
    for mode, names in mapping.items():
        for split, fname in names.items():
            src = os.path.join(source_root, fname)
            dst_rel = target[mode][split]
            dst = dst_rel if os.path.isabs(dst_rel) else os.path.join(root, dst_rel)
            if os.path.isfile(dst):
                print(f"already exists, skipping: {dst}")
                continue
            _copy_csv(src, dst)
            print(f"copied {src} -> {dst}")


@dataclass
class PairObs:
    a: str
    b: str
    y: float  # +1 win a, 0 tie, -1 win b


def _parse_outcome(v: str) -> float:
    s = str(v).strip().lower()
    if s in {"a", "win_a", "a_win", "left", "1", "a_wins"}:
        return 1.0
    if s in {"b", "win_b", "b_win", "right", "-1", "b_wins"}:
        return -1.0
    return 0.0


def _load_pairs(pairwise_csv: str, col_a: str, col_b: str, col_outcome: str) -> List[PairObs]:
    df = pd.read_csv(pairwise_csv)
    out = []
    for _, row in df.iterrows():
        out.append(PairObs(str(row[col_a]), str(row[col_b]), _parse_outcome(row[col_outcome])))
    return out


def _fit_bt_scores(pairs: List[PairObs], iters: int = 300, lr: float = 0.05, l2: float = 1e-4) -> Dict[str, float]:
    items = sorted({p.a for p in pairs} | {p.b for p in pairs})
    idx = {k: i for i, k in enumerate(items)}
    s = np.zeros(len(items), dtype=np.float64)
    # Tie is encoded as two half games.
    for _ in range(iters):
        grad = np.zeros_like(s)
        for p in pairs:
            ia, ib = idx[p.a], idx[p.b]
            pa = 1.0 / (1.0 + np.exp(-(s[ia] - s[ib])))
            if p.y > 0:
                grad[ia] += 1.0 - pa
                grad[ib] -= 1.0 - pa
            elif p.y < 0:
                grad[ia] -= pa
                grad[ib] += pa
            else:
                # tie contributes half-win each
                grad[ia] += 0.5 - pa
                grad[ib] -= 0.5 - (1.0 - pa)
        grad -= l2 * s
        s += lr * grad
        s -= s.mean()
    return {k: float(s[idx[k]]) for k in items}


def _fit_mse_scores(pairs: List[PairObs], iters: int = 400, lr: float = 0.05, l2: float = 1e-4) -> Dict[str, float]:
    items = sorted({p.a for p in pairs} | {p.b for p in pairs})
    idx = {k: i for i, k in enumerate(items)}
    s = np.zeros(len(items), dtype=np.float64)
    for _ in range(iters):
        grad = np.zeros_like(s)
        for p in pairs:
            ia, ib = idx[p.a], idx[p.b]
            err = (s[ia] - s[ib]) - p.y
            grad[ia] += err
            grad[ib] -= err
        grad = grad / max(1, len(pairs))
        grad += l2 * s
        s -= lr * grad
        s -= s.mean()
    return {k: float(s[idx[k]]) for k in items}


def _map_to_1_5(scores: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(scores.values()), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo + 1e-12:
        return {k: 3.0 for k in scores}
    out = {}
    for k, v in scores.items():
        out[k] = 1.0 + 4.0 * (float(v) - lo) / (hi - lo)
    return out


def _pairwise_scale_and_split(
    cfg: dict,
    pairwise_csv: str,
    out_manifest: str,
    out_split_dir: str,
    col_a: str,
    col_b: str,
    col_outcome: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    pairs = _load_pairs(pairwise_csv, col_a=col_a, col_b=col_b, col_outcome=col_outcome)
    bt = _map_to_1_5(_fit_bt_scores(pairs))
    mse = _map_to_1_5(_fit_mse_scores(pairs))

    ids = sorted(bt.keys())
    df = pd.DataFrame(
        {
            "item_id": ids,
            "score_bt_1to5": [bt[i] for i in ids],
            "score_mse_1to5": [mse[i] for i in ids],
        }
    )
    _ensure_parent(out_manifest)
    df.to_csv(out_manifest, index=False)
    print(f"wrote scaled manifest: {out_manifest} ({len(df)} rows)")

    # Split by item_id (simple global random split).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)
    i_train = perm[:n_train]
    i_val = perm[n_train : n_train + n_val]
    i_test = perm[n_train + n_val :]

    os.makedirs(out_split_dir, exist_ok=True)
    df.iloc[i_train].to_csv(os.path.join(out_split_dir, "train.csv"), index=False)
    df.iloc[i_val].to_csv(os.path.join(out_split_dir, "val.csv"), index=False)
    df.iloc[i_test].to_csv(os.path.join(out_split_dir, "test.csv"), index=False)
    print(f"wrote splits under: {out_split_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase7 data preprocessing entrypoint.")
    ap.add_argument("--config", default="config/paths.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("musiceval-copy-splits", help="Copy existing MusicEval split CSVs into data/splits.")
    p1.add_argument("--source-root", default="", help="Directory containing train/val/test CSVs.")

    p2 = sub.add_parser("pairwise-scale-and-split", help="For full datasets: map pairwise labels to 1-5 and split.")
    p2.add_argument("--pairwise-csv", required=True)
    p2.add_argument("--out-manifest", required=True)
    p2.add_argument("--out-split-dir", required=True)
    p2.add_argument("--col-a", default="item_a")
    p2.add_argument("--col-b", default="item_b")
    p2.add_argument("--col-outcome", default="outcome")
    p2.add_argument("--train-ratio", type=float, default=0.8)
    p2.add_argument("--val-ratio", type=float, default=0.1)
    p2.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    root = cfg["project_root"]

    if args.cmd == "musiceval-copy-splits":
        default_src = cfg.get("data", {}).get("preprocess", {}).get(
            "musiceval_source_split_root",
            "experiments/phase7/loss_eval_experiments/exp11_large_scale_replication/1_data_preparation",
        )
        source_root = args.source_root or default_src
        src = source_root if os.path.isabs(source_root) else os.path.join(root, source_root)
        _musiceval_copy_splits(cfg, src)
        return

    if args.cmd == "pairwise-scale-and-split":
        pairwise_csv = args.pairwise_csv if os.path.isabs(args.pairwise_csv) else os.path.join(root, args.pairwise_csv)
        out_manifest = args.out_manifest if os.path.isabs(args.out_manifest) else os.path.join(root, args.out_manifest)
        out_split_dir = args.out_split_dir if os.path.isabs(args.out_split_dir) else os.path.join(root, args.out_split_dir)
        _pairwise_scale_and_split(
            cfg=cfg,
            pairwise_csv=pairwise_csv,
            out_manifest=out_manifest,
            out_split_dir=out_split_dir,
            col_a=args.col_a,
            col_b=args.col_b,
            col_outcome=args.col_outcome,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        return


if __name__ == "__main__":
    main()
