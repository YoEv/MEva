#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from lib.repro.curve_channels import read_entropy_codebook_curves, read_loss_codebook_curves
from lib.repro.data_paths import get_exp11_split_paths, resolve_project_root
from lib.repro.metrics import safe_pearson_spearman

STATE_COLS = [
    "HH_RgtG",
    "HH_GgtR",
    "HL_RgtG",
    "LH_GgtR",
    "LL_RgtG",
    "LL_GgtR",
]


def _build_entropy_maps(manifest_paths: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    by_token: Dict[str, str] = {}
    by_audio: Dict[str, str] = {}
    for path in manifest_paths:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        if "entropy_curve_path" not in df.columns:
            continue
        for _, row in df.iterrows():
            ep = str(row["entropy_curve_path"])
            if not ep:
                continue
            if "token_loss_path" in df.columns:
                by_token[str(row["token_loss_path"])] = ep
            if "source_token_loss_path" in df.columns:
                by_token[str(row["source_token_loss_path"])] = ep
            if "audio_path" in df.columns:
                by_audio[str(row["audio_path"])] = ep
    return by_token, by_audio


def _state_name(r_bar: float, g_bar: float) -> str:
    if r_bar >= 0 and g_bar >= 0:
        return "HH_RgtG" if r_bar >= g_bar else "HH_GgtR"
    if r_bar >= 0 and g_bar < 0:
        return "HL_RgtG"
    if r_bar < 0 and g_bar >= 0:
        return "LH_GgtR"
    # both < 0
    return "LL_RgtG" if r_bar >= g_bar else "LL_GgtR"


def _build_clip_state_table(
    loss_curve_t: np.ndarray, entropy_curve_t: np.ndarray, token_window: int
) -> pd.DataFrame:
    t = min(int(loss_curve_t.shape[0]), int(entropy_curve_t.shape[0]))
    if t <= 0:
        return pd.DataFrame(columns=["w_start", "w_end", "r_bar", "g_bar", "d", "alignment", "state"])

    loss_t = loss_curve_t[:t].astype(np.float32, copy=False)
    ent_t = entropy_curve_t[:t].astype(np.float32, copy=False)
    r = loss_t - float(loss_t.mean())
    g = ent_t - float(ent_t.mean())

    rows = []
    for st in range(0, t, token_window):
        ed = min(st + token_window, t)
        if ed <= st:
            continue
        r_bar = float(np.mean(r[st:ed]))
        g_bar = float(np.mean(g[st:ed]))
        d = float(abs(r_bar - g_bar))
        rows.append(
            {
                "w_start": st,
                "w_end": ed,
                "r_bar": r_bar,
                "g_bar": g_bar,
                "d": d,
                "alignment": -d,
                "state": _state_name(r_bar, g_bar),
            }
        )
    return pd.DataFrame(rows)


def _segment_state_features(seg_row: pd.Series, clip_states: pd.DataFrame) -> Dict[str, float]:
    st = int(seg_row["start_token"])
    ed = int(seg_row["end_token"])
    overlap = clip_states[(clip_states["w_end"] > st) & (clip_states["w_start"] < ed)]
    if overlap.empty:
        out = {"r_bar_mean": float("nan"), "g_bar_mean": float("nan"), "alignment_mean": float("nan"), "d_mean": float("nan")}
        for s in STATE_COLS:
            out[f"state_frac_{s}"] = float("nan")
        out["dominant_state"] = ""
        return out

    out = {
        "r_bar_mean": float(overlap["r_bar"].mean()),
        "g_bar_mean": float(overlap["g_bar"].mean()),
        "alignment_mean": float(overlap["alignment"].mean()),
        "d_mean": float(overlap["d"].mean()),
    }
    n = float(len(overlap))
    vc = overlap["state"].value_counts()
    for s in STATE_COLS:
        out[f"state_frac_{s}"] = float(vc.get(s, 0.0) / n)
    out["dominant_state"] = str(vc.index[0]) if len(vc) > 0 else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="clean", choices=["clean"])
    ap.add_argument("--reports-dir", default="")
    ap.add_argument("--segment-window-seconds", type=int, default=5, choices=[1, 2, 5])
    ap.add_argument("--token-window", type=int, default=5)
    ap.add_argument("--experiments", action="append", default=[])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project_root = resolve_project_root(cfg)
    split_paths = get_exp11_split_paths(args.config, splits=args.splits)
    test_df = pd.read_csv(split_paths["test"])

    out_cfg = cfg.get("outputs", {})
    reports_dir = args.reports_dir or out_cfg.get("reports", "outputs/reports")
    reports_dir = reports_dir if os.path.isabs(reports_dir) else os.path.join(project_root, reports_dir)
    entropy_root = out_cfg.get("features_entropy", "outputs/features/entropy")
    entropy_root = entropy_root if os.path.isabs(entropy_root) else os.path.join(project_root, entropy_root)

    manifests = [
        os.path.join(entropy_root, "entropy_manifest_train.csv"),
        os.path.join(entropy_root, "entropy_manifest_val.csv"),
        os.path.join(entropy_root, "entropy_manifest_test.csv"),
    ]
    entropy_by_token, entropy_by_audio = _build_entropy_maps(manifests)

    # Build per-clip state tables once.
    clip_state_map: Dict[str, pd.DataFrame] = {}
    for _, row in test_df.iterrows():
        audio_path = str(row["audio_path"])
        token_loss_path = str(row["token_loss_path"])
        ep = entropy_by_token.get(token_loss_path) or entropy_by_audio.get(audio_path)
        if not ep or (not os.path.isfile(ep)):
            continue
        try:
            loss_ct = read_loss_codebook_curves(token_loss_path)  # [4, T]
            ent_ct = read_entropy_codebook_curves(ep)  # [4, T]
        except Exception:
            continue
        if loss_ct.shape[1] == 0 or ent_ct.shape[1] == 0:
            continue
        loss_t = loss_ct.mean(axis=0)  # avg over 4 codebooks
        ent_t = ent_ct.mean(axis=0)
        clip_state_map[audio_path] = _build_clip_state_table(loss_t, ent_t, token_window=args.token_window)

    seg_root = os.path.join(reports_dir, "segment_curves")
    experiments = args.experiments or [
        f"f01_loss_only_cnn_{args.splits}",
        f"f02_entropy_only_cnn_{args.splits}",
        f"f04_loss_entropy_cnn_{args.splits}",
    ]
    summary_rows = []

    for exp in experiments:
        seg_csv = os.path.join(seg_root, exp, f"segments_{args.segment_window_seconds}s.csv")
        if not os.path.isfile(seg_csv):
            summary_rows.append({"experiment": exp, "status": "missing_segments_csv"})
            continue
        sdf = pd.read_csv(seg_csv)
        if sdf.empty:
            summary_rows.append({"experiment": exp, "status": "empty_segments_csv"})
            continue

        feats = []
        for _, srow in sdf.iterrows():
            audio_path = str(srow["audio_path"])
            clip_states = clip_state_map.get(audio_path)
            if clip_states is None or clip_states.empty:
                feats.append(
                    {"alignment_mean": np.nan, "d_mean": np.nan, **{f"state_frac_{s}": np.nan for s in STATE_COLS}}
                )
                continue
            feats.append(_segment_state_features(srow, clip_states))
        fdf = pd.DataFrame(feats)
        out = pd.concat([sdf.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)

        enriched_dir = os.path.join(seg_root, exp)
        os.makedirs(enriched_dir, exist_ok=True)
        enriched_csv = os.path.join(enriched_dir, f"segments_{args.segment_window_seconds}s_with_curve_states.csv")
        out.to_csv(enriched_csv, index=False)

        y_pred = out["pred"].to_numpy(dtype=float)
        pr_r, sp_r = safe_pearson_spearman(y_pred, out["r_bar_mean"].to_numpy(dtype=float))
        pr_g, sp_g = safe_pearson_spearman(y_pred, out["g_bar_mean"].to_numpy(dtype=float))
        pr_d, sp_d = safe_pearson_spearman(y_pred, out["d_mean"].to_numpy(dtype=float))
        pr_align, sp_align = safe_pearson_spearman(y_pred, out["alignment_mean"].to_numpy(dtype=float))
        state_means = out.groupby("dominant_state")["pred"].agg(["count", "mean", "std"]).reset_index()
        means = state_means["mean"].to_numpy(dtype=float) if len(state_means) > 0 else np.array([], dtype=float)
        global_var = float(np.var(y_pred)) if len(y_pred) > 0 else float("nan")
        between_var = float(np.var(means)) if len(means) > 0 else float("nan")
        row = {
            "experiment": exp,
            "status": "ok",
            "segment_window_seconds": args.segment_window_seconds,
            "token_window": args.token_window,
            "n_segments": int(len(out)),
            "pearson_pred_vs_r_bar_mean": pr_r,
            "spearman_pred_vs_r_bar_mean": sp_r,
            "pearson_pred_vs_g_bar_mean": pr_g,
            "spearman_pred_vs_g_bar_mean": sp_g,
            "pearson_pred_vs_d_mean": pr_d,
            "spearman_pred_vs_d_mean": sp_d,
            "pearson_pred_vs_alignment": pr_align,
            "spearman_pred_vs_alignment": sp_align,
            "pred_global_var": global_var,
            "pred_between_state_mean_var": between_var,
            "enriched_segments_csv": enriched_csv,
        }
        for s in STATE_COLS:
            pr_s, sp_s = safe_pearson_spearman(y_pred, out[f"state_frac_{s}"].to_numpy(dtype=float))
            row[f"pearson_pred_vs_{s}"] = pr_s
            row[f"spearman_pred_vs_{s}"] = sp_s
        state_means_csv = os.path.join(enriched_dir, f"state_dominant_pred_stats_{args.segment_window_seconds}s.csv")
        state_means.to_csv(state_means_csv, index=False)
        row["state_dominant_stats_csv"] = state_means_csv
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    out_csv = os.path.join(seg_root, f"curve_state_alignment_summary_{args.splits}_{args.segment_window_seconds}s.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
