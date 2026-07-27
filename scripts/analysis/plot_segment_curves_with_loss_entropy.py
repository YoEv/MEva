#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from lib.repro.curve_channels import read_entropy_codebook_curves, read_loss_codebook_curves
from lib.repro.data_paths import get_exp11_split_paths, resolve_project_root

matplotlib.use("Agg")


def _entropy_maps(manifest_paths: List[str]) -> Dict[str, str]:
    by_audio: Dict[str, str] = {}
    by_token: Dict[str, str] = {}
    for p in manifest_paths:
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        if "entropy_curve_path" not in df.columns:
            continue
        for _, r in df.iterrows():
            ep = str(r["entropy_curve_path"])
            if not ep:
                continue
            if "audio_path" in df.columns:
                by_audio[str(r["audio_path"])] = ep
            if "token_loss_path" in df.columns:
                by_token[str(r["token_loss_path"])] = ep
            if "source_token_loss_path" in df.columns:
                by_token[str(r["source_token_loss_path"])] = ep
    # audio key has priority at callsite; keep both dicts packed
    return {"__AUDIO__": by_audio, "__TOKEN__": by_token}


def _load_segments(seg_root: str, experiments: List[str], windows: List[int]) -> Dict[str, Dict[int, pd.DataFrame]]:
    out: Dict[str, Dict[int, pd.DataFrame]] = {}
    for exp in experiments:
        out[exp] = {}
        for w in windows:
            p = os.path.join(seg_root, exp, f"segments_{w}s.csv")
            if os.path.isfile(p):
                out[exp][w] = pd.read_csv(p)
            else:
                out[exp][w] = pd.DataFrame()
    return out


def _plot_one_audio(
    audio_path: str,
    score: float,
    loss_ct: np.ndarray,
    entropy_ct: np.ndarray,
    seg_data: Dict[str, Dict[int, pd.DataFrame]],
    experiments: List[str],
    windows: List[int],
    tokens_per_second: float,
    out_png: str,
) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(14, 24), sharex=False)
    fig.suptitle(f"{os.path.basename(audio_path)} | human score={score:.3f}", fontsize=14)

    # 1/2/5s model score curves
    for ax_idx, w in enumerate(windows):
        ax = axes[ax_idx]
        seen_labels = set()
        for exp in experiments:
            df = seg_data.get(exp, {}).get(w, pd.DataFrame())
            if df.empty:
                continue
            sub = df[df["audio_path"] == audio_path].copy()
            if sub.empty:
                continue
            x = ((sub["start_sec"].to_numpy(dtype=float) + sub["end_sec"].to_numpy(dtype=float)) / 2.0)
            y = sub["pred"].to_numpy(dtype=float)
            label = exp if exp not in seen_labels else "_nolegend_"
            ax.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=label)
            seen_labels.add(exp)
        ax.set_title(f"Segment score curves ({w}s window)")
        ax.set_ylabel("Pred score")
        ax.set_xlim(0.0, 15.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # loss codebook curves
    ax_loss = axes[3]
    t_loss = np.arange(loss_ct.shape[1], dtype=np.float32) / float(tokens_per_second)
    cb_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    for k in range(min(4, loss_ct.shape[0])):
        ax_loss.plot(t_loss, loss_ct[k], linewidth=1.0, alpha=0.8, color=cb_colors[k], label=f"loss_cb{k}")
    ax_loss.plot(t_loss, loss_ct[:4].mean(axis=0), linewidth=2.0, color="tab:blue", label="loss_mean_4cb")
    ax_loss.set_title("Loss curve (4 codebooks + mean)")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlim(0.0, 15.0)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best", fontsize=8)

    # entropy codebook curves
    ax_ent = axes[4]
    t_ent = np.arange(entropy_ct.shape[1], dtype=np.float32) / float(tokens_per_second)
    for k in range(min(4, entropy_ct.shape[0])):
        ax_ent.plot(t_ent, entropy_ct[k], linewidth=1.0, alpha=0.8, color=cb_colors[k], label=f"entropy_cb{k}")
    ax_ent.plot(t_ent, entropy_ct[:4].mean(axis=0), linewidth=2.0, color="black", label="entropy_mean_4cb")
    ax_ent.set_title("Entropy curve (4 codebooks + mean)")
    ax_ent.set_ylabel("Entropy")
    ax_ent.set_xlim(0.0, 15.0)
    ax_ent.grid(True, alpha=0.3)
    ax_ent.legend(loc="best", fontsize=8)

    # loss cb0 + entropy cb0 overlay
    ax_cb0 = axes[5]
    t_joint = np.arange(min(loss_ct.shape[1], entropy_ct.shape[1]), dtype=np.float32) / float(tokens_per_second)
    if t_joint.size > 0:
        ax_cb0.plot(t_joint, loss_ct[0, : t_joint.size], color="tab:blue", linewidth=1.8, label="loss_cb0")
        ax_cb0.plot(t_joint, entropy_ct[0, : t_joint.size], color="black", linewidth=1.8, label="entropy_cb0")
    ax_cb0.set_title("Overlay: loss cb0 (blue) vs entropy cb0 (black)")
    ax_cb0.set_ylabel("Value")
    ax_cb0.set_xlim(0.0, 15.0)
    ax_cb0.grid(True, alpha=0.3)
    ax_cb0.legend(loc="best", fontsize=8)

    # loss mean + entropy mean overlay
    ax_mean = axes[6]
    if t_joint.size > 0:
        loss_mean = loss_ct[:4, : t_joint.size].mean(axis=0)
        ent_mean = entropy_ct[:4, : t_joint.size].mean(axis=0)
        ax_mean.plot(t_joint, loss_mean, color="tab:blue", linewidth=2.0, label="loss_mean_4cb")
        ax_mean.plot(t_joint, ent_mean, color="black", linewidth=2.0, label="entropy_mean_4cb")
    ax_mean.set_title("Overlay: loss mean(4cb) (blue) vs entropy mean(4cb) (black)")
    ax_mean.set_xlabel("Time (sec)")
    ax_mean.set_ylabel("Value")
    ax_mean.set_xlim(0.0, 15.0)
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="clean", choices=["clean"])
    ap.add_argument("--reports-dir", default="")
    ap.add_argument("--tokens-per-second", type=float, default=100.0)
    ap.add_argument("--experiments", action="append", default=[])
    ap.add_argument("--windows", default="1,2,5")
    ap.add_argument("--max-audios", type=int, default=0)
    args = ap.parse_args()

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    project_root = resolve_project_root(cfg)
    split_paths = get_exp11_split_paths(args.config, splits=args.splits)
    test_df = pd.read_csv(split_paths["test"])
    if args.max_audios > 0:
        test_df = test_df.head(args.max_audios).copy()

    out_cfg = cfg.get("outputs", {})
    reports_dir = args.reports_dir or out_cfg.get("reports", "outputs/reports")
    reports_dir = reports_dir if os.path.isabs(reports_dir) else os.path.join(project_root, reports_dir)
    seg_root = os.path.join(reports_dir, "segment_curves")
    plot_root = os.path.join(seg_root, "all_audio_curve_panels")
    os.makedirs(plot_root, exist_ok=True)

    entropy_root = out_cfg.get("features_entropy", "outputs/features/entropy")
    entropy_root = entropy_root if os.path.isabs(entropy_root) else os.path.join(project_root, entropy_root)
    emaps = _entropy_maps(
        [
            os.path.join(entropy_root, "entropy_manifest_train.csv"),
            os.path.join(entropy_root, "entropy_manifest_val.csv"),
            os.path.join(entropy_root, "entropy_manifest_test.csv"),
        ]
    )
    by_audio = emaps["__AUDIO__"]
    by_token = emaps["__TOKEN__"]

    experiments = args.experiments or [
        f"f01_loss_only_cnn_{args.splits}",
        f"f02_entropy_only_cnn_{args.splits}",
        f"f04_loss_entropy_cnn_{args.splits}",
    ]
    seg_data = _load_segments(seg_root=seg_root, experiments=experiments, windows=windows)

    index_rows = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="plot-audio-panels"):
        audio_path = str(row["audio_path"])
        token_loss_path = str(row["token_loss_path"])
        score = float(row["score"])

        try:
            loss_ct = read_loss_codebook_curves(token_loss_path)  # [4, T]
        except Exception:
            continue
        ep = by_audio.get(audio_path) or by_token.get(token_loss_path)
        if not ep or (not os.path.isfile(ep)):
            continue
        try:
            entropy_ct = read_entropy_codebook_curves(ep)  # [4, T]
        except Exception:
            continue

        safe_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_png = os.path.join(plot_root, f"{safe_name}_panel.png")
        _plot_one_audio(
            audio_path=audio_path,
            score=score,
            loss_ct=loss_ct,
            entropy_ct=entropy_ct,
            seg_data=seg_data,
            experiments=experiments,
            windows=windows,
            tokens_per_second=args.tokens_per_second,
            out_png=out_png,
        )
        index_rows.append(
            {
                "audio_path": audio_path,
                "score": score,
                "token_loss_path": token_loss_path,
                "entropy_curve_path": ep,
                "panel_png": out_png,
            }
        )

    index_csv = os.path.join(plot_root, "panel_index.csv")
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)
    print(f"[done] wrote {index_csv} ({len(index_rows)} rows)")


if __name__ == "__main__":
    main()
