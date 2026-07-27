#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from lib.repro.curve_channels import (
    NUM_CODEBOOKS,
    pack_curves_with_mask,
    read_entropy_codebook_curves,
    read_loss_codebook_curves,
)
from lib.repro.data_paths import get_exp11_split_paths, resolve_project_root
from lib.repro.metrics import safe_pearson_spearman
from lib.repro.nets import LossCurveCNN


@dataclass
class LoadedModel:
    model: torch.nn.Module
    architecture: str  # "cnn" (transformer was removed)
    mode: str  # loss | entropy | loss_entropy


def expected_experiments(split_tag: str) -> List[str]:
    return [
        f"f01_loss_only_cnn_{split_tag}",
        f"f02_entropy_only_cnn_{split_tag}",
        f"f03_sae_only_cnn_{split_tag}",
        f"f04_loss_entropy_cnn_{split_tag}",
        f"f05_entropy_sae_cnn_{split_tag}",
        f"f06_loss_sae_cnn_{split_tag}",
        f"f07_loss_entropy_sae_cnn_{split_tag}",
    ]


def _mode_for_experiment(exp_name: str) -> Optional[str]:
    if "_loss_only_" in exp_name:
        return "loss"
    if "_entropy_only_" in exp_name:
        return "entropy"
    if "_loss_entropy_" in exp_name:
        return "loss_entropy"
    return None


def _arch_for_experiment(exp_name: str) -> Optional[str]:
    if exp_name.endswith("_cnn_clean"):
        return "cnn"
    return None


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


def _resolve_entropy_path(token_loss_path: str, audio_path: str, by_token: Dict[str, str], by_audio: Dict[str, str]) -> str:
    ep = by_token.get(token_loss_path)
    if ep is None:
        ep = by_audio.get(audio_path)
    return ep or ""


def _build_window_features(
    mode: str,
    loss_curves: np.ndarray,
    entropy_curves: Optional[np.ndarray],
    start: int,
    end: int,
    max_len: int,
) -> np.ndarray:
    if mode == "loss":
        x, _ = pack_curves_with_mask(loss_curves[:, start:end], max_len=max_len)  # [5, L]
        return x
    if mode == "entropy":
        e = entropy_curves if entropy_curves is not None else np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
        x, _ = pack_curves_with_mask(e[:, start:end], max_len=max_len)  # [5, L]
        return x
    if mode == "loss_entropy":
        e = entropy_curves if entropy_curves is not None else np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
        l = loss_curves[:, start:end]
        ee = e[:, start:end]
        joint = min(int(l.shape[1]), int(ee.shape[1]), max_len)
        x = np.zeros((NUM_CODEBOOKS * 2 + 1, max_len), dtype=np.float32)  # [9, L]
        if joint > 0:
            x[:NUM_CODEBOOKS, :joint] = l[:, :joint]
            x[NUM_CODEBOOKS : NUM_CODEBOOKS * 2, :joint] = ee[:, :joint]
            x[-1, :joint] = 1.0
        return x
    raise ValueError(f"Unsupported mode: {mode}")


def _load_model(exp_name: str, ckpt_dir: str, device: torch.device, seq_len: int) -> Tuple[Optional[LoadedModel], str]:
    mode = _mode_for_experiment(exp_name)
    arch = _arch_for_experiment(exp_name)
    if mode is None or arch is None:
        return None, "unsupported_experiment_name"
    if "_sae_" in exp_name:
        return None, "sae_segment_scoring_not_implemented"

    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_best.pth")
    if not os.path.isfile(ckpt_path):
        return None, "missing_checkpoint"

    in_ch = {
        "loss": NUM_CODEBOOKS + 1,
        "entropy": NUM_CODEBOOKS + 1,
        "loss_entropy": NUM_CODEBOOKS * 2 + 1,
    }[mode]
    if arch != "cnn":
        return None, "transformer_backbone_removed"
    model = LossCurveCNN(input_channels=in_ch, sequence_length=seq_len).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return LoadedModel(model=model, architecture=arch, mode=mode), "ok"


def _predict_one(loaded: LoadedModel, feat_cl: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        x = torch.from_numpy(feat_cl).float().unsqueeze(0).to(device)  # [1, C, L]
        y = loaded.model(x)
        return float(y.detach().cpu().view(-1)[0].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="clean", choices=["clean"])
    parser.add_argument("--experiments", action="append", default=[])
    parser.add_argument("--window-seconds", default="2,5", help="Comma-separated, e.g. 2,5")
    parser.add_argument("--hop-seconds", default="", help="Comma-separated or empty (=window)")
    parser.add_argument("--tokens-per-second", type=float, default=100.0)
    parser.add_argument("--max-audios", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    split_tag = args.splits
    project_root = resolve_project_root(cfg)
    seq_len = int(cfg.get("seq_len", 1500))
    out_cfg = cfg.get("outputs", {})
    reports_dir = out_cfg.get("reports", "outputs/reports")
    reports_dir = reports_dir if os.path.isabs(reports_dir) else os.path.join(project_root, reports_dir)
    ckpt_dir = out_cfg.get("checkpoints", "outputs/checkpoints")
    ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(project_root, ckpt_dir)
    entropy_root = out_cfg.get("features_entropy", "outputs/features/entropy")
    entropy_root = entropy_root if os.path.isabs(entropy_root) else os.path.join(project_root, entropy_root)

    split_paths = get_exp11_split_paths(args.config, splits=split_tag)
    test_df = pd.read_csv(split_paths["test"])
    if args.max_audios > 0:
        test_df = test_df.head(args.max_audios).copy()

    windows = [float(x.strip()) for x in args.window_seconds.split(",") if x.strip()]
    hops = [float(x.strip()) for x in args.hop_seconds.split(",") if x.strip()] if args.hop_seconds else windows
    if len(hops) != len(windows):
        raise ValueError("hop-seconds count must match window-seconds count")

    manifests = [
        os.path.join(entropy_root, "entropy_manifest_train.csv"),
        os.path.join(entropy_root, "entropy_manifest_val.csv"),
        os.path.join(entropy_root, "entropy_manifest_test.csv"),
    ]
    entropy_by_token, entropy_by_audio = _build_entropy_maps(manifests)

    experiments = args.experiments if args.experiments else expected_experiments(split_tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_root = os.path.join(reports_dir, "segment_curves")
    os.makedirs(segment_root, exist_ok=True)

    summary_rows = []
    for exp_name in experiments:
        loaded, status = _load_model(exp_name, ckpt_dir=ckpt_dir, device=device, seq_len=seq_len)
        if loaded is None:
            summary_rows.append(
                {
                    "experiment": exp_name,
                    "window_sec": "",
                    "status": status,
                    "audio_count": 0,
                    "segment_count": 0,
                    "pearson_audio_mean": float("nan"),
                    "spearman_audio_mean": float("nan"),
                }
            )
            continue

        exp_out_dir = os.path.join(segment_root, exp_name)
        os.makedirs(exp_out_dir, exist_ok=True)
        print(f"[segment-score] experiment={exp_name} mode={loaded.mode} arch={loaded.architecture}")

        for w_sec, h_sec in zip(windows, hops):
            win_tokens = max(1, int(round(w_sec * args.tokens_per_second)))
            hop_tokens = max(1, int(round(h_sec * args.tokens_per_second)))
            seg_rows = []
            audio_means = []
            for _, row in test_df.iterrows():
                score = float(row["score"])
                audio_path = str(row["audio_path"])
                token_loss_path = str(row["token_loss_path"])
                try:
                    loss_curves = read_loss_codebook_curves(token_loss_path, num_codebooks=NUM_CODEBOOKS)
                except Exception:
                    continue
                if loss_curves.shape[1] == 0:
                    continue

                entropy_curves = None
                if loaded.mode in {"entropy", "loss_entropy"}:
                    ep = _resolve_entropy_path(token_loss_path, audio_path, entropy_by_token, entropy_by_audio)
                    if not ep or (not os.path.isfile(ep)):
                        continue
                    try:
                        entropy_curves = read_entropy_codebook_curves(ep, num_codebooks=NUM_CODEBOOKS)
                    except Exception:
                        continue
                    if entropy_curves.shape[1] == 0:
                        continue

                t = int(loss_curves.shape[1]) if loaded.mode == "loss" else int(min(loss_curves.shape[1], entropy_curves.shape[1]))
                if t <= 0:
                    continue

                starts = list(range(0, max(1, t - win_tokens + 1), hop_tokens))
                if not starts:
                    starts = [0]
                preds_audio = []
                for seg_idx, st in enumerate(starts):
                    ed = min(st + win_tokens, t)
                    feat = _build_window_features(
                        mode=loaded.mode,
                        loss_curves=loss_curves,
                        entropy_curves=entropy_curves,
                        start=st,
                        end=ed,
                        max_len=seq_len,
                    )
                    pred = _predict_one(loaded, feat_cl=feat, device=device)
                    preds_audio.append(pred)
                    seg_rows.append(
                        {
                            "audio_path": audio_path,
                            "score": score,
                            "window_sec": w_sec,
                            "start_token": st,
                            "end_token": ed,
                            "start_sec": st / args.tokens_per_second,
                            "end_sec": ed / args.tokens_per_second,
                            "segment_idx": seg_idx,
                            "pred": pred,
                        }
                    )
                if preds_audio:
                    audio_means.append((score, float(np.mean(preds_audio))))

            seg_csv = os.path.join(exp_out_dir, f"segments_{int(w_sec)}s.csv")
            pd.DataFrame(seg_rows).to_csv(seg_csv, index=False)
            if audio_means:
                y_true = np.array([x[0] for x in audio_means], dtype=np.float32)
                y_pred = np.array([x[1] for x in audio_means], dtype=np.float32)
                pr, sp = safe_pearson_spearman(y_true, y_pred)
            else:
                pr, sp = float("nan"), float("nan")
            summary_rows.append(
                {
                    "experiment": exp_name,
                    "window_sec": w_sec,
                    "status": "ok" if len(seg_rows) > 0 else "no_segments",
                    "audio_count": len(audio_means),
                    "segment_count": len(seg_rows),
                    "pearson_audio_mean": pr,
                    "spearman_audio_mean": sp,
                    "segments_csv": seg_csv,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(segment_root, f"segment_curve_summary_{split_tag}.csv")
    summary_md = os.path.join(segment_root, f"segment_curve_summary_{split_tag}.md")
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("| experiment | window_sec | status | audio_count | segment_count | pearson_audio_mean | spearman_audio_mean |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|\n")
        for _, r in summary_df.iterrows():
            f.write(
                f"| {r.get('experiment','')} | {r.get('window_sec','')} | {r.get('status','')} | "
                f"{r.get('audio_count',0)} | {r.get('segment_count',0)} | "
                f"{r.get('pearson_audio_mean',float('nan'))} | {r.get('spearman_audio_mean',float('nan'))} |\n"
            )
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {summary_md}")


if __name__ == "__main__":
    main()
