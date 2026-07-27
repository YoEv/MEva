#!/usr/bin/env python3
"""Extract per-codebook MusicGen per-token cross-entropy curves for every
clip in a split, honouring the same rater-aligned / chunked / pooled
window conventions as the entropy extractor.

See ``extract_entropy_curves.py`` for the full rationale; the loss
extractor shares the audio-loading logic via
``lib/repro/audio_window.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import MusicgenForConditionalGeneration
from transformers.models.musicgen.modeling_musicgen import MusicgenSinusoidalPositionalEmbedding

if not hasattr(MusicgenSinusoidalPositionalEmbedding, "_offset_patched"):
    _orig_init = MusicgenSinusoidalPositionalEmbedding.__init__
    def _patched_init(self, num_positions, embedding_dim):
        self.offset = 2
        _orig_init(self, num_positions, embedding_dim)
    MusicgenSinusoidalPositionalEmbedding.__init__ = _patched_init
    MusicgenSinusoidalPositionalEmbedding._offset_patched = True

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.repro.audio_window import (  # noqa: E402
    SR_OUT,
    get_row_window,
    load_audio_mono,
    uniform_pool_time_last,
)
from lib.repro.data_paths import get_exp11_split_paths  # noqa: E402
from lib.repro.musicgen_pattern import delayed_aligned_logits  # noqa: E402

NUM_CODEBOOKS = 4


def _uid(token_loss_path: str) -> str:
    return hashlib.md5(token_loss_path.encode("utf-8")).hexdigest()[:16]


def _loss_from_one_chunk(
    model: MusicgenForConditionalGeneration,
    wav_chunk: torch.Tensor,
    pattern: str = "parallel",
) -> np.ndarray:
    """Return ``[K, T-1]`` per-codebook cross-entropy for a single chunk.

    ``pattern="parallel"``: legacy protocol, aligned codes fed directly.
    ``pattern="delay"``: MusicGen-native delayed interleaving for teacher
    forcing; logits re-aligned to audio frames (see lib/repro/musicgen_pattern.py).
    Empty/short chunks return an empty ``(K, 0)`` array.
    """
    with torch.no_grad():
        enc = model.audio_encoder.encode(wav_chunk.unsqueeze(0))
        codes = enc.audio_codes.long()
        if codes.ndim != 4:
            return np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
        bsz, channels, codebooks, t_all = codes.shape
        if bsz != 1 or codebooks < NUM_CODEBOOKS or t_all <= 1:
            return np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
        if pattern == "delay":
            logits = delayed_aligned_logits(model, codes)[:, :NUM_CODEBOOKS, :, :]
        else:
            inp = codes[:, :, :, :-1].contiguous().view(bsz, channels * codebooks, t_all - 1)
            out = model.decoder(inp, output_hidden_states=False)
            logits = out.logits
            if logits.dim() == 4:
                logits = logits[0]
            logits = logits.view(channels, codebooks, logits.shape[1], logits.shape[2])[:, :NUM_CODEBOOKS, :, :]
        target = codes[0, :, :NUM_CODEBOOKS, 1:]
        t = min(int(logits.shape[2]), int(target.shape[2]))
        if t <= 0:
            return np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
        ce_rows: List[np.ndarray] = []
        for ch in range(channels):
            logits_ch = logits[ch, :, :t, :].reshape(NUM_CODEBOOKS * t, logits.shape[-1])
            target_ch = target[ch, :, :t].reshape(NUM_CODEBOOKS * t)
            ce_flat = F.cross_entropy(logits_ch, target_ch, reduction="none")
            ce_rows.append(ce_flat.view(NUM_CODEBOOKS, t).detach().cpu().numpy().astype(np.float32))
        return np.stack(ce_rows, axis=0).mean(axis=0).astype(np.float32)


def _extract_loss_curve(
    model: MusicgenForConditionalGeneration,
    audio_path: str,
    device: torch.device,
    *,
    begin_s: Optional[float],
    end_s: Optional[float],
    max_audio_sec: float,
    chunk_sec: float,
    pattern: str = "parallel",
) -> np.ndarray:
    """Per-codebook loss curve over the whole (optionally chunked) window."""
    wav = load_audio_mono(
        audio_path, begin_s=begin_s, end_s=end_s,
        max_audio_sec=max_audio_sec, device=device,
    )  # [1, S]
    total = wav.size(-1)
    chunk_samples = int(round(chunk_sec * SR_OUT)) if chunk_sec > 0 else total
    n_chunks = max(1, (total + chunk_samples - 1) // chunk_samples)
    parts: List[np.ndarray] = []
    for c_idx in range(n_chunks):
        a = c_idx * chunk_samples
        b = min(a + chunk_samples, total)
        if b - a < int(0.1 * SR_OUT):
            continue
        part = _loss_from_one_chunk(model, wav[..., a:b], pattern=pattern)
        if part.shape[1] > 0:
            parts.append(part)
    if not parts:
        return np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1)


def _validate_curve_csv(path: str, mode: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if len(df) == 0:
        return False
    if "token_position" not in df.columns:
        return False
    if mode == "per_codebook":
        req = [f"codebook_{i}" for i in range(NUM_CODEBOOKS)]
        if any(c not in df.columns for c in req):
            return False
    if "avg_loss_value" not in df.columns:
        return False
    return True


def _write_curve_csv(out_csv: str, curve_kt: np.ndarray, mode: str) -> Dict[str, float]:
    curve_kt = curve_kt.astype(np.float32, copy=False)
    k, t = curve_kt.shape
    rows = {"token_position": np.arange(t, dtype=np.int32), "avg_loss_value": curve_kt.mean(axis=0)}
    if mode == "per_codebook":
        for i in range(min(k, NUM_CODEBOOKS)):
            rows[f"codebook_{i}"] = curve_kt[i, :]
    df = pd.DataFrame(rows)
    if mode == "per_codebook":
        ordered = ["token_position"] + [f"codebook_{i}" for i in range(NUM_CODEBOOKS)] + ["avg_loss_value"]
        for c in ordered:
            if c not in df.columns:
                df[c] = np.nan
        df = df[ordered]
    else:
        df = df[["token_position", "avg_loss_value"]]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return {
        "mean_loss": float(df["avg_loss_value"].mean()) if len(df) > 0 else float("nan"),
        "time_steps": int(t),
    }


def _resolve_split_csv(args: argparse.Namespace) -> str:
    if args.source_split_csv:
        return os.path.abspath(args.source_split_csv)
    return os.path.abspath(get_exp11_split_paths(args.config, splits=args.splits)[args.split])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"],
                        help="Only 'clean' is supported (noisy was removed).")
    parser.add_argument("--source-split-csv", type=str, default="")
    parser.add_argument("--output-split-csv", type=str, default="")
    parser.add_argument("--manifest-csv", type=str, default="")
    parser.add_argument("--out-dir", type=str,
                        default="outputs/features/loss/musiceval/clean")
    parser.add_argument("--musicgen-model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--mode", type=str, choices=["per_codebook", "avg"], default="per_codebook")
    parser.add_argument("--pattern", type=str, choices=["parallel", "delay"], default="delay",
                        help="delay=paper protocol (MusicGen-native interleaving); parallel=legacy")
    parser.add_argument("--chunk-sec", type=float, default=0.0,
                        help="If > 0, compute the loss curve on non-overlapping "
                             "windows of this many seconds and concatenate.")
    parser.add_argument("--pool-to-frames", type=int, default=0,
                        help="Uniformly mean-pool the concatenated loss curve "
                             "to this many frames (use with --chunk-sec).")
    parser.add_argument("--max-audio-sec", type=float, default=0.0,
                        help="Cap on audio duration in seconds (0 = unlimited).")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    split_csv = _resolve_split_csv(args)
    df = pd.read_csv(split_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    out_dir = os.path.abspath(args.out_dir)
    curve_dir = os.path.join(out_dir, args.split)
    os.makedirs(curve_dir, exist_ok=True)
    manifest_path = os.path.abspath(args.manifest_csv) if args.manifest_csv else os.path.join(out_dir, f"loss_manifest_{args.split}.csv")
    output_split_csv = os.path.abspath(args.output_split_csv) if args.output_split_csv else os.path.join(out_dir, f"split_{args.split}.csv")

    expected_paths = [
        os.path.abspath(os.path.join(curve_dir, f"{_uid(str(r['token_loss_path']))}_loss.csv"))
        for _, r in df.iterrows()
    ]
    have_curves = sum(1 for p in expected_paths if _validate_curve_csv(p, args.mode))
    print(f"[paths] split={args.split} out_dir={out_dir} curve_dir={curve_dir} manifest={manifest_path} output_split={output_split_csv}")
    print(f"[progress] split={args.split} need={len(df)} have={have_curves} missing={max(0, len(df)-have_curves)}")
    print(f"[window] chunk_sec={args.chunk_sec} pool_to_frames={args.pool_to_frames} max_audio_sec={args.max_audio_sec}")

    def _write_relinked_split() -> None:
        out_df = df.copy()
        out_df["token_loss_path"] = expected_paths
        os.makedirs(os.path.dirname(output_split_csv), exist_ok=True)
        out_df.to_csv(output_split_csv, index=False)

    def _manifest_complete() -> bool:
        if not os.path.isfile(manifest_path):
            return False
        try:
            mdf = pd.read_csv(manifest_path)
        except Exception:
            return False
        if len(mdf) != len(df) or "loss_curve_path" not in mdf.columns:
            return False
        paths = mdf["loss_curve_path"].astype(str).tolist()
        return len(paths) == len(df) and all(_validate_curve_csv(p, args.mode) for p in paths)

    def _build_manifest_from_existing() -> bool:
        rows = []
        for (_, row), out_csv in zip(df.iterrows(), expected_paths):
            if not _validate_curve_csv(out_csv, args.mode):
                return False
            cdf = pd.read_csv(out_csv)
            rows.append({
                "audio_path": str(row["audio_path"]),
                "source_token_loss_path": str(row["token_loss_path"]),
                "loss_curve_path": out_csv,
                "mean_loss": float(pd.to_numeric(cdf["avg_loss_value"], errors="coerce").dropna().mean()),
                "time_steps": int(len(cdf)),
            })
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        _write_relinked_split()
        print(f"[skip-existing] {args.split}: rebuilt manifest+split from existing curves")
        return True

    if args.skip_existing and _manifest_complete() and os.path.isfile(output_split_csv):
        try:
            odf = pd.read_csv(output_split_csv)
            if len(odf) == len(df) and "token_loss_path" in odf.columns:
                if all(_validate_curve_csv(p, args.mode) for p in odf["token_loss_path"].astype(str).tolist()):
                    print(f"[skip-existing] {args.split}: complete manifest and relinked split already present")
                    return
        except Exception:
            pass
    if args.skip_existing and _build_manifest_from_existing():
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicgenForConditionalGeneration.from_pretrained(args.musicgen_model).to(device)
    model.eval()

    records = []
    reused = 0
    recomputed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"loss-{args.splits}-{args.split}"):
        source_token_loss = str(row["token_loss_path"])
        audio_path = str(row["audio_path"])
        out_csv = os.path.abspath(os.path.join(curve_dir, f"{_uid(source_token_loss)}_loss.csv"))
        begin_s, end_s = get_row_window(row)
        if args.skip_existing and _validate_curve_csv(out_csv, args.mode):
            cdf = pd.read_csv(out_csv)
            mean_loss = float(pd.to_numeric(cdf["avg_loss_value"], errors="coerce").dropna().mean())
            time_steps = int(len(cdf))
            reused += 1
        else:
            curve_kt = _extract_loss_curve(
                model, audio_path, device=device,
                begin_s=begin_s, end_s=end_s,
                max_audio_sec=args.max_audio_sec,
                chunk_sec=args.chunk_sec,
                pattern=args.pattern,
            )
            if args.pool_to_frames > 0 and curve_kt.shape[1] > args.pool_to_frames:
                curve_kt = uniform_pool_time_last(curve_kt, args.pool_to_frames)
            stats = _write_curve_csv(out_csv, curve_kt=curve_kt, mode=args.mode)
            mean_loss = stats["mean_loss"]
            time_steps = stats["time_steps"]
            recomputed += 1
        records.append({
            "audio_path": audio_path,
            "source_token_loss_path": source_token_loss,
            "loss_curve_path": out_csv,
            "mean_loss": mean_loss,
            "time_steps": time_steps,
        })

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    _write_relinked_split()
    print(f"Wrote {manifest_path} ({len(records)} rows)")
    print(f"Wrote {output_split_csv} ({len(df)} rows)")
    print(f"[resume-stats] {args.split}: reused={reused}, recomputed={recomputed}")


if __name__ == "__main__":
    main()
