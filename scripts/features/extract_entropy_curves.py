"""Extract per-codebook MusicGen entropy curves for every clip in a split.

Honours the following per-dataset conventions:

* ``begin_s`` / ``end_s`` columns (AIME, Music Arena) crop the audio to the
  rater-aligned window before extraction.
* ``--chunk-sec`` processes long clips (SongEval full songs) as a sequence
  of non-overlapping ``chunk_sec``-second windows and concatenates the
  entropy curves along time.
* ``--pool-to-frames`` uniformly mean-pools the concatenated curve down to
  a fixed number of frames -- essential for SongEval so the CNN sees a
  fixed-shape summary of the whole song regardless of duration.
* ``--max-audio-sec`` is an optional upper bound (``0`` = unlimited).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MusicgenForConditionalGeneration
from transformers.models.musicgen.modeling_musicgen import MusicgenSinusoidalPositionalEmbedding

# transformers 4.38.1 bug: offset is referenced in forward() but never set in __init__
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


def _entropy_from_logits_multi_codebook(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), dim=-1)
    log_p = torch.log(p.clamp_min(1e-12))
    return -(p * log_p).sum(dim=-1)


def _fix_len_kt(h_kt: np.ndarray, fixed_t: int) -> np.ndarray:
    h_kt = h_kt.astype(np.float32, copy=False)
    if h_kt.ndim == 1:
        h_kt = h_kt[None, :]
    if h_kt.shape[0] > NUM_CODEBOOKS:
        h_kt = h_kt[:NUM_CODEBOOKS, :]
    elif h_kt.shape[0] < NUM_CODEBOOKS:
        pad_k = np.zeros((NUM_CODEBOOKS - h_kt.shape[0], h_kt.shape[1]), dtype=np.float32)
        h_kt = np.concatenate([h_kt, pad_k], axis=0)
    if fixed_t <= 0:
        return h_kt
    if h_kt.shape[1] >= fixed_t:
        return h_kt[:, :fixed_t].copy()
    out = np.full((h_kt.shape[0], fixed_t), np.nan, dtype=np.float32)
    out[:, : h_kt.shape[1]] = h_kt
    return out


def _extract_entropy_curve(
    model: MusicgenForConditionalGeneration,
    audio_path: str,
    device: torch.device,
    *,
    begin_s: float | None,
    end_s: float | None,
    max_audio_sec: float,
    chunk_sec: float,
    pattern: str = "parallel",
) -> np.ndarray:
    """Return ``[K, T]`` per-codebook entropy, optionally chunked."""
    wav = load_audio_mono(
        audio_path, begin_s=begin_s, end_s=end_s,
        max_audio_sec=max_audio_sec, device=device,
    )  # [1, S]
    chunk_samples = int(round(chunk_sec * SR_OUT)) if chunk_sec > 0 else wav.size(-1)
    total = wav.size(-1)
    n_chunks = max(1, (total + chunk_samples - 1) // chunk_samples)

    curves: list[np.ndarray] = []
    with torch.no_grad():
        for c_idx in range(n_chunks):
            a = c_idx * chunk_samples
            b = min(a + chunk_samples, total)
            # <100 ms tail is too short for MusicGen's encoder.
            if b - a < int(0.1 * SR_OUT):
                continue
            sub = wav[..., a:b].unsqueeze(0)
            enc = model.audio_encoder.encode(sub)
            codes = enc.audio_codes.long()
            bsz, c, k, t = codes.shape
            if t < 2:
                continue
            if pattern == "delay":
                logits = delayed_aligned_logits(model, codes).squeeze(0)  # [K, T-1, V]
            else:
                inp = codes[:, :, :, :-1].contiguous().view(bsz, c * k, t - 1)
                out = model.decoder(inp, output_hidden_states=False)
                logits = out.logits
                if logits.dim() == 4:
                    logits = logits.squeeze(0)
            ent_kt = _entropy_from_logits_multi_codebook(logits)  # [K, T-1]
            curves.append(ent_kt.detach().cpu().numpy().astype(np.float32))
    if not curves:
        return np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
    return np.concatenate(curves, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"],
                        help="Only 'clean' is supported (noisy was removed).")
    parser.add_argument("--musicgen-model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--pattern", type=str, choices=["parallel", "delay"], default="delay",
                        help="delay=paper protocol (MusicGen-native interleaving); parallel=legacy")
    parser.add_argument("--fixed-time-steps", type=int, default=1500,
                        help="Pad/truncate final curve to this many frames "
                             "(matches the CNN's receptive field).")
    parser.add_argument("--chunk-sec", type=float, default=0.0,
                        help="If > 0, run MusicGen on non-overlapping windows "
                             "of this many seconds and concatenate entropy "
                             "curves along time (full-song mode).")
    parser.add_argument("--pool-to-frames", type=int, default=0,
                        help="If > 0, uniformly mean-pool the concatenated "
                             "entropy curve to this many frames before the "
                             "final fixed-length fit (use with --chunk-sec).")
    parser.add_argument("--max-audio-sec", type=float, default=0.0,
                        help="Cap on audio duration in seconds (0 = unlimited).")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default="outputs/features/entropy/musiceval/clean")
    parser.add_argument("--skip-existing",
                        action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split_csv = get_exp11_split_paths(args.config, splits=args.splits)[args.split]
    df = pd.read_csv(split_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    curve_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(curve_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, f"entropy_manifest_{args.split}.csv")

    def _maybe_build_manifest_from_existing() -> bool:
        rows = []
        for _, row in df.iterrows():
            token_loss_path = str(row["token_loss_path"])
            audio_path = str(row["audio_path"])
            uid = _uid(token_loss_path)
            npy_path = os.path.abspath(os.path.join(curve_dir, f"{uid}_entropy.npy"))
            if not os.path.isfile(npy_path):
                return False
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim != 2:
                return False
            rows.append({
                "audio_path": audio_path,
                "token_loss_path": token_loss_path,
                "entropy_curve_path": npy_path,
                "mean_entropy": float(np.nanmean(arr)),
                "time_steps": int(arr.shape[1]),
            })
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print(f"[skip-existing] {args.split}: rebuilt missing manifest from existing curves -> {manifest_path}")
        return True

    existing_curve_count = (
        len([p for p in os.listdir(curve_dir) if p.endswith("_entropy.npy")])
        if os.path.isdir(curve_dir) else 0
    )
    print(f"[paths] split={args.split} out_dir={args.out_dir} curve_dir={curve_dir} manifest={manifest_path}")
    print(f"[progress] split={args.split} need={len(df)} have={existing_curve_count} missing={max(0, len(df)-existing_curve_count)}")
    print(f"[window] chunk_sec={args.chunk_sec} pool_to_frames={args.pool_to_frames} max_audio_sec={args.max_audio_sec} fixed_time_steps={args.fixed_time_steps}")

    if args.skip_existing and os.path.isfile(manifest_path):
        try:
            mdf = pd.read_csv(manifest_path)
            if len(mdf) == len(df) and "entropy_curve_path" in mdf.columns:
                paths = mdf["entropy_curve_path"].dropna().astype(str).tolist()
                if len(paths) == len(df) and all(os.path.isfile(p) for p in paths):
                    print(f"[skip-existing] {args.split}: found complete entropy manifest {manifest_path} (rows={len(mdf)})")
                    return
        except Exception:
            pass
    if args.skip_existing and _maybe_build_manifest_from_existing():
        return
    if args.skip_existing:
        print(f"[resume-info] {args.split}: existing curves incomplete ({existing_curve_count}/{len(df)}); continue extraction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicgenForConditionalGeneration.from_pretrained(args.musicgen_model).to(device)
    model.eval()

    records = []
    reused = 0
    recomputed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"entropy-{args.splits}-{args.split}"):
        token_loss_path = str(row["token_loss_path"])
        audio_path = str(row["audio_path"])
        uid = _uid(token_loss_path)
        out_npy = os.path.abspath(os.path.join(curve_dir, f"{uid}_entropy.npy"))
        begin_s, end_s = get_row_window(row)

        def _compute() -> np.ndarray:
            raw = _extract_entropy_curve(
                model, audio_path, device=device,
                begin_s=begin_s, end_s=end_s,
                max_audio_sec=args.max_audio_sec,
                chunk_sec=args.chunk_sec,
                pattern=args.pattern,
            )
            if args.pool_to_frames > 0 and raw.shape[1] > args.pool_to_frames:
                raw = uniform_pool_time_last(raw, args.pool_to_frames)
            return _fix_len_kt(raw, args.fixed_time_steps)

        if args.skip_existing and os.path.isfile(out_npy):
            try:
                entropy = np.load(out_npy, mmap_mode="r")
                if entropy.ndim == 2:
                    reused += 1
                else:
                    raise ValueError("bad dim")
            except Exception:
                entropy = _compute()
                np.save(out_npy, entropy)
                recomputed += 1
        else:
            entropy = _compute()
            np.save(out_npy, entropy)
            recomputed += 1

        records.append({
            "audio_path": audio_path,
            "token_loss_path": token_loss_path,
            "entropy_curve_path": out_npy,
            "mean_entropy": float(np.nanmean(entropy)),
            "time_steps": int(entropy.shape[1]),
        })

    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path} ({len(records)} rows)")
    print(f"[resume-stats] {args.split}: reused={reused}, recomputed={recomputed}")


if __name__ == "__main__":
    main()
