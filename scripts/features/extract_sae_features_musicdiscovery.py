"""Extract SAE features for every clip in a split using the musicdiscovery
MusicGen hook + pretrained SAE.

Honours the same rater-aligned / chunked / pooled conventions as
``extract_entropy_curves.py``:

* Reads optional ``begin_s`` / ``end_s`` columns from the split CSV
  (AIME + Music Arena) and crops the audio to that window before running
  the hook.
* ``--chunk-sec > 0`` switches to **sequential per-clip extraction**:
  each waveform is processed in non-overlapping ``chunk_sec``-second
  windows, SAE activations are concatenated along time, and
  ``--pool-to-frames > 0`` uniformly mean-pools the concatenated
  activations to a fixed length so the downstream CNN sees a single
  fixed-shape tensor regardless of song duration.  This is what lets
  SongEval (2-6 min full songs) work without OOM and without a 30-s
  prefix cap.
* When ``--chunk-sec == 0`` the batched code-path (originally written
  for MusicEval's short clips) is preserved.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm


MUSICGEN_TOKEN_RATE = 50  # tokens per second at 32 kHz

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lib.repro.audio_window import (  # noqa: E402
    SR_OUT,
    get_row_window,
    load_audio_mono,
    uniform_pool_time_first,
)


def _load_audio_mono_32k(
    path: str,
    *,
    begin_s: Optional[float] = None,
    end_s: Optional[float] = None,
    max_audio_sec: float = 0.0,
):
    """Thin wrapper returning ``(wav[samples], duration_seconds)`` using the
    shared ``load_audio_mono`` helper (no zero-padding, mono @ 32 kHz).
    """
    wav = load_audio_mono(
        path, begin_s=begin_s, end_s=end_s, max_audio_sec=max_audio_sec, sr_out=SR_OUT,
    )
    wav = wav.squeeze(0)
    return wav, wav.size(-1) / SR_OUT


def _extract_one_chunked(
    model, sae, wav_samples: torch.Tensor, device: str,
    chunk_sec: float, pool_to_frames: int,
) -> np.ndarray:
    """Per-clip SAE activations ``[T, D]`` with chunked inference + optional pooling."""
    chunk_samples = int(round(chunk_sec * 32000)) if chunk_sec > 0 else wav_samples.size(-1)
    total = wav_samples.size(-1)
    n_chunks = max(1, (total + chunk_samples - 1) // chunk_samples)
    parts: List[np.ndarray] = []
    with torch.no_grad():
        for c_idx in range(n_chunks):
            a = c_idx * chunk_samples
            b = min(a + chunk_samples, total)
            if b - a < int(0.1 * 32000):
                continue
            sub = wav_samples[a:b].unsqueeze(0).to(device)  # [1, S]
            _, cache = model.run_with_cache(sub, names_filter=[sae.cfg.hook_name])
            act = cache[sae.cfg.hook_name]
            zc = sae.encode(act).squeeze(0).detach().cpu().float().numpy()
            parts.append(zc)
    if not parts:
        return np.zeros((1, int(sae.cfg.d_sae)), dtype=np.float32)
    z = np.concatenate(parts, axis=0)
    if pool_to_frames > 0 and z.shape[0] > pool_to_frames:
        z = uniform_pool_time_first(z, pool_to_frames)
    return z


def _extract_batch(
    model, sae, batch_rows: List[dict], device: str,
    max_seq_len: int, hidden_dim: int, max_audio_sec: float,
) -> List[np.ndarray]:
    """Batched path used when ``chunk_sec == 0`` (MusicEval-style short clips).

    ``batch_rows`` entries are ``{'audio_path': str, 'begin_s': Optional[float],
    'end_s': Optional[float]}``.
    """
    wavs, durs = [], []
    for row in batch_rows:
        try:
            w, d = _load_audio_mono_32k(
                row["audio_path"],
                begin_s=row.get("begin_s"),
                end_s=row.get("end_s"),
                max_audio_sec=max_audio_sec,
            )
        except Exception:
            w = torch.zeros(32000, dtype=torch.float32)
            d = 1.0
        wavs.append(w)
        durs.append(d)

    max_samples = max(w.shape[0] for w in wavs)
    padded = torch.zeros(len(wavs), max_samples)
    for j, w in enumerate(wavs):
        padded[j, : w.shape[0]] = w
    padded = padded.to(device)

    feats: List[np.ndarray] = []
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(padded, names_filter=[sae.cfg.hook_name])
            sae_in = cache[sae.cfg.hook_name]   # [B, T_max, D]
            encoded = sae.encode(sae_in)         # [B, T_max, H]
        for j, dur in enumerate(durs):
            t = max(1, int(dur * MUSICGEN_TOKEN_RATE))
            feat = encoded[j, :t].detach().cpu().float().numpy()
            if max_seq_len > 0 and feat.shape[0] > max_seq_len:
                feat = feat[:max_seq_len]
            if feat.ndim != 2 or feat.shape[1] != hidden_dim:
                feat = np.zeros((1, hidden_dim), dtype=np.float32)
            feats.append(feat)
    except Exception:
        feats = [np.zeros((1, hidden_dim), dtype=np.float32) for _ in batch_rows]
    return feats


def _shard_path(shard_dir: str, idx: int) -> str:
    return os.path.join(shard_dir, f"{idx:06d}.npy")


def _partial_meta_path(out_dir: str, split: str, rank: int) -> str:
    return os.path.join(out_dir, f"_partial_meta_{split}_rank{rank}.pt")


def _write_final_meta(out_dir: str, split: str, world_size: int, out_prefix: str,
                      hidden_dim: int, n_total: int) -> None:
    all_lengths: Dict[int, int] = {}
    for r in range(world_size):
        p = _partial_meta_path(out_dir, split, r)
        part = torch.load(p, map_location="cpu")
        all_lengths.update(part["lengths_by_idx"])

    lengths = [int(all_lengths[i]) for i in range(n_total)]
    shard_dir = os.path.join(out_dir, split)
    out_meta = os.path.join(out_dir, f"{out_prefix}_{split}_meta.pt")
    torch.save({
        "mode": "sharded",
        "shard_dir": shard_dir,
        "lengths": lengths,
        "hidden_dim": hidden_dim,
        "backend": "musicdiscovery",
    }, out_meta)
    for r in range(world_size):
        p = _partial_meta_path(out_dir, split, r)
        if os.path.isfile(p):
            os.remove(p)
    print(f"[meta] wrote {out_meta} samples={n_total}")


# ---------------------------------------------------------------------------
# Monolithic path (legacy; no chunk/pool support)
# ---------------------------------------------------------------------------

def _run_monolithic(args, rows: List[dict], out_dir: str) -> None:
    from sae_components.musicgen_hooked import HookedMusicGen
    from sae_lens import SAE

    if args.chunk_sec > 0 or args.pool_to_frames > 0:
        raise NotImplementedError(
            "--chunk-sec / --pool-to-frames are only implemented for --mode sharded."
        )

    out_npy = os.path.join(out_dir, f"{args.out_prefix}_{args.split}.npy")
    out_meta = os.path.join(out_dir, f"{args.out_prefix}_{args.split}_meta.pt")

    if args.skip_existing and os.path.isfile(out_npy) and os.path.isfile(out_meta):
        try:
            meta_obj = torch.load(out_meta, map_location="cpu")
            lengths = meta_obj.get("lengths", [])
            if isinstance(lengths, list) and len(lengths) == len(rows):
                print(f"[skip-existing] {args.split}: complete monolithic artifacts found")
                return
        except Exception:
            pass

    model = HookedMusicGen(model_name="facebook/musicgen-small", device=args.device)
    sae = SAE.load_from_pretrained(args.checkpoint_dir, device=args.device)
    hidden_dim = int(sae.cfg.d_sae)

    def _extract_one(path: str, begin_s, end_s) -> np.ndarray:
        wav, _ = _load_audio_mono_32k(
            path, begin_s=begin_s, end_s=end_s, max_audio_sec=args.max_audio_sec,
        )
        wav = wav.unsqueeze(0).to(args.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(wav, names_filter=[sae.cfg.hook_name])
            sae_in = cache[sae.cfg.hook_name]
            feat = sae.encode(sae_in).squeeze(0).detach().cpu().float().numpy()
        if args.max_seq_len > 0:
            feat = feat[: args.max_seq_len]
        return feat

    total_est = 0
    for row in tqdm(rows, desc="pre-scan", leave=False):
        try:
            info = sf.info(row["audio_path"])
            t = int(info.frames / info.samplerate * MUSICGEN_TOKEN_RATE)
        except Exception:
            t = 180 * MUSICGEN_TOKEN_RATE
        if args.max_seq_len > 0:
            t = min(t, args.max_seq_len)
        total_est += max(1, t)
    total_est = int(total_est * 1.1)

    arr = np.memmap(out_npy, dtype="float32", mode="w+", shape=(total_est, hidden_dim))
    lengths_all, frame_offset = [], 0
    for row in tqdm(rows, desc=args.split):
        try:
            feat = _extract_one(row["audio_path"], row.get("begin_s"), row.get("end_s"))
            if feat.ndim != 2:
                feat = np.zeros((1, hidden_dim), dtype=np.float32)
        except Exception:
            feat = np.zeros((1, hidden_dim), dtype=np.float32)
        t = feat.shape[0]
        if frame_offset + t > total_est:
            total_est = int((frame_offset + t) * 1.05)
            arr.flush()
            del arr
            arr = np.memmap(out_npy, dtype="float32", mode="r+", shape=(total_est, hidden_dim))
        arr[frame_offset:frame_offset + t] = feat
        lengths_all.append(t)
        frame_offset += t

    arr.flush()
    del arr
    frame_total = frame_offset
    if frame_total < total_est:
        tmp = out_npy + ".trim"
        src = np.memmap(out_npy, dtype="float32", mode="r", shape=(total_est, hidden_dim))
        np.save(tmp, src[:frame_total])
        del src
        os.replace(tmp, out_npy)

    torch.save({"lengths": lengths_all, "hidden_dim": hidden_dim, "backend": "musicdiscovery"}, out_meta)
    print(f"Wrote {out_npy} shape=({frame_total}, {hidden_dim})")


# ---------------------------------------------------------------------------
# Sharded path — one .npy per audio (supports chunk_sec / pool_to_frames)
# ---------------------------------------------------------------------------

def _run_sharded(args, rows: List[dict], out_dir: str) -> None:
    from sae_components.musicgen_hooked import HookedMusicGen
    from sae_lens import SAE

    shard_dir = os.path.join(out_dir, args.split)
    os.makedirs(shard_dir, exist_ok=True)

    out_meta = os.path.join(out_dir, f"{args.out_prefix}_{args.split}_meta.pt")
    n_total = len(rows)

    if args.rank == 0 and args.skip_existing and os.path.isfile(out_meta):
        try:
            meta_obj = torch.load(out_meta, map_location="cpu")
            if (meta_obj.get("mode") == "sharded"
                    and isinstance(meta_obj.get("lengths"), list)
                    and len(meta_obj["lengths"]) == n_total):
                print(f"[skip-existing] {args.split}: complete sharded meta found")
                return
        except Exception:
            pass

    model = HookedMusicGen(model_name="facebook/musicgen-small", device=args.device)
    sae = SAE.load_from_pretrained(args.checkpoint_dir, device=args.device)
    hidden_dim = int(sae.cfg.d_sae)

    sizes = []
    for i, row in enumerate(rows):
        try:
            sizes.append((os.path.getsize(row["audio_path"]), i))
        except OSError:
            sizes.append((0, i))
    sorted_indices = [i for _, i in sorted(sizes)]
    my_indices = sorted_indices[args.rank::args.world_size]

    undone = [i for i in my_indices if not os.path.isfile(_shard_path(shard_dir, i))]
    print(f"[rank {args.rank}] {args.split}: {len(my_indices)} assigned, "
          f"{len(my_indices) - len(undone)} already done, {len(undone)} to extract")

    lengths_by_idx: Dict[int, int] = {}
    for i in my_indices:
        sp = _shard_path(shard_dir, i)
        if os.path.isfile(sp):
            try:
                arr = np.load(sp, mmap_mode="r")
                lengths_by_idx[i] = arr.shape[0]
            except Exception:
                lengths_by_idx[i] = 1

    if args.chunk_sec > 0:
        # Sequential per-clip chunked extraction (needed for long-form audio).
        for i in tqdm(undone, desc=f"rank{args.rank}/{args.split}"):
            row = rows[i]
            try:
                wav, _ = _load_audio_mono_32k(
                    row["audio_path"],
                    begin_s=row.get("begin_s"),
                    end_s=row.get("end_s"),
                    max_audio_sec=args.max_audio_sec,
                )
                feat = _extract_one_chunked(
                    model, sae, wav, args.device,
                    chunk_sec=args.chunk_sec,
                    pool_to_frames=args.pool_to_frames,
                )
                if args.max_seq_len > 0 and feat.shape[0] > args.max_seq_len:
                    feat = feat[: args.max_seq_len]
                if feat.ndim != 2 or feat.shape[1] != hidden_dim:
                    feat = np.zeros((1, hidden_dim), dtype=np.float32)
            except Exception:
                feat = np.zeros((1, hidden_dim), dtype=np.float32)
            np.save(_shard_path(shard_dir, i), feat)
            lengths_by_idx[i] = feat.shape[0]
    else:
        batch_size = max(1, args.batch_size)
        for batch_start in tqdm(range(0, len(undone), batch_size),
                                desc=f"rank{args.rank}/{args.split}", unit="batch"):
            batch_idx = undone[batch_start:batch_start + batch_size]
            batch_rows = [rows[i] for i in batch_idx]
            feats = _extract_batch(
                model, sae, batch_rows, args.device,
                args.max_seq_len, hidden_dim, args.max_audio_sec,
            )
            for i, feat in zip(batch_idx, feats):
                np.save(_shard_path(shard_dir, i), feat)
                lengths_by_idx[i] = feat.shape[0]

    partial_path = _partial_meta_path(out_dir, args.split, args.rank)
    torch.save({"lengths_by_idx": lengths_by_idx, "rank": args.rank}, partial_path)
    print(f"[rank {args.rank}] wrote partial meta {partial_path}")

    if args.rank == 0 and args.world_size > 1:
        print(f"[rank 0] waiting for {args.world_size - 1} other ranks...")
        deadline = time.time() + 7200
        while time.time() < deadline:
            missing = [r for r in range(1, args.world_size)
                       if not os.path.isfile(_partial_meta_path(out_dir, args.split, r))]
            if not missing:
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"Timed out waiting for ranks: {missing}")

    if args.rank == 0:
        _write_final_meta(out_dir, args.split, args.world_size, args.out_prefix, hidden_dim, n_total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="clean", choices=["clean"],
                        help="Only 'clean' is supported (noisy was removed).")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--out-prefix", default="sae_features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", default="sharded", choices=["sharded", "monolithic"])
    parser.add_argument("--max-seq-len", type=int, default=1500,
                        help="Cap tokens per file at extraction. 0=no cap.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Audio files per GPU forward pass (batched path, chunk_sec==0).")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--chunk-sec", type=float, default=0.0,
                        help="If > 0, process each audio in non-overlapping "
                             "windows of this many seconds (sequential per "
                             "clip).  Use for SongEval / long clips.")
    parser.add_argument("--pool-to-frames", type=int, default=0,
                        help="Uniformly mean-pool the concatenated SAE features "
                             "to this many frames along time (use with --chunk-sec).")
    parser.add_argument("--max-audio-sec", type=float, default=0.0,
                        help="Cap on audio duration in seconds (0 = unlimited).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = args.project_root
    splits_label = args.splits

    data_cfg = cfg.get("data", cfg)
    csv_path = None
    splits_cfg = data_cfg.get("splits", {})
    if splits_label in splits_cfg and args.split in splits_cfg[splits_label]:
        csv_path = splits_cfg[splits_label][args.split]
        if os.path.isfile(csv_path):
            print(f"[{args.split}] Using CSV from runtime config: {csv_path}")

    if csv_path is None:
        manifests_dir = os.path.join(project_root, data_cfg.get("manifests_dir", data/manifests"))
        dataset_name = data_cfg.get("dataset_name", "musiceval")
        csv_name = f"{dataset_name}_{splits_label}_{args.split}.csv"
        candidates = [
            os.path.join(manifests_dir, csv_name),
            os.path.join(project_root, "data", "full_splits", csv_name),
            os.path.join(project_root, "data", "musiceval_splits", csv_name),
        ]
        for c in candidates:
            if os.path.isfile(c):
                csv_path = c
                break

    if csv_path is None:
        raise FileNotFoundError(f"Cannot find CSV for {splits_label} {args.split}")

    df = pd.read_csv(csv_path)
    rows: List[dict] = []
    for _, r in df.iterrows():
        p = str(r["audio_path"])
        if not os.path.isabs(p):
            p = os.path.join(project_root, p)
        begin_s, end_s = get_row_window(r)
        rows.append({"audio_path": p, "begin_s": begin_s, "end_s": end_s})
    print(f"[{args.split}] {len(rows)} audio files from {csv_path}")
    print(f"[window] chunk_sec={args.chunk_sec} pool_to_frames={args.pool_to_frames} "
          f"max_audio_sec={args.max_audio_sec}")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        sae_cfg = cfg.get("sae", {})
        dataset_name = data_cfg.get("dataset_name", "musiceval")
        out_dir = os.path.join(
            project_root,
            sae_cfg.get("output_dir", outputs/full/features/sae"),
            dataset_name,
            splits_label,
        )
    os.makedirs(out_dir, exist_ok=True)

    md_path = os.path.join(project_root, "external", "musicdiscovery")
    if md_path not in sys.path:
        sys.path.insert(0, md_path)

    if args.mode == "sharded":
        _run_sharded(args, rows, out_dir)
    else:
        _run_monolithic(args, rows, out_dir)


if __name__ == "__main__":
    main()
