#!/usr/bin/env python3
"""
Compute mean per-token MusicGen cross-entropy loss (mean loss baseline)
for all annotated clips in 4 datasets, honouring the rater-aligned audio
window where applicable.

Rater window policy (mirrors extract_loss_curves.py / full_datasets.yaml):
  MusicEval  - full WAV (already short clips)
  SongEval   - full MP3 song, chunked into 30-s windows, results concatenated
  AIME       - 10-second rater window from pairwise survey (begin_s / end_s)
  MusicPref  - full parquet clip (5.1 / 10.2 / 29.9 s already pre-cut)

Output (one CSV per dataset):
  outputs/mean_loss/<dataset>/mean_loss_scores.csv
  columns: audio_id, mean_loss

Usage (torch21 conda env):
  python run_mean_loss_all_datasets.py [--datasets musiceval songeval aime musicpref]
                                        [--musicgen-model facebook/musicgen-small]
                                        [--chunk-sec 30] [--max-audio-sec 360]
                                        [--skip-existing]

Environment: conda torch21
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.nn import functional as F
from tqdm import tqdm
from transformers import MusicgenForConditionalGeneration
from transformers.models.musicgen.modeling_musicgen import (
    MusicgenSinusoidalPositionalEmbedding,
)

# ── Fix for older transformers / newer torch ─────────────────────────────────
if not hasattr(MusicgenSinusoidalPositionalEmbedding, "_offset_patched"):
    _orig_init = MusicgenSinusoidalPositionalEmbedding.__init__
    def _patched_init(self, num_positions, embedding_dim):
        self.offset = 2
        _orig_init(self, num_positions, embedding_dim)
    MusicgenSinusoidalPositionalEmbedding.__init__ = _patched_init
    MusicgenSinusoidalPositionalEmbedding._offset_patched = True

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PH7_ROOT   = SCRIPT_DIR.parents[1]          # 
REPO_ROOT  = PH7_ROOT.parent                # noiseloss/

RAW_HF    = PH7_ROOT / "raw_hf"
MANIFESTS = PH7_ROOT / "data" / "manifests"
SPLITS    = PH7_ROOT / "data" / "splits"
OUT_BASE  = PH7_ROOT / "outputs" / "mean_loss"

SR_OUT     = 32_000
NUM_CODEBOOKS = 4


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_bytes_to_tensor(audio_bytes: bytes,
                           begin_s: Optional[float] = None,
                           end_s:   Optional[float] = None,
                           chunk_sec: float = 0.0,
                           max_audio_sec: float = 0.0,
                           device: torch.device = torch.device("cpu"),
                           ) -> torch.Tensor:
    """Load raw audio bytes → mono float32 [1, S] @ SR_OUT, with window."""
    data, sr = sf.read(io.BytesIO(audio_bytes))
    return _tensor_from_array(data, sr, begin_s, end_s, max_audio_sec, device)


def _load_file_to_tensor(path: str,
                          begin_s: Optional[float] = None,
                          end_s:   Optional[float] = None,
                          max_audio_sec: float = 0.0,
                          device: torch.device = torch.device("cpu"),
                          ) -> torch.Tensor:
    """Load audio file → mono float32 [1, S] @ SR_OUT, with window."""
    data, sr = sf.read(path)
    return _tensor_from_array(data, sr, begin_s, end_s, max_audio_sec, device)


def _tensor_from_array(data: np.ndarray, sr: int,
                        begin_s: Optional[float],
                        end_s: Optional[float],
                        max_audio_sec: float,
                        device: torch.device) -> torch.Tensor:
    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T           # [C, S]
    wav = torch.from_numpy(data).float()
    if sr != SR_OUT:
        wav = torchaudio.functional.resample(wav, sr, SR_OUT)
    wav = wav.mean(dim=0, keepdim=True)   # [1, S]
    if begin_s is not None and end_s is not None and end_s > begin_s:
        a = max(0, int(round(begin_s * SR_OUT)))
        b = min(wav.size(-1), int(round(end_s * SR_OUT)))
        if b > a:
            wav = wav[..., a:b]
    if max_audio_sec > 0:
        n_max = int(round(max_audio_sec * SR_OUT))
        wav = wav[..., :n_max]
    return wav.to(device)


# ── MusicGen loss ─────────────────────────────────────────────────────────────

def _loss_one_chunk(model: MusicgenForConditionalGeneration,
                    wav_chunk: torch.Tensor) -> Optional[float]:
    """Mean per-token CE loss for a single chunk. Returns None on failure."""
    with torch.no_grad():
        enc = model.audio_encoder.encode(wav_chunk.unsqueeze(0))
        codes = enc.audio_codes.long()
        if codes.ndim != 4:
            return None
        bsz, channels, codebooks, t_all = codes.shape
        if bsz != 1 or codebooks < NUM_CODEBOOKS or t_all <= 1:
            return None
        inp = codes[:, :, :NUM_CODEBOOKS, :-1].contiguous().view(
            bsz, channels * NUM_CODEBOOKS, t_all - 1)
        out = model.decoder(inp, output_hidden_states=False)
        logits = out.logits
        if logits.dim() == 4:
            logits = logits[0]
        logits = logits.view(
            channels, NUM_CODEBOOKS, logits.shape[1], logits.shape[2]
        )[:, :NUM_CODEBOOKS, :, :]
        target = codes[0, :, :NUM_CODEBOOKS, 1:]
        t = min(int(logits.shape[2]), int(target.shape[2]))
        if t <= 0:
            return None
        losses = []
        for ch in range(channels):
            lg = logits[ch, :, :t, :].reshape(NUM_CODEBOOKS * t, logits.shape[-1])
            tg = target[ch, :, :t].reshape(NUM_CODEBOOKS * t)
            ce = F.cross_entropy(lg, tg, reduction="mean")
            losses.append(ce.item())
        return float(np.mean(losses))


def mean_loss_from_wav(model: MusicgenForConditionalGeneration,
                       wav: torch.Tensor,
                       chunk_sec: float = 0.0) -> Optional[float]:
    """Compute mean per-token loss over full wav [1, S], optionally chunked."""
    total = wav.size(-1)
    chunk_samples = int(round(chunk_sec * SR_OUT)) if chunk_sec > 0 else total
    chunk_samples = max(chunk_samples, 1)
    n_chunks = max(1, (total + chunk_samples - 1) // chunk_samples)
    vals, weights = [], []
    for c in range(n_chunks):
        a = c * chunk_samples
        b = min(a + chunk_samples, total)
        if b - a < int(0.1 * SR_OUT):   # skip tiny tail chunks
            continue
        v = _loss_one_chunk(model, wav[..., a:b])
        if v is not None:
            vals.append(v)
            weights.append(b - a)
    if not vals:
        return None
    # weighted average by chunk length
    w = np.array(weights, dtype=np.float64)
    return float(np.average(vals, weights=w))


# ── Ground-truth + audio iterators ───────────────────────────────────────────

def parse_time(s: str) -> float:
    parts = [float(x) for x in str(s).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def iter_musiceval(device):
    """Yield (audio_id, wav_tensor, gt_score) for all MusicEval clips."""
    dfs = []
    for split in ("train", "val", "test"):
        p = SPLITS / "musiceval" / f"{split}.csv"
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["audio_path"])
    for _, row in df.iterrows():
        audio_path = str(row["audio_path"])
        if not os.path.isfile(audio_path):
            print(f"  [WARN] missing: {audio_path}")
            continue
        audio_id = Path(audio_path).stem
        wav = _load_file_to_tensor(audio_path, device=device)
        yield audio_id, wav, float(row["score"])


def iter_songeval(device):
    """Yield (audio_id, wav_tensor, gt_score) for all SongEval songs."""
    meta_path = RAW_HF / "SongEval" / "metadata.jsonl"
    with open(meta_path) as f:
        entries = [json.loads(l) for l in f]
    for obj in entries:
        fname = Path(obj["file_name"]).stem
        mp3_path = RAW_HF / "SongEval" / "mp3" / f"{fname}.mp3"
        if not mp3_path.is_file():
            print(f"  [WARN] missing: {mp3_path}")
            continue
        mus = [float(a["Musicality"]) for a in obj.get("annotation", [])
               if "Musicality" in a]
        if not mus:
            continue
        gt = float(np.mean(mus))
        # Load full song; chunking handled in mean_loss_from_wav
        wav = _load_file_to_tensor(str(mp3_path), max_audio_sec=360.0, device=device)
        yield fname, wav, gt


def iter_aime(device):
    """Yield (audio_id, wav_tensor, gt_score) for all 1300 AIME tracks.

    Audio from raw_hf parquets, trimmed to 10-second rater window.
    GT = Music Quality win-rate from aime_track_winrate_by_question_type.csv.
    """
    survey = pd.read_csv(MANIFESTS / "aime_pairwise_survey.csv")
    wr     = pd.read_csv(MANIFESTS / "aime_track_winrate_by_question_type.csv")
    mq     = wr[wr["question_type"] == "Music Quality"].set_index("id")["win_rate"]

    window_map: dict[str, Tuple[float, float]] = {}
    for _, row in survey.iterrows():
        for prefix in ("track_1", "track_2"):
            pid = str(row[f"{prefix}_id_str"]).zfill(5)
            beg = parse_time(str(row[f"{prefix}_begin"]))
            end = parse_time(str(row[f"{prefix}_end"]))
            window_map[pid] = (beg, end)

    parquet_files = sorted(glob.glob(str(RAW_HF / "AIME" / "data" / "*.parquet")))
    print(f"  [AIME] {len(parquet_files)} parquet files, {len(window_map)} tracked windows")

    for pfile in tqdm(parquet_files, desc="AIME parquets"):
        df = pd.read_parquet(pfile)
        for _, r in df.iterrows():
            pid_str = str(r["id"])                       # e.g. "04501"
            if pid_str not in window_map:
                continue
            track_id_int = int(pid_str)
            if track_id_int not in mq.index:
                continue
            gt = float(mq.loc[track_id_int])
            beg, end = window_map[pid_str]
            wav = _load_bytes_to_tensor(r["audio"]["bytes"],
                                         begin_s=beg, end_s=end,
                                         max_audio_sec=10.0, device=device)
            yield track_id_int, wav, gt


def iter_musicpref(device):
    """Yield (audio_id, wav_tensor, gt_score) for all MusicPref clips.

    Audio from raw_hf parquets (already short clips: 5.1 / 10.2 / 29.9 s).
    GT = musicality win-rate computed from human_preference.csv.
    audio_id K = pair_id * 2 + side  (path '{pair_id}_{side}.wav')
    """
    import re
    _re = re.compile(r"^(\d+)_([01])\.wav$")

    pref = pd.read_csv(RAW_HF / "MusicPref" / "human_preference.csv")
    # Build win-rate per global audio_id
    records: dict[int, dict] = {}
    for pair_id, row in enumerate(pref.itertuples(index=False)):
        for side, result_key in ((0, "a"), (1, "b")):
            aid = pair_id * 2 + side
            records.setdefault(aid, {"wins": 0, "n": 0})
        records[pair_id * 2]["n"] += 1
        records[pair_id * 2 + 1]["n"] += 1
        result = row.musicality
        if result == "a_wins":
            records[pair_id * 2]["wins"] += 1
        elif result == "b_wins":
            records[pair_id * 2 + 1]["wins"] += 1
        else:
            records[pair_id * 2]["wins"] += 0.5
            records[pair_id * 2 + 1]["wins"] += 0.5
    win_rate = {k: v["wins"] / v["n"] for k, v in records.items() if v["n"] > 0}

    parquet_files = sorted(
        glob.glob(str(RAW_HF / "MusicPref" / "data" / "*.parquet")))
    print(f"  [MusicPref] {len(parquet_files)} parquet files")

    for pfile in tqdm(parquet_files, desc="MusicPref parquets"):
        df = pd.read_parquet(pfile)
        for _, r in df.iterrows():
            raw_path = r["audio"]["path"]   # e.g. "1630_0.wav"
            m = _re.match(raw_path)
            if m is None:
                continue
            aid = int(m.group(1)) * 2 + int(m.group(2))
            if aid not in win_rate:
                continue
            wav = _load_bytes_to_tensor(r["audio"]["bytes"],
                                         max_audio_sec=30.0, device=device)
            yield aid, wav, win_rate[aid]


# ── Main ─────────────────────────────────────────────────────────────────────

DATASET_MAP = {
    "musiceval": (iter_musiceval, {}),
    "songeval":  (iter_songeval,  {"chunk_sec": 30.0}),
    "aime":      (iter_aime,      {}),
    "musicpref": (iter_musicpref, {}),
}


def _save(results, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = out_dir / "mean_loss_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"  → Saved {len(df)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASET_MAP.keys()),
                        choices=list(DATASET_MAP.keys()))
    parser.add_argument("--musicgen-model", type=str,
                        default="facebook/musicgen-small")
    parser.add_argument("--chunk-sec", type=float, default=30.0,
                        help="Chunk size for long-form audio (SongEval)")
    parser.add_argument("--out-dir", type=str, default=str(OUT_BASE))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing",
                        action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading MusicGen model: {args.musicgen_model}")
    model = MusicgenForConditionalGeneration.from_pretrained(
        args.musicgen_model).to(device)
    model.eval()
    print("Model loaded.")

    out_root = Path(args.out_dir)

    for ds in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*60}")
        out_dir = out_root / ds
        out_path = out_dir / "mean_loss_scores.csv"

        # Load existing results for resume support
        done_ids: set = set()
        existing_rows = []
        if args.skip_existing and out_path.is_file():
            existing_df = pd.read_csv(out_path)
            existing_rows = existing_df.to_dict("records")
            done_ids = set(existing_df["audio_id"].astype(str).tolist())
            print(f"  Resuming: {len(done_ids)} already done")

        iterator_fn, extra_kwargs = DATASET_MAP[ds]
        chunk_sec = extra_kwargs.get("chunk_sec",
                                     args.chunk_sec if ds == "songeval" else 0.0)

        results = list(existing_rows)
        skipped = 0

        for audio_id, wav, gt_score in iterator_fn(device):
            if str(audio_id) in done_ids:
                skipped += 1
                continue
            ml = mean_loss_from_wav(model, wav, chunk_sec=chunk_sec)
            if ml is None:
                print(f"  [WARN] could not compute loss for {audio_id}")
                continue
            results.append({"audio_id": audio_id, "gt_score": gt_score,
                             "mean_loss": ml})
            done_ids.add(str(audio_id))
            # Save checkpoint every 100 new clips
            if (len(results) - len(existing_rows) - skipped) % 100 == 0 and len(results) > 0:
                _save(results, out_dir, ds)

        _save(results, out_dir, ds)
        print(f"  Skipped (already done): {skipped}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
