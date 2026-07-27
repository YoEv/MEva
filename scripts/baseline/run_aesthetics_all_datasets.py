"""
Run Audiobox Aesthetics baseline on all 4 datasets.

Datasets:
  musiceval  - WAV files from datasets/musiceval/audio/
  songeval   - MP3 files from raw_hf/SongEval/mp3/
  aime       - Audio bytes in raw_hf/AIME/data/*.parquet,
               trimmed to rater-aligned 10-s window (from pairwise survey)
  musicpref  - Audio bytes in raw_hf/MusicPref/data/*.parquet, full clips

Outputs (one CSV per dataset):
  outputs/aesthetics/<dataset>/aesthetics_scores.csv
  columns: audio_id, CE, CU, PC, PQ

Usage (in audiobox conda env):
  python run_aesthetics_all_datasets.py [--datasets musiceval songeval aime musicpref]
                                         [--batch-size 8] [--ckpt None]

Environment: conda audiobox
"""
import os
import sys
import io
import glob
import argparse
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Repo root (3 levels up from this file) ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PH7_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # 
REPO_ROOT = os.path.dirname(PH7_ROOT)                      # noiseloss/

AESTHETICS_SRC = os.path.join(REPO_ROOT, "external", "audiobox-aesthetics", "src")
sys.path.insert(0, AESTHETICS_SRC)

AXES = ["CE", "CU", "PC", "PQ"]

RAW_HF      = os.path.join(PH7_ROOT, "raw_hf")
DATASETS    = os.path.join(PH7_ROOT, "datasets")
MANIFESTS   = os.path.join(PH7_ROOT, "data", "manifests")
OUT_BASE    = os.path.join(PH7_ROOT, "outputs", "aesthetics")


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_predictor_on_files(predictor, file_items, batch_size):
    """
    file_items: list of {"path": absolute_path, "audio_id": str}
    Returns list of dicts {audio_id, CE, CU, PC, PQ}
    """
    rows_in  = [{"path": it["path"]} for it in file_items]
    results  = []
    for i in tqdm(range(0, len(rows_in), batch_size)):
        batch = rows_in[i : i + batch_size]
        out   = predictor.forward(batch)
        for j, row in enumerate(out):
            results.append({"audio_id": file_items[i + j]["audio_id"],
                             **{ax: row[ax] for ax in AXES}})
    return results


def bytes_to_wav_file(audio_bytes, tmpdir, fname):
    """Write raw audio bytes to a tmp WAV file; return path."""
    path = os.path.join(tmpdir, fname)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def trim_wav(src_path, begin_sec, end_sec, dst_path):
    """Trim a WAV/MP3 file to [begin_sec, end_sec] using soundfile + numpy."""
    import soundfile as sf
    data, sr = sf.read(src_path)
    start = int(begin_sec * sr)
    stop  = int(end_sec   * sr)
    sf.write(dst_path, data[start:stop], sr)


def parse_time(s):
    """'00:01:03' or '00:53' -> seconds (float)."""
    parts = s.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_musiceval(predictor, batch_size, out_dir):
    """Full WAV files from datasets/musiceval/audio/"""
    audio_dir = os.path.join(DATASETS, "musiceval", "audio")
    wav_files  = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    print(f"[MusicEval] {len(wav_files)} WAV files")

    items = [{"path": p, "audio_id": os.path.splitext(os.path.basename(p))[0]}
             for p in wav_files]
    results = run_predictor_on_files(predictor, items, batch_size)
    _save(results, out_dir, "musiceval")


def load_songeval(predictor, batch_size, out_dir):
    """MP3 files from raw_hf/SongEval/mp3/ — full clips.

    SongEval songs are 2-6 min each.  The AesPredictor chunks them into
    10-second windows internally, so a batch of 8 songs means ~200 chunks
    on the GPU at once (OOM).  We always force batch_size=1 here.
    """
    mp3_dir   = os.path.join(RAW_HF, "SongEval", "mp3")
    mp3_files = sorted(glob.glob(os.path.join(mp3_dir, "*.mp3")))
    batch_size = 1   # songs are 2-6 min; 10-s chunk batching done internally
    print(f"[SongEval] {len(mp3_files)} MP3 files (batch_size forced to 1)")

    items = [{"path": p, "audio_id": os.path.splitext(os.path.basename(p))[0]}
             for p in mp3_files]
    results = run_predictor_on_files(predictor, items, batch_size)
    _save(results, out_dir, "songeval")


def load_aime(predictor, batch_size, out_dir):
    """
    Audio bytes in raw_hf/AIME/data/*.parquet.
    Each track has ONE unique rater-aligned window from pairwise_survey.csv.
    Audio ID: integer track_id (e.g. 5081 for AIME2025_5081.wav).
    """
    parquet_dir  = os.path.join(RAW_HF, "AIME", "data")
    survey_path  = os.path.join(MANIFESTS, "aime_pairwise_survey.csv")

    # Build window map: parquet_id_str -> (begin_sec, end_sec)
    survey = pd.read_csv(survey_path)
    window_map = {}
    for _, row in survey.iterrows():
        for prefix in ("track_1", "track_2"):
            pid_str = str(row[f"{prefix}_id_str"]).zfill(5)
            beg = parse_time(str(row[f"{prefix}_begin"]))
            end = parse_time(str(row[f"{prefix}_end"]))
            window_map[pid_str] = (beg, end)
    print(f"[AIME] Window map: {len(window_map)} unique tracks")

    # Load all parquets, keeping only tracks in window_map
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"[AIME] Reading {len(parquet_files)} parquet files …")

    results = []
    with tempfile.TemporaryDirectory(prefix="aime_aesthetics_") as tmpdir:
        batch_items = []

        def flush_batch():
            nonlocal batch_items
            if not batch_items:
                return
            out = predictor.forward([{"path": it["wav_path"]} for it in batch_items])
            for j, row in enumerate(out):
                results.append({"audio_id": batch_items[j]["audio_id"],
                                 **{ax: row[ax] for ax in AXES}})
            batch_items = []

        for pfile in tqdm(parquet_files, desc="parquets"):
            df = pd.read_parquet(pfile)
            for _, r in df.iterrows():
                pid_str = r["id"]  # e.g. "05081"
                if pid_str not in window_map:
                    continue
                beg, end = window_map[pid_str]
                audio_id = int(pid_str)  # e.g. 5081

                # Write raw bytes to temp file, trim, keep trimmed
                raw_path  = os.path.join(tmpdir, f"{pid_str}_raw.wav")
                trim_path = os.path.join(tmpdir, f"{pid_str}.wav")
                with open(raw_path, "wb") as f:
                    f.write(r["audio"]["bytes"])
                trim_wav(raw_path, beg, end, trim_path)
                os.remove(raw_path)

                batch_items.append({"audio_id": audio_id, "wav_path": trim_path})
                if len(batch_items) >= batch_size:
                    flush_batch()
                    # Remove trimmed files to save disk
                    for it in batch_items:
                        if os.path.exists(it["wav_path"]):
                            os.remove(it["wav_path"])
                    batch_items = []

        flush_batch()

    print(f"[AIME] Scored {len(results)} tracks")
    _save(results, out_dir, "aime")


def load_musicpref(predictor, batch_size, out_dir):
    """
    Audio bytes in raw_hf/MusicPref/data/*.parquet.
    Full clips (no windowing).

    Audio ID (K) is derived from the parquet path field:
      path = '{pair_id}_{side}.wav'  →  K = pair_id * 2 + side

    This matches the naming convention used in the test set:
      'MusicPref2025_K.wav' has audio_id = K
    """
    parquet_dir   = os.path.join(RAW_HF, "MusicPref", "data")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"[MusicPref] Reading {len(parquet_files)} parquet files …")

    import re
    _path_re = re.compile(r"^(\d+)_([01])\.wav$")

    results = []
    with tempfile.TemporaryDirectory(prefix="musicpref_aesthetics_") as tmpdir:
        batch_items = []

        def flush_batch():
            nonlocal batch_items
            if not batch_items:
                return
            out = predictor.forward([{"path": it["wav_path"]} for it in batch_items])
            for j, row in enumerate(out):
                results.append({"audio_id": batch_items[j]["audio_id"],
                                 **{ax: row[ax] for ax in AXES}})
            for it in batch_items:
                if os.path.exists(it["wav_path"]):
                    os.remove(it["wav_path"])
            batch_items.clear()

        for pfile in tqdm(parquet_files, desc="parquets"):
            df = pd.read_parquet(pfile)
            for _, r in df.iterrows():
                raw_path = r["audio"]["path"]          # e.g. "1630_0.wav"
                m = _path_re.match(raw_path)
                if m is None:
                    print(f"  WARNING: unexpected path format: {raw_path!r} – skip")
                    continue
                pair_id = int(m.group(1))
                side    = int(m.group(2))
                audio_id = pair_id * 2 + side          # matches MusicPref2025_K.wav

                wav_path = os.path.join(tmpdir, f"mp_{audio_id}.wav")
                with open(wav_path, "wb") as f:
                    f.write(r["audio"]["bytes"])

                batch_items.append({"audio_id": audio_id, "wav_path": wav_path})
                if len(batch_items) >= batch_size:
                    flush_batch()

        flush_batch()

    print(f"[MusicPref] Scored {len(results)} audio files")
    _save(results, out_dir, "musicpref")


def _save(results, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "aesthetics_scores.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"  → Saved {len(df)} rows to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

DATASET_MAP = {
    "musiceval": load_musiceval,
    "songeval":  load_songeval,
    "aime":      load_aime,
    "musicpref": load_musicpref,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASET_MAP.keys()),
                        choices=list(DATASET_MAP.keys()),
                        help="Datasets to process (default: all 4)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Local checkpoint path; None = HF facebook/audiobox-aesthetics")
    parser.add_argument("--out-dir", type=str, default=OUT_BASE,
                        help=f"Root output dir (default: {OUT_BASE})")
    args = parser.parse_args()

    from audiobox_aesthetics.infer import AesPredictor
    predictor = AesPredictor(checkpoint_pth=args.ckpt, data_col="path")
    print(f"Loaded AesPredictor (ckpt={'HF' if args.ckpt is None else args.ckpt})")

    for ds in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds.upper()}")
        print(f"{'='*60}")
        out_dir = os.path.join(args.out_dir, ds)
        DATASET_MAP[ds](predictor, args.batch_size, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
