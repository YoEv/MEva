#!/usr/bin/env python3
"""Generate train/val/test splits for the pairwise datasets and SongEval.

Reads the per-dataset pairwise manifest CSVs produced by
``fit_pairwise_manifests.py`` (Elo scoring) plus SongEval's raw
``metadata.jsonl`` (Musicality-only mean across the 4 annotators) and
emits ``{train,val,test}.csv`` under ``data/full_splits/<name>/``.

Columns written to each split CSV
---------------------------------
Always present::

    score, audio_path, token_loss_path

Per-dataset extras (passed through so that the feature extractors can honour
the rater-aligned audio window):

- AIME          : ``begin_s``, ``end_s``  (10-second survey window per track)
- Music Arena   : ``begin_s``, ``end_s``  (0..listen_sec_used)

SongEval and MusicPref write only the base three columns; extractors use
the full audio file (SongEval = whole song; MusicPref = 30-second clip).

Noisy splits (``*_noisy.csv``) are **not** emitted.  The ``noisy`` family
has been removed from 
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd


DEFAULT_COLS: Sequence[str] = ("score", "audio_path", "token_loss_path")


def make_splits(
    df: pd.DataFrame,
    out_dir: str | os.PathLike,
    *,
    extra_cols: Sequence[str] = (),
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols: List[str] = list(DEFAULT_COLS) + [c for c in extra_cols if c not in DEFAULT_COLS]

    before = len(df)
    df = df[df["audio_path"].astype(str).map(os.path.exists)].reset_index(drop=True)
    print(f"  {out_dir.name}: {len(df)}/{before} rows with existing audio")
    if len(df) == 0:
        print(f"  {out_dir.name}: nothing to write (all audio_path missing on disk).")
        return

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_tr = int(train_frac * len(df))
    n_va = int(val_frac * len(df))
    splits = {
        "train": df.iloc[perm[:n_tr]],
        "val":   df.iloc[perm[n_tr:n_tr + n_va]],
        "test":  df.iloc[perm[n_tr + n_va:]],
    }
    for k, part in splits.items():
        part[cols].to_csv(out_dir / f"{k}.csv", index=False)
        print(f"    {k}: {len(part)}")


def _abs(p: str) -> str:
    return os.path.abspath(p)


def _aime_audio_path(track_id_str: str) -> str:
    """Resolve an AIME audio path, tolerating both HF id formats.

    ``hf_ingest_smoke.py`` now uses ``row["id"]`` verbatim as the filename
    stem (``AIME2025_<id>.wav``).  The AIME survey CSV stores ``track_id``
    as an integer, and ``aime_join_survey.py`` zero-pads it to 5 digits.
    Because HF may or may not zero-pad the ``id`` field, we try the zero-
    padded form first and fall back to the bare integer form.
    """
    base = "datasets/aime/audio"
    s = str(track_id_str)
    try:
        int_form = str(int(s))
    except ValueError:
        int_form = s
    zfill_form = s.zfill(5) if s.isdigit() else s
    for cand in (
        f"{base}/AIME2025_{zfill_form}.wav",
        f"{base}/AIME2025_{int_form}.wav",
    ):
        if os.path.isfile(cand):
            return _abs(cand)
    # Deterministic default for clear diagnostics (make_splits will drop it
    # because os.path.exists is False).
    return _abs(f"{base}/AIME2025_{zfill_form}.wav")


def _aime_token_loss_path(track_id_str: str) -> str:
    """Token-loss path stem; mirrors whichever audio filename actually exists."""
    p = _aime_audio_path(track_id_str)
    stem = Path(p).stem  # e.g. 'AIME2025_05331'
    return f"aime/{stem}"


def _build_aime() -> pd.DataFrame:
    # Read track_id as string so that zero-padded ids like "05331" are not
    # silently coerced to int 5331 by pandas (which previously produced
    # 'AIME2025_5331.wav' lookups that never matched HF-id-based filenames).
    mq = pd.read_csv(
        "data/manifests/pairwise_relu/aime_music_quality_1to5.csv",
        dtype={"track_id": str},
    )
    track_ids = mq["track_id"].astype(str)
    # Scoring is music_quality only; text_audio_alignment was removed.
    return pd.DataFrame({
        "score": mq["score_1to5"].astype(float).values,
        "audio_path": track_ids.map(_aime_audio_path),
        "token_loss_path": track_ids.map(_aime_token_loss_path),
        "begin_s": mq["begin_s"].astype(float).values,
        "end_s":   mq["end_s"].astype(float).values,
    })


def _build_musicpref() -> pd.DataFrame:
    mu = pd.read_csv(
        "data/manifests/pairwise_relu/musicpref_musicality_1to5.csv"
    )
    mu = mu[mu["audio_path"].astype(str) != ""].reset_index(drop=True)
    return pd.DataFrame({
        "score": mu["score_1to5"].astype(float).values,
        "audio_path": mu["audio_path"].map(_abs),
        "token_loss_path": mu["filename"].map(lambda x: f"musicpref/{x}"),
    })


def _build_musicarena() -> pd.DataFrame:
    ma = pd.read_csv(
        "data/manifests/pairwise_relu/musicarena_1to5.csv"
    )
    return pd.DataFrame({
        "score": ma["score_1to5"].astype(float).values,
        "audio_path": ma["audio_path"].map(_abs),
        "token_loss_path": ma["audio_path"].map(lambda p: f"music_arena/{Path(p).stem}"),
        "begin_s": np.zeros(len(ma), dtype=float),
        "end_s":   ma["listen_sec_used"].astype(float).values,
    })


def _build_songeval() -> pd.DataFrame:
    """SongEval score = mean of the 4 annotators' **Musicality** rating.

    SongEval rates full songs (105-361 s, median ~214 s), so the audio
    path points at the full WAV on disk; chunking/pooling happens inside
    the feature extractors (see ``extract_entropy_curves.py`` /
    ``extract_sae_features_musicdiscovery.py`` ``--chunk-sec`` /
    ``--pool-to-frames``).
    """
    rows = []
    with open("raw_hf/SongEval/metadata.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fname = Path(r["file_name"]).stem
            raters = r.get("annotation", [])
            mus = [float(a["Musicality"]) for a in raters if "Musicality" in a]
            if not mus:
                continue
            rows.append({
                "score": sum(mus) / len(mus),
                "audio_path": _abs(f"datasets/songeval/audio/{fname}.wav"),
                "token_loss_path": f"songeval/{fname}",
            })
    return pd.DataFrame(rows)


def main() -> None:
    # Sanity: all pairwise manifests must be present.
    for rel in (
        "data/manifests/pairwise_relu/aime_music_quality_1to5.csv",
        "data/manifests/pairwise_relu/musicpref_musicality_1to5.csv",
        "data/manifests/pairwise_relu/musicarena_1to5.csv",
    ):
        if not Path(rel).is_file():
            print(
                f"missing {rel}; run fit_pairwise_manifests.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

    aime = _build_aime()
    make_splits(aime, "data/full_splits/aime",
                extra_cols=("begin_s", "end_s"))

    mp = _build_musicpref()
    make_splits(mp, "data/full_splits/musicpref")

    ma = _build_musicarena()
    make_splits(ma, "data/full_splits/music_arena",
                extra_cols=("begin_s", "end_s"))

    se = _build_songeval()
    make_splits(se, "data/full_splits/songeval")

    print("Done.")


if __name__ == "__main__":
    main()
