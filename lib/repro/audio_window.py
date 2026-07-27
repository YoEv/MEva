"""Audio loading, rater-aligned windowing, and uniform-pooling helpers used
by the feature extractors (entropy / SAE / per-token loss).

Why this file exists
--------------------
The three extractors used to each hard-code a ``MAX_AUDIO_SECONDS = 30``
prefix.  That was wrong for two reasons:

1. Music Arena and AIME ship a **rater-aligned window** per clip
   (``begin_s``/``end_s`` columns written by ``gen_full_splits.py``) --
   the CNN should see what the human judged, not a fixed 30-s prefix.
2. SongEval raters scored **full songs** (~2-6 min).  A 30-s prefix
   destroys the label-feature alignment.  We now process long songs in
   non-overlapping ``--chunk-sec`` windows and optionally uniformly
   mean-pool the concatenated features to a fixed length so the
   downstream CNN still sees a single fixed-shape tensor.

All three helpers are cheap (no model loads) and have no heavy deps.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR_OUT = 32000


def load_audio_mono(
    audio_path: str,
    *,
    begin_s: Optional[float] = None,
    end_s: Optional[float] = None,
    max_audio_sec: float = 0.0,
    sr_out: int = SR_OUT,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load ``audio_path`` as mono @ ``sr_out`` Hz, optionally cropped to
    ``[begin_s, end_s]`` and then capped to ``max_audio_sec`` seconds.

    Returns a ``[1, S]`` float32 tensor (optionally moved to ``device``).
    Short clips are returned as-is -- we deliberately do **not** zero-pad
    to avoid fabricating audio the rater never heard (the downstream
    pooling / fixed-length logic handles short features).
    """
    wav_np, sr = sf.read(audio_path)
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]
    else:
        wav_np = wav_np.T  # [C, S]
    wav = torch.from_numpy(wav_np).float()
    if sr != sr_out:
        wav = torchaudio.functional.resample(wav, sr, sr_out)
    wav = wav.mean(dim=0, keepdim=True)  # [1, S]

    if begin_s is not None and end_s is not None and end_s > begin_s:
        a = max(0, int(round(float(begin_s) * sr_out)))
        b = min(wav.size(-1), int(round(float(end_s) * sr_out)))
        if b > a:
            wav = wav[..., a:b]

    if max_audio_sec and max_audio_sec > 0.0:
        n_max = int(round(float(max_audio_sec) * sr_out))
        if wav.size(-1) > n_max:
            wav = wav[..., :n_max]

    if device is not None:
        wav = wav.to(device)
    return wav


def uniform_pool_time_last(x: np.ndarray, target_t: int) -> np.ndarray:
    """Uniform mean-pool ``x[..., T]`` along the last axis to ``target_t``
    frames.  Used for entropy / loss curves stored as ``[K, T]``.
    """
    if x.ndim == 1:
        x = x[None, :]
    T = x.shape[-1]
    if T == target_t:
        return x.astype(np.float32, copy=False)
    if T < target_t:
        out = np.full(x.shape[:-1] + (target_t,), np.nan, dtype=np.float32)
        out[..., :T] = x
        return out
    edges = np.linspace(0, T, target_t + 1).astype(np.int64)
    out = np.empty(x.shape[:-1] + (target_t,), dtype=np.float32)
    for i in range(target_t):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = a + 1
        out[..., i] = x[..., a:b].mean(axis=-1)
    return out


def uniform_pool_time_first(z: np.ndarray, target_t: int) -> np.ndarray:
    """Uniform mean-pool ``z[T, D]`` along the first axis to ``target_t``.

    Used for SAE activations stored as ``[T, D]``.
    """
    T = z.shape[0]
    if T == target_t:
        return z.astype(np.float32, copy=False)
    if T < target_t:
        out = np.zeros((target_t, z.shape[1]), dtype=np.float32)
        out[:T] = z
        return out
    edges = np.linspace(0, T, target_t + 1).astype(np.int64)
    out = np.empty((target_t, z.shape[1]), dtype=np.float32)
    for i in range(target_t):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = a + 1
        out[i] = z[a:b].mean(axis=0)
    return out


def get_row_window(row) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(begin_s, end_s)`` from a pandas row, or ``(None, None)``
    when the columns are missing or NaN.
    """
    b = row["begin_s"] if hasattr(row, "index") and "begin_s" in row.index else None
    e = row["end_s"] if hasattr(row, "index") and "end_s" in row.index else None
    if b is None or e is None:
        return None, None
    try:
        b = float(b); e = float(e)
    except (TypeError, ValueError):
        return None, None
    if np.isnan(b) or np.isnan(e):
        return None, None
    return b, e
