from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

NUM_CODEBOOKS = 4


def _to_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for col in df.columns:
        cols.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32))
    if not cols:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.stack(cols, axis=1)
    valid_rows = np.isfinite(arr).any(axis=1)
    if valid_rows.any():
        arr = arr[valid_rows]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return arr


def _normalize_channels(arr_tc: np.ndarray, num_codebooks: int = NUM_CODEBOOKS) -> np.ndarray:
    if arr_tc.ndim == 1:
        arr_tc = arr_tc[:, None]
    if arr_tc.shape[1] >= num_codebooks:
        out_tc = arr_tc[:, :num_codebooks]
    elif arr_tc.shape[1] == 1:
        out_tc = np.repeat(arr_tc, num_codebooks, axis=1)
    else:
        pad = np.zeros((arr_tc.shape[0], num_codebooks - arr_tc.shape[1]), dtype=np.float32)
        out_tc = np.concatenate([arr_tc, pad], axis=1)
    return out_tc.T.astype(np.float32, copy=False)  # [C, T]


def read_loss_codebook_curves(csv_path: str, num_codebooks: int = NUM_CODEBOOKS) -> np.ndarray:
    df = pd.read_csv(csv_path)
    feat_df = df.iloc[:, 1:] if df.shape[1] > 1 else df
    arr_tc = _to_numeric_matrix(feat_df)
    if arr_tc.size == 0:
        return np.zeros((num_codebooks, 0), dtype=np.float32)
    return _normalize_channels(arr_tc, num_codebooks=num_codebooks)


def read_entropy_codebook_curves(npy_path: str, num_codebooks: int = NUM_CODEBOOKS) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim == 1:
        arr_tc = arr[:, None].astype(np.float32, copy=False)
    elif arr.ndim == 2:
        # Support both [T, K] and [K, T].
        if arr.shape[0] <= 8 and arr.shape[1] > 8:
            arr_tc = arr.T.astype(np.float32, copy=False)
        else:
            arr_tc = arr.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported entropy shape: {arr.shape}")

    finite_any = np.isfinite(arr_tc).any(axis=1)
    if finite_any.any():
        first_bad = np.where(~finite_any)[0]
        if first_bad.size > 0:
            arr_tc = arr_tc[: first_bad[0]]
    arr_tc = np.nan_to_num(arr_tc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if arr_tc.size == 0:
        return np.zeros((num_codebooks, 0), dtype=np.float32)
    return _normalize_channels(arr_tc, num_codebooks=num_codebooks)


def pack_curves_with_mask(curves_ct: np.ndarray, max_len: int) -> Tuple[np.ndarray, int]:
    c, t = int(curves_ct.shape[0]), int(curves_ct.shape[1])
    valid_len = min(t, max_len)
    out = np.zeros((c + 1, max_len), dtype=np.float32)
    if valid_len > 0:
        out[:c, :valid_len] = curves_ct[:, :valid_len]
        out[c, :valid_len] = 1.0
    return out, valid_len
