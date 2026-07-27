"""统一内在信号 npz 的读写与校验(schema 见同目录 README.md)。

适配器只需实现: wav -> (loss[K,T], entropy[K,T], mask[T]) 并调用 save_signals()。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SignalMeta:
    model_tag: str
    model_hf_id: str
    codec: str
    frame_hz: float
    K: int
    clip_id: str
    benchmark: str
    window: dict = field(default_factory=dict)
    context_limit_tokens: int = 0
    extracted_at: str = ""

    def to_json(self) -> str:
        d = dict(self.__dict__)
        d["extracted_at"] = d["extracted_at"] or time.strftime("%Y-%m-%dT%H:%M:%S")
        return json.dumps(d, ensure_ascii=False)


def save_signals(path: str | Path, loss: np.ndarray, entropy: np.ndarray,
                 mask: np.ndarray, meta: SignalMeta) -> Path:
    path = Path(path)
    loss = np.asarray(loss, dtype=np.float32)
    entropy = np.asarray(entropy, dtype=np.float32)
    if loss.ndim == 1:
        loss = loss[None, :]
    if entropy.ndim == 1:
        entropy = entropy[None, :]
    mask = np.asarray(mask, dtype=np.uint8)
    _check_shapes(loss, entropy, mask, meta.K)
    bad = ~np.isfinite(loss).all(axis=0) | ~np.isfinite(entropy).all(axis=0)
    if bad.any():
        loss[:, bad] = 0.0
        entropy[:, bad] = 0.0
        mask = mask.copy()
        mask[bad] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, loss=loss, entropy=entropy, mask=mask,
                        meta=np.array(meta.to_json(), dtype=object))
    return path


def load_signals(path: str | Path):
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta"].item()))
        return z["loss"], z["entropy"], z["mask"], meta


def validate(path: str | Path) -> bool:
    """--skip-existing 用:文件存在且 schema 完整才算已完成。"""
    try:
        loss, entropy, mask, meta = load_signals(path)
        _check_shapes(loss, entropy, mask, int(meta["K"]))
        return bool(mask.any()) and np.isfinite(loss).all() and np.isfinite(entropy).all()
    except Exception:
        return False


def pool_to_frames(x: np.ndarray, frames: int = 1500) -> np.ndarray:
    """(K,T) -> (K,frames) 均匀 mean-pool,与论文/既有管线一致;T<frames 时线性插值上采样。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    K, T = x.shape
    if T == frames:
        return x
    if T < frames:
        idx = np.linspace(0, T - 1, frames)
        return np.stack([np.interp(idx, np.arange(T), row) for row in x]).astype(np.float32)
    edges = np.linspace(0, T, frames + 1).astype(int)
    return np.stack([
        np.array([row[a:b].mean() if b > a else row[min(a, T - 1)]
                  for a, b in zip(edges[:-1], edges[1:])], dtype=np.float32)
        for row in x
    ])


def _check_shapes(loss: np.ndarray, entropy: np.ndarray, mask: np.ndarray, K: int) -> None:
    if loss.shape != entropy.shape:
        raise ValueError(f"loss {loss.shape} != entropy {entropy.shape}")
    if loss.ndim != 2 or loss.shape[0] != K:
        raise ValueError(f"expected (K={K}, T), got {loss.shape}")
    if mask.shape != (loss.shape[1],):
        raise ValueError(f"mask {mask.shape} != (T={loss.shape[1]},)")
