import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lib.repro.curve_channels import (
    NUM_CODEBOOKS,
    pack_curves_with_mask,
    read_entropy_codebook_curves,
    read_loss_codebook_curves,
)


def _build_entropy_map(manifest_paths: Optional[List[str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not manifest_paths:
        return {}, {}
    by_token: Dict[str, str] = {}
    by_audio: Dict[str, str] = {}
    for p in manifest_paths:
        df = pd.read_csv(p)
        if "entropy_curve_path" not in df.columns:
            raise ValueError(f"{p} must contain entropy_curve_path")
        for _, row in df.iterrows():
            ep = row["entropy_curve_path"]
            if pd.isna(ep):
                continue
            epv = str(ep)
            if "token_loss_path" in df.columns:
                by_token[str(row["token_loss_path"])] = epv
            if "source_token_loss_path" in df.columns:
                by_token[str(row["source_token_loss_path"])] = epv
            if "audio_path" in df.columns:
                by_audio[str(row["audio_path"])] = epv
    return by_token, by_audio


class HybridPrecomputedDataset(Dataset):
    """Self-contained dataset for hybrid (curve + SAE) training.

    All data is resolved via ``split_csv_path`` + ``sae_feature_dir`` + ``token_loss_root``
    (all relative paths are joined against ``data_root``). This class does not reference
    any legacy exp10 / exp11 directory.
    """

    def __init__(
        self,
        split,
        data_root,
        *,
        split_csv_path,
        sae_feature_dir,
        token_loss_root,
        seq_len=1500,
        sae_dim=16384,
        sae_variant_suffix="",
        curve_mode="loss",
        entropy_manifest_csv=None,
    ):
        self.seq_len = seq_len
        self.sae_dim = sae_dim
        self.token_loss_root = token_loss_root
        if self.token_loss_root and (not os.path.isabs(self.token_loss_root)):
            self.token_loss_root = os.path.join(data_root, self.token_loss_root)
        self.curve_mode = curve_mode
        if self.curve_mode not in {"loss", "entropy", "loss_entropy", "none"}:
            raise ValueError("curve_mode must be one of: loss, entropy, loss_entropy, none")
        self.curve_channels = {
            "loss": NUM_CODEBOOKS + 1,
            "entropy": NUM_CODEBOOKS + 1,
            "loss_entropy": NUM_CODEBOOKS * 2 + 1,
            "none": 0,
        }[self.curve_mode]
        if entropy_manifest_csv is None:
            entropy_paths = None
        elif isinstance(entropy_manifest_csv, str):
            entropy_paths = [entropy_manifest_csv]
        else:
            entropy_paths = list(entropy_manifest_csv)
        self.entropy_map, self.entropy_map_by_audio = _build_entropy_map(entropy_paths)
        if self.curve_mode in {"entropy", "loss_entropy"} and (not self.entropy_map and not self.entropy_map_by_audio):
            raise ValueError("entropy/loss_entropy mode requires entropy manifests")
        if not split_csv_path:
            raise ValueError("split_csv_path is required")
        csv_path = split_csv_path if os.path.isabs(split_csv_path) else os.path.join(data_root, split_csv_path)
        self.df = pd.read_csv(csv_path)

        if not sae_feature_dir:
            raise ValueError("sae_feature_dir is required")
        base_feat_dir = sae_feature_dir if os.path.isabs(sae_feature_dir) else os.path.join(data_root, sae_feature_dir)
        meta_path = os.path.join(base_feat_dir, f"sae_features_{split}{sae_variant_suffix}_meta.pt")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Need {meta_path}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        lengths = [int(x) for x in meta["lengths"]]
        meta_hidden = int(meta.get("hidden_dim", self.sae_dim))

        self._sae_mode = meta.get("mode", "monolithic")
        if self._sae_mode == "sharded":
            shard_dir = str(meta.get("shard_dir", os.path.join(base_feat_dir, split)))
            self._shard_dir = shard_dir
            self.sae_dim = meta_hidden
            self._sae_memmap = None
            self._sae_cum = None
        else:
            npy_path = os.path.join(base_feat_dir, f"sae_features_{split}{sae_variant_suffix}.npy")
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"Need {npy_path}")
            total_frames = int(sum(lengths))
            self._sae_memmap = self._open_sae_array(npy_path, total_frames=total_frames, hidden_dim=meta_hidden)
            self.sae_dim = int(self._sae_memmap.shape[1])
            self._sae_cum = np.concatenate([[0], np.cumsum(lengths)])

        if len(self.df) != len(lengths):
            raise ValueError("CSV rows vs SAE lengths mismatch")

    @staticmethod
    def _open_sae_array(npy_path: str, total_frames: int, hidden_dim: int):
        file_bytes = os.path.getsize(npy_path)
        raw_expected = int(total_frames) * int(hidden_dim) * np.dtype(np.float32).itemsize
        if file_bytes == raw_expected:
            # Current musicdiscovery extractor writes raw float32 memmap bytes.
            return np.memmap(npy_path, dtype="float32", mode="r", shape=(total_frames, hidden_dim))
        try:
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"SAE array must be 2D, got {arr.shape}")
            if int(arr.shape[0]) != int(total_frames):
                raise ValueError(f"SAE frame mismatch: meta={total_frames}, file={arr.shape[0]}")
            return arr
        except Exception as e:
            raise ValueError(
                f"Cannot read SAE features from {npy_path}. "
                f"file_bytes={file_bytes}, expected_raw_bytes={raw_expected}, "
                f"total_frames={total_frames}, hidden_dim={hidden_dim}"
            ) from e

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        score = float(row["score"])
        token_loss_path = str(row["token_loss_path"])
        audio_path = str(row.get("audio_path", ""))
        loss_path = token_loss_path
        if not os.path.isabs(loss_path):
            loss_path = os.path.normpath(os.path.join(self.token_loss_root, loss_path))

        if self.curve_mode in {"loss", "loss_entropy", "entropy"}:
            try:
                loss_curves = read_loss_codebook_curves(loss_path, num_codebooks=NUM_CODEBOOKS)  # [4, T]
            except Exception:
                loss_curves = np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
        else:
            loss_curves = np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)

        entropy_curve = None
        if self.curve_mode in {"entropy", "loss_entropy"}:
            ep = self.entropy_map.get(token_loss_path)
            if ep is None and audio_path:
                ep = self.entropy_map_by_audio.get(audio_path)
            if ep and os.path.isfile(ep):
                entropy_curve = read_entropy_codebook_curves(ep, num_codebooks=NUM_CODEBOOKS)
            else:
                entropy_curve = np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)

        if self.curve_mode == "loss":
            stacked_curve, curve_valid_len = pack_curves_with_mask(loss_curves, self.seq_len)  # [5, L]
        elif self.curve_mode == "entropy":
            entropy_curves = entropy_curve if entropy_curve is not None else np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
            stacked_curve, curve_valid_len = pack_curves_with_mask(entropy_curves, self.seq_len)  # [5, L]
        elif self.curve_mode == "loss_entropy":
            entropy_curves = entropy_curve if entropy_curve is not None else np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
            joint = min(int(loss_curves.shape[1]), int(entropy_curves.shape[1]), self.seq_len)
            curve_valid_len = joint
            if joint <= 0:
                stacked_curve = np.zeros((NUM_CODEBOOKS * 2 + 1, self.seq_len), dtype=np.float32)
            else:
                stacked_curve = np.zeros((NUM_CODEBOOKS * 2 + 1, self.seq_len), dtype=np.float32)  # [9, L]
                stacked_curve[:NUM_CODEBOOKS, :joint] = loss_curves[:, :joint]
                stacked_curve[NUM_CODEBOOKS : NUM_CODEBOOKS * 2, :joint] = entropy_curves[:, :joint]
                stacked_curve[-1, :joint] = 1.0
        else:
            stacked_curve = np.zeros((0, self.seq_len), dtype=np.float32)
            curve_valid_len = 0

        if self._sae_mode == "sharded":
            shard_file = os.path.join(self._shard_dir, f"{idx:06d}.npy")
            try:
                feat = np.load(shard_file).astype(np.float32)
            except Exception:
                feat = np.zeros((1, self.sae_dim), dtype=np.float32)
        else:
            start, end = int(self._sae_cum[idx]), int(self._sae_cum[idx + 1])
            feat = np.array(self._sae_memmap[start:end], dtype=np.float32)
        sae_valid_len = min(int(feat.shape[0]), self.seq_len)
        if feat.shape[0] >= self.seq_len:
            feat = feat[: self.seq_len]
        else:
            feat = np.pad(feat, ((0, self.seq_len - feat.shape[0]), (0, 0)), constant_values=0)
        return {
            "loss_curve": torch.tensor(stacked_curve, dtype=torch.float32),
            "sae_features": torch.tensor(feat, dtype=torch.float32),
            "sae_len": torch.tensor(sae_valid_len, dtype=torch.long),
            "curve_len": torch.tensor(curve_valid_len, dtype=torch.long),
            "score": torch.tensor([score], dtype=torch.float32),
        }
