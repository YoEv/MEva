from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lib.repro.curve_channels import NUM_CODEBOOKS, pack_curves_with_mask, read_entropy_codebook_curves


def _resolve_curve_path(manifest_csv: str, raw: str) -> str:
    v = str(raw).strip()
    if not v:
        return v
    if os.path.isfile(v):
        return os.path.abspath(v)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_csv))
    cand = os.path.normpath(os.path.join(manifest_dir, v))
    if os.path.isfile(cand):
        return cand
    return cand


def _build_entropy_map(manifest_paths: Optional[List[str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not manifest_paths:
        return {}, {}
    by_token: Dict[str, str] = {}
    by_audio: Dict[str, str] = {}
    for path in manifest_paths:
        em = pd.read_csv(path)
        if "entropy_curve_path" not in em.columns:
            raise ValueError(f"{path} must contain entropy_curve_path")
        for _, row in em.iterrows():
            curve_raw = row["entropy_curve_path"]
            if pd.isna(curve_raw):
                continue
            curve_path = _resolve_curve_path(path, str(curve_raw).strip())
            if not curve_path:
                continue
            if "token_loss_path" in em.columns:
                k = str(row["token_loss_path"])
                by_token[k] = curve_path
            # Backward compatibility for rewritten runtime splits that may point token_loss_path elsewhere.
            if "source_token_loss_path" in em.columns:
                sk = str(row["source_token_loss_path"])
                by_token[sk] = curve_path
            if "audio_path" in em.columns:
                ak = str(row["audio_path"])
                by_audio[ak] = curve_path
    return by_token, by_audio


class EntropyCurveDataset(Dataset):
    def __init__(self, index_file_path: str, max_len: int = 1500, entropy_manifest_csv: Optional[Union[str, List[str]]] = None):
        self.index_df = pd.read_csv(index_file_path)
        self.max_len = max_len
        if entropy_manifest_csv is None:
            paths = None
        elif isinstance(entropy_manifest_csv, str):
            paths = [entropy_manifest_csv]
        else:
            paths = list(entropy_manifest_csv)
        self.entropy_map, self.entropy_map_by_audio = _build_entropy_map(paths)
        self.output_channels = NUM_CODEBOOKS + 1
        if not self.entropy_map and not self.entropy_map_by_audio:
            raise ValueError("entropy manifest is required for entropy-only experiments")

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        score = float(row["score"])
        token_loss_path = str(row["token_loss_path"])
        audio_path = str(row.get("audio_path", ""))
        ep = self.entropy_map.get(token_loss_path)
        if ep is None and audio_path:
            ep = self.entropy_map_by_audio.get(audio_path)
        if ep is None or not os.path.isfile(ep):
            return None, None

        entropy_curves = read_entropy_codebook_curves(ep, num_codebooks=NUM_CODEBOOKS)  # [4, T]
        if entropy_curves.shape[1] == 0:
            return None, None
        x, _ = pack_curves_with_mask(entropy_curves, self.max_len)  # [5, L]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([score], dtype=torch.float32)


def entropy_curve_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)
