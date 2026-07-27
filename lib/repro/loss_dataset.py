from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

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



def resolve_entropy_npy_path(manifest_csv: str, raw: str) -> str:
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
        manifest_abs = os.path.abspath(path)
        for _, r in em.iterrows():
            ev = r["entropy_curve_path"]
            if pd.isna(ev):
                continue
            v = str(ev).strip()
            if v:
                ep = resolve_entropy_npy_path(manifest_abs, v)
                if "token_loss_path" in em.columns:
                    key = str(r["token_loss_path"])
                    by_token[key] = ep
                if "source_token_loss_path" in em.columns:
                    skey = str(r["source_token_loss_path"])
                    by_token[skey] = ep
                if "audio_path" in em.columns:
                    akey = str(r["audio_path"])
                    by_audio[akey] = ep
    return by_token, by_audio


class LossCurveDataset(Dataset):
    def __init__(
        self,
        index_file_path: str,
        max_len: int = 1500,
        entropy_manifest_csv: Optional[Union[str, List[str]]] = None,
        skip_if_entropy_missing: bool = False,
        entropy_fill_missing: bool = False,
        feature_root: Optional[str] = None,
    ):
        self.index_df = pd.read_csv(index_file_path)
        self.max_len = max_len
        if entropy_manifest_csv is None:
            em_paths = None
        elif isinstance(entropy_manifest_csv, str):
            em_paths = [entropy_manifest_csv]
        else:
            em_paths = list(entropy_manifest_csv)
        self.entropy_map, self.entropy_map_by_audio = _build_entropy_map(em_paths)
        self.use_entropy = (len(self.entropy_map) > 0) or (len(self.entropy_map_by_audio) > 0)
        self.output_channels = (NUM_CODEBOOKS * 2 + 1) if self.use_entropy else (NUM_CODEBOOKS + 1)
        self.skip_if_entropy_missing = skip_if_entropy_missing
        self.entropy_fill_missing = entropy_fill_missing
        self.feature_root = feature_root
        self._index_file_path = index_file_path

        # Build source->feature_root mapping for multi-dataset (all_5_datasets)
        # Map points to parent like /features/loss/music_arena/clean
        self._source_root_map: Dict[str, str] = {}
        if feature_root and os.path.isdir(feature_root):
            for src in os.listdir(feature_root):
                src_path = os.path.join(feature_root, src)
                if os.path.isdir(src_path):
                    self._source_root_map[src] = src_path

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        score = row["score"]
        csv_path = str(row["token_loss_path"])

        # Multi-dataset lookup: prepend source-specific feature root if available
        if self.feature_root and self._source_root_map:
            source = str(row.get("source", ""))
            if source in self._source_root_map:
                # token_loss_path format: {source}/clean/{split}/xxx_loss.csv
                # Need to join: feature_root/{source} -> .../features/loss/music_arena
                # Then add the /clean/{split}/xxx part
                prefix = self._source_root_map[source]
                # Extract the part after {source}/ from token_loss_path
                remaining = csv_path.split(source + "/", 1)[1] if source in csv_path else csv_path
                csv_path = prefix + "/" + remaining

        loss_curves = read_loss_codebook_curves(csv_path, num_codebooks=NUM_CODEBOOKS)  # [4, T]
        if loss_curves.shape[1] == 0:
            return None, None

        if not self.use_entropy:
            stacked, _ = pack_curves_with_mask(loss_curves, self.max_len)  # [5, L]
            return torch.tensor(stacked, dtype=torch.float32), torch.tensor([score], dtype=torch.float32)

        ep = self.entropy_map.get(csv_path)
        if ep is None:
            ap = str(row.get("audio_path", ""))
            if ap:
                ep = self.entropy_map_by_audio.get(ap)
        if ep is None or not os.path.isfile(ep):
            if self.skip_if_entropy_missing:
                return None, None
            entropy_curves = np.zeros((NUM_CODEBOOKS, 0), dtype=np.float32)
        else:
            entropy_curves = read_entropy_codebook_curves(ep, num_codebooks=NUM_CODEBOOKS)
        if entropy_curves.shape[1] == 0 and self.skip_if_entropy_missing:
            return None, None

        joint_len = int(min(loss_curves.shape[1], entropy_curves.shape[1], self.max_len))
        if joint_len == 0:
            if self.entropy_fill_missing:
                joint_len = int(min(loss_curves.shape[1], self.max_len))
                entropy_joint = np.zeros((NUM_CODEBOOKS, joint_len), dtype=np.float32)
            else:
                return None, None
        else:
            entropy_joint = entropy_curves[:, :joint_len].astype(np.float32, copy=False)

        loss_joint = loss_curves[:, :joint_len].astype(np.float32, copy=False)
        stacked = np.zeros((NUM_CODEBOOKS * 2 + 1, self.max_len), dtype=np.float32)  # [9, L]
        if joint_len > 0:
            stacked[:NUM_CODEBOOKS, :joint_len] = loss_joint
            stacked[NUM_CODEBOOKS : NUM_CODEBOOKS * 2, :joint_len] = entropy_joint
            stacked[-1, :joint_len] = 1.0
        return torch.tensor(stacked, dtype=torch.float32), torch.tensor([score], dtype=torch.float32)


def loss_curve_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)
