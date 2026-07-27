#!/usr/bin/env python3
"""HF 数据 smoke ingest：导出 wav + 追加写入 master_index.csv（列：source, path, score）。"""
from __future__ import annotations

import argparse
import io
import os
import re

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def pick_score(row, keys):
    candidates = [
        "score",
        "mos",
        "label",
        "preference",
        "rating",
        "quality",
        "overall",
        "human_score",
        "avg_score",
        "MOS",
    ]
    for c in candidates:
        if c in keys and row[c] is not None:
            try:
                return float(row[c])
            except (TypeError, ValueError):
                pass
    for _k, v in row.items():
        if isinstance(v, dict):
            for c in candidates:
                if c in v:
                    try:
                        return float(v[c])
                    except (TypeError, ValueError):
                        pass
    return float("nan")


def audio_array_and_sr(row, keys):
    for k in ("audio", "speech", "array", "sound"):
        if k not in row:
            continue
        v = row[k]
        if isinstance(v, dict) and "array" in v:
            arr = v["array"]
            sr = int(v.get("sampling_rate", 32000))
            return np.asarray(arr, dtype=np.float32), sr
        if isinstance(v, dict) and "bytes" in v and v["bytes"]:
            arr, sr = sf.read(io.BytesIO(v["bytes"]), always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
        if isinstance(v, dict) and "path" in v and v["path"] and os.path.exists(v["path"]):
            arr, sr = sf.read(v["path"], always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-config", default=None, help="Optional dataset config name, e.g. music-arena monthly config.")
    p.add_argument("--source-tag", required=True, help="写入 master 的 source；勿用 MusicPrefs（legacy 特殊子目录）")
    p.add_argument("--split", default="train")
    p.add_argument("--out-audio-dir", required=True)
    p.add_argument("--master-csv", required=True)
    p.add_argument("--max-samples", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.out_audio_dir, exist_ok=True)
    from datasets import Audio, load_dataset

    split = args.split
    if args.max_samples and args.max_samples > 0:
        split = f"{split}[:{args.max_samples}]"
    if args.repo_config:
        ds = load_dataset(args.repo, args.repo_config, split=split)
    else:
        ds = load_dataset(args.repo, split=split)
    for col, feat in ds.features.items():
        if feat.__class__.__name__ == "Audio":
            ds = ds.cast_column(col, Audio(decode=False))
    keys = ds.column_names
    rows_out = []

    for i in tqdm(range(len(ds)), desc=args.source_tag):
        row = ds[i]
        sc = pick_score(row, keys)
        arr, sr = audio_array_and_sr(row, keys)
        if arr is None:
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        # Prefer a stable id field from the HF row (e.g. disco-eth/AIME's
        # ``id`` column that the survey joins on).  Falling back to the
        # enumeration index silently loses this mapping and was the root cause
        # of the AIME track<->audio misalignment on the server.
        id_field = None
        for key in ("id", "item_id", "track_id", "track_id_str"):
            if key in keys:
                v = row.get(key)
                if v is not None and str(v) != "":
                    id_field = str(v)
                    break
        stem_src = id_field if id_field is not None else str(i)
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", f"{args.source_tag}_{stem_src}")
        wav_path = os.path.join(args.out_audio_dir, f"{stem}.wav")
        sf.write(wav_path, arr, sr if sr else 32000)
        rows_out.append({
            "source": args.source_tag,
            "path": os.path.abspath(wav_path),
            "score": sc,
            "id": id_field if id_field is not None else "",
            "row_index": i,
        })

    df = pd.DataFrame(rows_out)
    if df.empty:
        print("No rows exported; check dataset schema.")
        return
    if os.path.isfile(args.master_csv):
        old = pd.read_csv(args.master_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.master_csv, index=False)
    print(f"Wrote {len(rows_out)} rows; master -> {args.master_csv}")


if __name__ == "__main__":
    main()
