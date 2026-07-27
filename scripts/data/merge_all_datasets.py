#!/usr/bin/env python3
"""Merge per-dataset full_splits into all_5_datasets (or all available)."""
import pandas as pd
from pathlib import Path

names = ["musicpref", "aime", "songeval", "music_arena", "musiceval"]
splits = ["train", "val", "test"]
out = Path("data/full_splits/all_5_datasets")
out.mkdir(parents=True, exist_ok=True)

for s in splits:
    parts = []
    for n in names:
        p = Path(f"data/full_splits/{n}/{s}.csv")
        if p.exists():
            df = pd.read_csv(p)
            df["source"] = n
            parts.append(df)
    if not parts:
        print(f"SKIP {s}: no parts found")
        continue
    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(out / f"{s}.csv", index=False)
    print(f"{s}: {len(merged)} rows from {[p['source'].iloc[0] for p in parts]}")
