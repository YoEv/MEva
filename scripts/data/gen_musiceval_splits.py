#!/usr/bin/env python3
"""Generate data/splits/musiceval/ CSVs from MusicEval-full set lists."""
import os
import pandas as pd
from pathlib import Path

MUSICEVAL_ROOT = Path("datasets/musiceval/MusicEval-full")
WAV_DIR = MUSICEVAL_ROOT / "wav"
SETS_DIR = MUSICEVAL_ROOT / "sets"
OUT_DIR = Path("data/splits/musiceval")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mos_list(txt_path: Path) -> pd.DataFrame:
    rows = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        fname, s1, s2 = parts[0].strip(), float(parts[1]), float(parts[2])
        rows.append({
            "score": (s1 + s2) / 2,
            "audio_path": os.path.abspath(str(WAV_DIR / fname)),
            "token_loss_path": f"musiceval/{fname}",
        })
    return pd.DataFrame(rows)


mapping = {
    "train": "train_mos_list.txt",
    "val": "dev_mos_list.txt",
    "test": "test_mos_list.txt",
}

for split, fname in mapping.items():
    df = load_mos_list(SETS_DIR / fname)
    missing = (~df["audio_path"].map(os.path.exists)).sum()
    if missing:
        print(f"WARNING {split}: {missing}/{len(df)} audio files missing")
    df.to_csv(OUT_DIR / f"{split}.csv", index=False)
    print(f"{split}: {len(df)} rows")

print("Done.")
