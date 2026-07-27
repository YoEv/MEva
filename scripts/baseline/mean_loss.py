import argparse
import os

import pandas as pd
from tqdm import tqdm

from lib.repro.data_paths import load_exp11_splits


def mean_loss_from_csv(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    if "avg_loss_value" in df.columns:
        return float(df["avg_loss_value"].mean())
    col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split_map = load_exp11_splits(args.config, splits=args.splits)
    suffix = ""

    for split_name, df in split_map.items():
        means = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"mean_loss {split_name}"):
            p = row["token_loss_path"]
            if not os.path.exists(p):
                means.append(float("nan"))
                continue
            means.append(mean_loss_from_csv(p))
        out_df = df[["score", "audio_path"]].copy()
        out_df["mean_loss"] = means
        out_path = os.path.join(args.out_dir, f"mean_loss_{split_name}{suffix}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
