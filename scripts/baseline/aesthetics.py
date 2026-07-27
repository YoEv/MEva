import argparse
import os
import sys

# Ensure repo root is on path (audiobox env doesn't have phase7_release installed)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from tqdm import tqdm

from lib.repro.data_paths import load_exp11_splits

AXES_NAME = ["CE", "CU", "PC", "PQ"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    args = parser.parse_args()

    from lib.repro.data_paths import detect_project_root
    repo_root = detect_project_root()
    aesthetics_src = os.path.join(repo_root, "external", "audiobox-aesthetics", "src")
    sys.path.insert(0, aesthetics_src)
    from audiobox_aesthetics.infer import AesPredictor

    os.makedirs(args.out_dir, exist_ok=True)
    predictor = AesPredictor(checkpoint_pth=args.ckpt, data_col="path")
    split_map = load_exp11_splits(args.config, splits=args.splits)
    suffix = ""

    for split_name, df in split_map.items():
        rows = [{"path": p} for p in df["audio_path"].tolist()]
        all_scores = []
        for i in tqdm(range(0, len(rows), args.batch_size), desc=f"aesthetics {split_name}"):
            batch = rows[i : i + args.batch_size]
            out = predictor.forward(batch)
            for row in out:
                all_scores.append([row[ax] for ax in AXES_NAME])
        out_df = df[["score", "audio_path"]].copy()
        for j, ax in enumerate(AXES_NAME):
            out_df[ax] = [s[j] for s in all_scores]
        out_path = os.path.join(args.out_dir, f"aesthetics_scores_{split_name}{suffix}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
