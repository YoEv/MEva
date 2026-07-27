import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.repro.data_paths import get_exp11_split_paths, resolve_project_root
from lib.repro.entropy_dataset import EntropyCurveDataset, entropy_curve_collate_fn
from lib.repro.nets import LossCurveCNN

MAX_LEN = 1500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--entropy-manifest-csv", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument("--run-name", type=str, default="entropy_curve_cnn")
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_cfg = cfg.get("outputs", {})
    project_root = resolve_project_root(cfg)

    paths = get_exp11_split_paths(args.config, splits=args.splits)
    test_ds = EntropyCurveDataset(paths["test"], max_len=MAX_LEN, entropy_manifest_csv=args.entropy_manifest_csv)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=entropy_curve_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LossCurveCNN(input_channels=int(test_ds.output_channels), sequence_length=MAX_LEN).to(device)
    ckpt_dir = out_cfg.get("checkpoints", "outputs/checkpoints")
    ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(project_root, ckpt_dir)
    ckpt = os.path.join(ckpt_dir, f"{args.run_name}_best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            if inputs.nelement() == 0:
                continue
            out = model(inputs.to(device))
            preds.extend(out.cpu().numpy().ravel().tolist())

    test_df = pd.read_csv(paths["test"])
    if len(preds) != len(test_df):
        raise RuntimeError(
            f"Prediction count mismatch for entropy model: preds={len(preds)} vs test_rows={len(test_df)}. "
            "This usually means test samples were filtered due entropy-manifest key mismatch."
        )
    out_df = test_df[["score", "audio_path"]].copy()
    out_df["pred"] = preds
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
