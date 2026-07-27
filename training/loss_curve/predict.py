import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.repro.data_paths import get_exp11_split_paths, resolve_project_root
from lib.repro.loss_dataset import LossCurveDataset, loss_curve_collate_fn
from lib.repro.nets import LossCurveCNN

MAX_LEN = 1500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument("--run-name", type=str, default="loss_curve_cnn")
    parser.add_argument("--entropy-manifest-csv", action="append", default=None)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_cfg = cfg.get("outputs", {})
    project_root = resolve_project_root(cfg)

    paths = get_exp11_split_paths(args.config, splits=args.splits)
    feature_roots = cfg.get("data", {}).get("feature_roots", {})
    token_loss_root = feature_roots.get("token_loss_root", "")
    token_loss_root = token_loss_root if os.path.isabs(token_loss_root) else os.path.join(project_root, token_loss_root) if token_loss_root else None
    test_ds = LossCurveDataset(
        paths["test"],
        max_len=MAX_LEN,
        entropy_manifest_csv=args.entropy_manifest_csv,
        skip_if_entropy_missing=False,
        entropy_fill_missing=bool(args.entropy_manifest_csv),
        feature_root=token_loss_root,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=loss_curve_collate_fn)
    in_ch = int(test_ds.output_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LossCurveCNN(input_channels=in_ch, sequence_length=MAX_LEN).to(device)
    ckpt_dir = out_cfg.get("checkpoints", "outputs/checkpoints")
    ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(project_root, ckpt_dir)
    ckpt = os.path.join(ckpt_dir, f"{args.run_name}_best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    state_dict = torch.load(ckpt, map_location=device)
    # Handle DataParallel checkpoint keys: add or remove "module." prefix as needed
    new_state_dict = {}
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    is_dataparallel = hasattr(model, 'module')
    
    for k, v in state_dict.items():
        if is_dataparallel and not has_module_prefix:
            new_state_dict["module." + k] = v
        elif has_module_prefix and not is_dataparallel:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            if inputs.nelement() == 0:
                continue
            out = model(inputs.to(device))
            preds.extend(out.cpu().numpy().ravel().tolist())

    test_df = pd.read_csv(paths["test"])
    out_df = test_df[["score", "audio_path"]].copy()
    out_df["pred"] = preds
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
