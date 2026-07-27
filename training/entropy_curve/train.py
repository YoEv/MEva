import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from lib.repro.data_paths import get_exp11_split_paths, resolve_project_root
from lib.repro.entropy_dataset import EntropyCurveDataset, entropy_curve_collate_fn
from lib.repro.nets import LossCurveCNN

MAX_LEN = 1500


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    n_batches = 0
    for inputs, targets in loader:
        if inputs.nelement() == 0:
            continue
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n_batches += 1
    return total / n_batches if n_batches > 0 else float("nan")


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, t in loader:
            if inputs.nelement() == 0:
                continue
            out = model(inputs.to(device))
            preds.extend(out.cpu().numpy().ravel())
            targets.extend(t.numpy().ravel())
    preds = np.array(preds)
    targets = np.array(targets)
    if len(preds) < 2:
        return float("nan"), float("nan")
    loss = np.mean((preds - targets) ** 2)
    corr = pearsonr(preds, targets)[0]
    return float(loss), float(corr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--entropy-manifest-csv", action="append", required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr-decay-factor", type=float, default=None)
    parser.add_argument("--lr-decay-patience", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--splits", type=str, default="clean", choices=["clean"])
    parser.add_argument("--run-name", type=str, default="entropy_curve_cnn")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg.get("training", {}).get("defaults", {}).get("entropy_curve_cnn", {})
    lr = float(args.lr if args.lr is not None else tcfg.get("lr", 1e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else tcfg.get("weight_decay", 1e-4))
    epochs = int(args.epochs if args.epochs is not None else tcfg.get("epochs", 100))
    batch_size = int(args.batch_size if args.batch_size is not None else tcfg.get("batch_size", 64))
    patience = int(args.patience if args.patience is not None else tcfg.get("patience", 10))
    lr_decay_factor = float(args.lr_decay_factor if args.lr_decay_factor is not None else tcfg.get("lr_decay_factor", 0.5))
    lr_decay_patience = int(args.lr_decay_patience if args.lr_decay_patience is not None else tcfg.get("lr_decay_patience", 3))
    min_lr = float(args.min_lr if args.min_lr is not None else tcfg.get("min_lr", 1e-6))

    out_cfg = cfg.get("outputs", {})
    project_root = resolve_project_root(cfg)
    ckpt_dir = out_cfg.get("checkpoints", "outputs/checkpoints")
    ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(project_root, ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    paths = get_exp11_split_paths(args.config, splits=args.splits)
    train_ds = EntropyCurveDataset(paths["train"], max_len=MAX_LEN, entropy_manifest_csv=args.entropy_manifest_csv)
    val_ds = EntropyCurveDataset(paths["val"], max_len=MAX_LEN, entropy_manifest_csv=args.entropy_manifest_csv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=entropy_curve_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=entropy_curve_collate_fn)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Entropy dataset is empty; check split csv and entropy manifests.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LossCurveCNN(input_channels=int(train_ds.output_channels), sequence_length=MAX_LEN).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_decay_factor, patience=lr_decay_patience, min_lr=min_lr
    )

    best_val_loss = float("inf")
    n_no_improve = 0
    ckpt_path = os.path.join(ckpt_dir, f"{args.run_name}_best.pth")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_corr = evaluate(model, val_loader, device)
        if not np.isfinite(train_loss):
            raise RuntimeError(
                "No valid training batches for entropy model. "
                "All samples were filtered by dataset/manifest alignment."
            )
        if not np.isfinite(val_loss):
            raise RuntimeError(
                "No valid validation batches for entropy model. "
                "All val samples were filtered by dataset/manifest alignment."
            )
        scheduler.step(val_loss)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_corr={val_corr:.4f} lr={cur_lr:.2e}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            n_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved {ckpt_path}")
        else:
            n_no_improve += 1
        if n_no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break


if __name__ == "__main__":
    main()
