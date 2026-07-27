import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import yaml
import time
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.hybrid.dataset import HybridPrecomputedDataset
from lib.repro.data_paths import detect_project_root, get_exp11_split_paths

matplotlib.use("Agg")
DATA_ROOT = detect_project_root()
SHM_SAFETY_RATIO = 0.9


def _masked_mean_pool(x: torch.Tensor, valid_out_lens: torch.Tensor) -> torch.Tensor:
    """Mean-pool x over valid time positions only, ignoring zero-padded tail.

    x: [B, C, T]  (output of final CNN layer, T may include zero-derived padding)
    valid_out_lens: [B] int64, number of valid frames in x after CNN downsampling
    Returns: [B, C]
    """
    T = x.shape[2]
    mask = torch.arange(T, device=x.device).unsqueeze(0) < valid_out_lens.unsqueeze(1)  # [B, T]
    mask = mask.unsqueeze(1).to(x.dtype)  # [B, 1, T]
    return (x * mask).sum(-1) / mask.sum(-1).clamp(min=1)


class LossCurveCNN(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.feature_dim = 128

    def forward(self, x, lengths=None):
        # x: [B, C, T]; lengths: [B] valid frames before padding
        out = self.cnn(x)  # [B, 128, T//4]
        if lengths is not None:
            # Two MaxPool1d(2): valid_out = floor(floor(L / 2) / 2) = L // 4
            valid_out = (lengths // 4).clamp(min=1)
            return _masked_mean_pool(out, valid_out)
        return out.mean(-1)


class SAETower(nn.Module):
    def __init__(self, sae_dim=16384, reduced_dim=256):
        super().__init__()
        self.compress = nn.Sequential(nn.Linear(sae_dim, reduced_dim), nn.ReLU(), nn.LayerNorm(reduced_dim))
        self.cnn = nn.Sequential(
            nn.Conv1d(reduced_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.feature_dim = 128

    def forward(self, x, lengths=None):
        # x: [B, T, sae_dim]; lengths: [B] valid SAE frames before padding
        x = self.compress(x)      # [B, T, reduced_dim]
        x = x.permute(0, 2, 1)    # [B, reduced_dim, T]
        x = self.cnn(x)            # [B, 128, T_out]
        if lengths is not None:
            # Two Conv1d(stride=2, kernel=5, padding=2): out = floor((L-1)/2) + 1
            valid_c1 = ((lengths - 1) // 2 + 1).clamp(min=1)
            valid_c2 = ((valid_c1 - 1) // 2 + 1).clamp(min=1)
            return _masked_mean_pool(x, valid_c2)
        return x.mean(-1)


class HybridTwoTowerModel(nn.Module):
    def __init__(self, loss_tower, sae_tower):
        super().__init__()
        fusion_dim = loss_tower.feature_dim + sae_tower.feature_dim
        self.loss_tower = loss_tower
        self.sae_tower = sae_tower
        self.head = nn.Sequential(nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(0.4), nn.Linear(fusion_dim // 2, 1))

    def forward(self, loss_curve, sae_features, curve_len=None, sae_len=None):
        return self.head(torch.cat([self.loss_tower(loss_curve, curve_len), self.sae_tower(sae_features, sae_len)], dim=1)).squeeze(-1)


class SAEOnlyModel(nn.Module):
    def __init__(self, sae_tower):
        super().__init__()
        self.sae_tower = sae_tower
        self.head = nn.Sequential(nn.Linear(sae_tower.feature_dim, sae_tower.feature_dim // 2), nn.ReLU(), nn.Dropout(0.4), nn.Linear(sae_tower.feature_dim // 2, 1))

    def forward(self, _loss_curve, sae_features, curve_len=None, sae_len=None):
        return self.head(self.sae_tower(sae_features, sae_len)).squeeze(-1)


def _safe_loader_params(
    seq_len: int,
    sae_dim: int,
    curve_channels: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    if num_workers <= 0:
        return batch_size, 0, prefetch_factor
    try:
        st = os.statvfs("/dev/shm")
        shm_avail = int(st.f_frsize * st.f_bavail)
    except Exception:
        shm_avail = 0
    if shm_avail <= 0:
        return batch_size, num_workers, prefetch_factor

    sample_bytes = int(seq_len) * int(max(1, sae_dim + max(0, curve_channels))) * 4
    batch_bytes = int(batch_size) * sample_bytes
    safe_limit = int(shm_avail * SHM_SAFETY_RATIO)
    nw = max(0, int(num_workers))
    pf = max(2, int(prefetch_factor))
    # Very conservative estimate for prefetched worker batches + pinned copies.
    est_inflight = batch_bytes * max(1, nw) * max(1, pf) * 2
    while nw > 0 and est_inflight > safe_limit:
        if pf > 2:
            pf -= 1
        else:
            nw -= 1
        est_inflight = batch_bytes * max(1, nw) * max(1, pf) * 2
    return batch_size, nw, pf


def _get_lengths(batch, device):
    curve_len = batch["curve_len"].to(device, non_blocking=True) if "curve_len" in batch else None
    sae_len = batch["sae_len"].to(device, non_blocking=True) if "sae_len" in batch else None
    return curve_len, sae_len


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        loss_curve = batch["loss_curve"].to(device, non_blocking=True)
        sae_features = batch["sae_features"].to(device, non_blocking=True)
        scores = batch["score"].to(device, non_blocking=True).squeeze(-1)
        curve_len, sae_len = _get_lengths(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            preds = model(loss_curve, sae_features, curve_len, sae_len)
            loss = criterion(preds, scores)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            loss_curve = batch["loss_curve"].to(device, non_blocking=True)
            sae_features = batch["sae_features"].to(device, non_blocking=True)
            scores = batch["score"].to(device, non_blocking=True).squeeze(-1)
            curve_len, sae_len = _get_lengths(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                preds = model(loss_curve, sae_features, curve_len, sae_len)
                loss = criterion(preds, scores)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    if len(all_preds) < 2:
        return float("nan"), float("nan"), float("nan"), all_preds, all_scores
    p, _ = pearsonr(all_preds, all_scores)
    s, _ = spearmanr(all_preds, all_scores)
    return total_loss / len(loader), p, s, all_preds, all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr-decay-factor", type=float, default=None)
    parser.add_argument("--lr-decay-patience", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--run_name", type=str, default="cnn")
    parser.add_argument("--sae_suffix", type=str, default="")
    parser.add_argument("--curve-mode", type=str, default="loss", choices=["loss", "entropy", "loss_entropy", "none"])
    parser.add_argument("--entropy-manifest-csv", action="append", default=None)
    parser.add_argument("--num-workers", type=int, default=max(2, min(8, os.cpu_count() or 4)))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg.get("training", {}).get("defaults", {}).get("hybrid_cnn", {})
    epochs = int(args.epochs if args.epochs is not None else tcfg.get("epochs", 50))
    default_batch_size = int(tcfg.get("batch_size", 8))
    if args.curve_mode == "none":
        default_batch_size = max(default_batch_size, 32)
    else:
        default_batch_size = max(default_batch_size, 16)
    batch_size = int(args.batch_size if args.batch_size is not None else default_batch_size)
    lr = float(args.lr if args.lr is not None else tcfg.get("lr", 1e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else tcfg.get("weight_decay", 1e-4))
    patience = int(args.patience if args.patience is not None else tcfg.get("patience", 8))
    lr_decay_factor = float(args.lr_decay_factor if args.lr_decay_factor is not None else tcfg.get("lr_decay_factor", 0.5))
    lr_decay_patience = int(args.lr_decay_patience if args.lr_decay_patience is not None else tcfg.get("lr_decay_patience", 3))
    min_lr = float(args.min_lr if args.min_lr is not None else tcfg.get("min_lr", 1e-6))
    data_cfg = cfg.get("data", {})
    feature_roots = data_cfg.get("feature_roots", {})
    split_paths = get_exp11_split_paths(args.config, splits="clean")
    sae_feature_dir = cfg.get("sae", {}).get("output_dir", feature_roots.get("sae_feature_root", "features/sae"))
    token_loss_root = feature_roots.get("token_loss_root", "")
    seq_len = cfg.get("seq_len", 1500)
    sae_dim = cfg.get("sae_hidden_dim", 16384)
    out_cfg = cfg.get("outputs", {})
    project_root = cfg.get("project_root", DATA_ROOT)
    plot_dir = out_cfg.get("plots", "outputs/plots")
    save_dir = out_cfg.get("checkpoints", "outputs/checkpoints")
    report_dir = out_cfg.get("reports", "outputs/reports")
    plot_dir = plot_dir if os.path.isabs(plot_dir) else os.path.join(project_root, plot_dir)
    save_dir = save_dir if os.path.isabs(save_dir) else os.path.join(project_root, save_dir)
    report_dir = report_dir if os.path.isabs(report_dir) else os.path.join(project_root, report_dir)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    def _make_ds(split_name, csv_path):
        return HybridPrecomputedDataset(
            split_name,
            DATA_ROOT,
            split_csv_path=csv_path,
            sae_feature_dir=sae_feature_dir,
            token_loss_root=token_loss_root,
            seq_len=seq_len,
            sae_dim=sae_dim,
            sae_variant_suffix=args.sae_suffix,
            curve_mode=args.curve_mode,
            entropy_manifest_csv=args.entropy_manifest_csv,
        )

    train_ds = _make_ds("train", split_paths["train"])
    val_ds = _make_ds("val", split_paths["val"])
    test_ds = _make_ds("test", split_paths["test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "3060" in gpu_name:
            if batch_size > 32:
                print(f"[gpu-profile] {gpu_name}: cap sae-only batch_size {batch_size}->32 for stable throughput")
                batch_size = 32
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    req_workers = max(0, int(args.num_workers))
    req_prefetch = max(2, int(args.prefetch_factor))
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0).lower()
        # 3060 + SAE-only has huge per-sample tensors; worker IPC often dominates and keeps GPU idle.
        if "3060" in gpu_name and args.curve_mode == "none":
            if req_workers > 2:
                print(f"[gpu-profile] {gpu_name}: SAE-only mode caps num_workers {req_workers}->2 to balance IPC and throughput")
            req_workers = min(req_workers, 2)
        if "3060" in gpu_name:
            if req_workers > 4 or req_prefetch > 2:
                print(
                    f"[gpu-profile] {gpu_name}: tune loader workers {req_workers}->4, prefetch {req_prefetch}->2"
                )
            req_workers = min(req_workers, 4)
            req_prefetch = min(req_prefetch, 2)
    batch_size, tuned_workers, tuned_prefetch = _safe_loader_params(
        seq_len=seq_len,
        sae_dim=int(train_ds.sae_dim),
        curve_channels=int(train_ds.curve_channels),
        batch_size=batch_size,
        num_workers=req_workers,
        prefetch_factor=req_prefetch,
    )
    if tuned_workers != int(args.num_workers) or tuned_prefetch != int(args.prefetch_factor):
        print(
            f"[loader-autotune] shm guard active: workers {args.num_workers}->{tuned_workers}, "
            f"prefetch {args.prefetch_factor}->{tuned_prefetch}"
        )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": tuned_workers,
        "pin_memory": device.type == "cuda",
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = tuned_prefetch
    if args.curve_mode == "none" and device.type == "cuda" and loader_kwargs["num_workers"] == 0:
        # When fully single-process loading, pinning large host tensors often hurts more than helps.
        loader_kwargs["pin_memory"] = False
    run_tag = args.run_name
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    print(
        f"[startup] run={run_tag} curve_mode={args.curve_mode} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"sae_dim={train_ds.sae_dim} batch={batch_size} workers={loader_kwargs['num_workers']} "
        f"prefetch={loader_kwargs.get('prefetch_factor', 0)} pin_memory={loader_kwargs['pin_memory']}"
    )
    t_probe = time.time()
    _ = next(iter(train_loader))
    print(f"[startup] first train batch ready in {time.time() - t_probe:.2f}s")

    sae_tower = SAETower(sae_dim=int(train_ds.sae_dim))
    if args.curve_mode == "none":
        model = SAEOnlyModel(sae_tower).to(device)
    else:
        model = HybridTwoTowerModel(LossCurveCNN(input_channels=train_ds.curve_channels), sae_tower).to(device)

    # Multi-GPU: DataParallel wrapper
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"[multi-gpu] using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_decay_factor, patience=lr_decay_patience, min_lr=min_lr
    )
    criterion = nn.MSELoss()
    sae_tag = args.sae_suffix.lstrip("_") if args.sae_suffix else "newsae"
    best_val_pearson = -1.0
    curve_tag = args.curve_mode
    ckpt_path = os.path.join(save_dir, f"exp13_hybrid_cnn_{curve_tag}_{sae_tag}_{run_tag}_best.pth")

    no_improve = 0
    epoch_pbar = tqdm(range(epochs), desc=f"{run_tag}-epochs", unit="ep", dynamic_ncols=True)
    for epoch in epoch_pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_loss, val_p, val_s, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_p if not torch.isnan(torch.tensor(val_p)) else -1.0)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        epoch_pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            p=f"{val_p:.4f}",
            lr=f"{cur_lr:.2e}",
        )
        print(f"Epoch {epoch+1}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | ValP {val_p:.4f} | ValS {val_s:.4f} | lr={cur_lr:.2e}")
        if val_p > best_val_pearson:
            best_val_pearson = val_p
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Best saved: {ckpt_path}")
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    epoch_pbar.close()

    state_dict = torch.load(ckpt_path, map_location=device)
    # Handle DataParallel checkpoint keys: add or remove "module." prefix as needed
    new_state_dict = {}
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    is_dataparallel = hasattr(model, 'module')  # Check if model is wrapped
    
    for k, v in state_dict.items():
        if is_dataparallel and not has_module_prefix:
            # Model is DataParallel but checkpoint is not: add "module." prefix
            new_state_dict["module." + k] = v
        elif has_module_prefix and not is_dataparallel:
            # Checkpoint has "module." but model is not: remove prefix
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    test_loss, test_p, test_s, test_preds, test_scores = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss {test_loss:.4f} | Test Pearson {test_p:.4f} | Test Spearman {test_s:.4f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(test_scores, test_preds, alpha=0.5)
    plt.title(f"Hybrid CNN ({curve_tag}+sae) [{sae_tag}] {run_tag}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.savefig(os.path.join(plot_dir, f"exp13_hybrid_cnn_{curve_tag}_{sae_tag}_{run_tag}_scatter.png"))
    plt.close()
    out_df = pd.DataFrame({"score": test_scores, "pred": test_preds})
    if "audio_path" in test_ds.df.columns:
        out_df["audio_path"] = test_ds.df["audio_path"].values[: len(out_df)]
    out_df = out_df[["score", "audio_path", "pred"]] if "audio_path" in out_df.columns else out_df[["score", "pred"]]
    out_csv = os.path.join(report_dir, f"{run_tag}_test_scores.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
