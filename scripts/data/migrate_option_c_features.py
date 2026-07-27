#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


SPLITS = ("train", "val", "test")
DEFAULT_DATASETS = ("musicpref", "aime", "songeval", "musiceval", "music_arena")


def _copy_or_move(src: Path, dst: Path, move: bool) -> None:
    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _rewrite_col_prefix(csv_path: Path, col: str, old_prefix: Path, new_prefix: Path) -> None:
    if not csv_path.is_file():
        return
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        return
    old_s = str(old_prefix.resolve())
    new_s = str(new_prefix.resolve())
    df[col] = df[col].astype(str).map(
        lambda v: v.replace(old_s, new_s, 1) if v.startswith(old_s) else v
    )
    df.to_csv(csv_path, index=False)


def _has_split_dirs(base: Path, splits: Iterable[str] = SPLITS) -> bool:
    return all((base / s).is_dir() for s in splits)


def _target_option_c_ready(dst_root: Path, dataset: str, splits_label: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    loss_base = dst_root / "loss" / dataset / splits_label
    entropy_base = dst_root / "entropy" / dataset / splits_label
    sae_base = dst_root / "sae" / dataset / splits_label

    if not _has_split_dirs(loss_base):
        issues.append("loss split dirs missing")
    if not _has_split_dirs(entropy_base):
        issues.append("entropy split dirs missing")
    if not _has_split_dirs(sae_base):
        issues.append("sae split dirs missing")

    for split in SPLITS:
        if not (loss_base / f"split_{split}.csv").is_file():
            issues.append(f"loss split_{split}.csv missing")
        if not (loss_base / f"loss_manifest_{split}.csv").is_file():
            issues.append(f"loss_manifest_{split}.csv missing")
        if not (entropy_base / f"entropy_manifest_{split}.csv").is_file():
            issues.append(f"entropy_manifest_{split}.csv missing")
        if not (sae_base / f"sae_features_{split}_meta.pt").is_file():
            issues.append(f"sae_features_{split}_meta.pt missing")

    return (len(issues) == 0), issues


def _source_status(
    src_loss: Path,
    src_entropy: Path,
    src_sae: Path,
    dataset: str,
    splits_label: str,
) -> dict[str, bool]:
    return {
        "loss": (src_loss / dataset / splits_label).is_dir(),
        "entropy": (src_entropy / dataset / splits_label).is_dir(),
        "sae_dataset_scoped": (src_sae / dataset / splits_label).is_dir(),
        "sae_flat_root": _has_split_dirs(src_sae) and all((src_sae / f"sae_features_{s}_meta.pt").is_file() for s in SPLITS),
    }


def _split_csv(project_root: Path, dataset: str, split: str) -> Path:
    return project_root / "data" / "full_splits" / dataset / f"{split}.csv"


def _audio_overlap_ratio(manifest_csv: Path, split_csv: Path) -> float:
    if not manifest_csv.is_file() or not split_csv.is_file():
        return 0.0
    mdf = pd.read_csv(manifest_csv)
    sdf = pd.read_csv(split_csv)
    if "audio_path" not in mdf.columns or "audio_path" not in sdf.columns:
        return 0.0
    ma = set(mdf["audio_path"].astype(str))
    sa = set(sdf["audio_path"].astype(str))
    if not ma:
        return 0.0
    return len(ma.intersection(sa)) / float(len(ma))


def _entropy_source_score(project_root: Path, entropy_root: Path, dataset: str, splits_label: str) -> float:
    base = entropy_root / dataset / splits_label
    if not base.is_dir():
        return 0.0

    overlap_scores = []
    row_scores = []
    contam_scores = []
    for split in SPLITS:
        manifest = base / f"entropy_manifest_{split}.csv"
        split_csv = _split_csv(project_root, dataset, split)
        overlap_scores.append(_audio_overlap_ratio(manifest, split_csv))

        if manifest.is_file() and split_csv.is_file():
            mdf = pd.read_csv(manifest)
            sdf = pd.read_csv(split_csv)
            m = len(mdf)
            s = len(sdf)
            row_score = max(0.0, 1.0 - abs(m - s) / float(max(s, 1)))
            row_scores.append(row_score)

            if "audio_path" in mdf.columns and dataset != "musiceval":
                contam = mdf["audio_path"].astype(str).str.contains("musiceval|MusicEval-full", case=False, na=False).mean()
                contam_scores.append(1.0 - float(contam))
            else:
                contam_scores.append(1.0)
        else:
            row_scores.append(0.0)
            contam_scores.append(0.0)

    if not overlap_scores:
        return 0.0

    overlap = sum(overlap_scores) / len(overlap_scores)
    row_fit = sum(row_scores) / len(row_scores)
    contamination_ok = sum(contam_scores) / len(contam_scores)
    return 0.4 * overlap + 0.4 * row_fit + 0.2 * contamination_ok


def _pick_entropy_source(
    project_root: Path,
    primary_root: Path,
    fallback_root: Path,
    dataset: str,
    splits_label: str,
    min_ok_score: float = 0.5,
) -> Path:
    primary_score = _entropy_source_score(project_root, primary_root, dataset, splits_label)
    fallback_score = _entropy_source_score(project_root, fallback_root, dataset, splits_label)

    if primary_score >= min_ok_score:
        print(f"[source] entropy root: {primary_root} (score={primary_score:.3f})")
        return primary_root

    if fallback_score > primary_score and fallback_score >= min_ok_score:
        print(
            f"[warn] entropy source looks mismatched for {dataset}: "
            f"primary={primary_score:.3f}, fallback={fallback_score:.3f}. "
            f"Using fallback root: {fallback_root}"
        )
        return fallback_root

    print(
        f"[warn] entropy source scores are low for {dataset}: "
        f"primary={primary_score:.3f}, fallback={fallback_score:.3f}. "
        f"Keeping primary root: {primary_root}"
    )
    return primary_root


def _run_precheck(
    datasets: Iterable[str],
    src_loss: Path,
    src_entropy: Path,
    src_sae: Path,
    dst_root: Path,
    splits_label: str,
) -> bool:
    print("[precheck] dataset feature path status")
    ok = True
    for ds in datasets:
        src = _source_status(src_loss, src_entropy, src_sae, ds, splits_label)
        ready, issues = _target_option_c_ready(dst_root, ds, splits_label)
        has_sae = src["sae_dataset_scoped"] or src["sae_flat_root"]
        row_ok = src["loss"] and src["entropy"] and has_sae
        ok = ok and row_ok
        print(
            f"- {ds}: src(loss={src['loss']}, entropy={src['entropy']}, "
            f"sae_dataset={src['sae_dataset_scoped']}, sae_flat={src['sae_flat_root']}) "
            f"target_ready={ready}"
        )
        if issues and not ready:
            print(f"  target gaps: {', '.join(issues[:4])}{' ...' if len(issues) > 4 else ''}")
    return ok


def _migrate_loss_or_entropy(
    kind: str,
    src_root: Path,
    dst_root: Path,
    dataset: str,
    splits_label: str,
    move: bool,
) -> tuple[Path, Path]:
    src_base = src_root / dataset / splits_label
    dst_base = dst_root / kind / dataset / splits_label
    if not src_base.is_dir():
        raise FileNotFoundError(f"Missing source directory: {src_base}")

    for split in SPLITS:
        src_dir = src_base / split
        if src_dir.is_dir():
            _copy_or_move(src_dir, dst_base / split, move=move)

    if kind == "loss":
        extra = [f"loss_manifest_{s}.csv" for s in SPLITS] + [f"split_{s}.csv" for s in SPLITS]
    else:
        extra = [f"entropy_manifest_{s}.csv" for s in SPLITS]
    for name in extra:
        src_file = src_base / name
        if src_file.is_file():
            _copy_or_move(src_file, dst_base / name, move=move)

    return src_base, dst_base


def _migrate_sae(src_root: Path, dst_root: Path, dataset: str, splits_label: str, move: bool) -> tuple[Path, Path]:
    dataset_scoped = src_root / dataset / splits_label
    src_base = dataset_scoped if dataset_scoped.is_dir() else src_root
    dst_base = dst_root / "sae" / dataset / splits_label
    dst_base.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src_dir = src_base / split
        if not src_dir.is_dir():
            fallback_dir = src_root / split
            if fallback_dir.is_dir():
                src_dir = fallback_dir
        if src_dir.is_dir():
            _copy_or_move(src_dir, dst_base / split, move=move)

        src_meta = src_base / f"sae_features_{split}_meta.pt"
        if not src_meta.is_file():
            fallback_meta = src_root / f"sae_features_{split}_meta.pt"
            if fallback_meta.is_file():
                src_meta = fallback_meta
        dst_meta = dst_base / f"sae_features_{split}_meta.pt"
        if src_meta.is_file():
            _copy_or_move(src_meta, dst_meta, move=move)
            meta = torch.load(dst_meta, map_location="cpu", weights_only=False)
            if isinstance(meta, dict):
                meta["shard_dir"] = str((dst_base / split).resolve())
                torch.save(meta, dst_meta)

    return src_base, dst_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate feature folders to Option C layout.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="music_arena")
    parser.add_argument("--splits-label", type=str, default="clean")
    parser.add_argument("--source-loss-root", type=str, default="outputs/full/features/loss")
    parser.add_argument("--source-entropy-root", type=str, default="outputs/full/features/entropy")
    parser.add_argument("--fallback-entropy-root", type=str, default="outputs/features/entropy")
    parser.add_argument("--source-sae-root", type=str, default="features/sae")
    parser.add_argument("--target-root", type=str, default="features")
    parser.add_argument("--all-datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--check-all-datasets-first", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying.")
    parser.add_argument("--no-rewrite-manifests", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    src_loss = (project_root / args.source_loss_root).resolve()
    src_entropy_primary = (project_root / args.source_entropy_root).resolve()
    src_entropy_fallback = (project_root / args.fallback_entropy_root).resolve()
    src_sae = (project_root / args.source_sae_root).resolve()
    dst_root = (project_root / args.target_root).resolve()

    src_entropy = _pick_entropy_source(
        project_root,
        src_entropy_primary,
        src_entropy_fallback,
        args.dataset,
        args.splits_label,
    )

    datasets = [d.strip() for d in args.all_datasets.split(",") if d.strip()]

    if args.check_all_datasets_first or args.check_only:
        _run_precheck(datasets, src_loss, src_entropy_primary, src_sae, dst_root, args.splits_label)
        if args.check_only:
            return 0

    print(f"[migrate] dataset={args.dataset} splits={args.splits_label} mode={'move' if args.move else 'copy'}")

    old_loss_base, new_loss_base = _migrate_loss_or_entropy(
        "loss", src_loss, dst_root, args.dataset, args.splits_label, args.move
    )
    old_entropy_base, new_entropy_base = _migrate_loss_or_entropy(
        "entropy", src_entropy, dst_root, args.dataset, args.splits_label, args.move
    )
    _migrate_sae(src_sae, dst_root, args.dataset, args.splits_label, args.move)

    if not args.no_rewrite_manifests:
        for split in SPLITS:
            _rewrite_col_prefix(
                new_loss_base / f"split_{split}.csv",
                "token_loss_path",
                old_loss_base,
                new_loss_base,
            )
            _rewrite_col_prefix(
                new_loss_base / f"loss_manifest_{split}.csv",
                "loss_curve_path",
                old_loss_base,
                new_loss_base,
            )
            _rewrite_col_prefix(
                new_entropy_base / f"entropy_manifest_{split}.csv",
                "entropy_curve_path",
                old_entropy_base,
                new_entropy_base,
            )

    print(f"[done] loss -> {new_loss_base}")
    print(f"[done] entropy -> {new_entropy_base}")
    print(f"[done] sae -> {dst_root / 'sae' / args.dataset / args.splits_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
