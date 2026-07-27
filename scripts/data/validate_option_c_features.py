#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")


def _check(cond: bool, ok: str, bad: str) -> bool:
    if cond:
        print(f"[ok] {ok}")
        return True
    print(f"[bad] {bad}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Option C feature integrity.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="music_arena")
    parser.add_argument("--splits-label", type=str, default="clean")
    parser.add_argument("--feature-root", type=str, default="features")
    parser.add_argument("--split-root", type=str, default="data/full_splits")
    parser.add_argument("--check-contamination", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    feature_root = (root / args.feature_root).resolve()
    split_root = (root / args.split_root / args.dataset).resolve()

    loss_base = feature_root / "loss" / args.dataset / args.splits_label
    entropy_base = feature_root / "entropy" / args.dataset / args.splits_label
    sae_base = feature_root / "sae" / args.dataset / args.splits_label

    status = True
    status &= _check(loss_base.is_dir(), f"loss base exists: {loss_base}", f"missing loss base: {loss_base}")
    status &= _check(entropy_base.is_dir(), f"entropy base exists: {entropy_base}", f"missing entropy base: {entropy_base}")
    status &= _check(sae_base.is_dir(), f"sae base exists: {sae_base}", f"missing sae base: {sae_base}")

    for split in SPLITS:
        split_csv = split_root / f"{split}.csv"
        split_df = pd.read_csv(split_csv)
        expected = len(split_df)

        loss_split_csv = loss_base / f"split_{split}.csv"
        loss_manifest = loss_base / f"loss_manifest_{split}.csv"
        entropy_manifest = entropy_base / f"entropy_manifest_{split}.csv"
        sae_meta = sae_base / f"sae_features_{split}_meta.pt"

        status &= _check(loss_split_csv.is_file(), f"loss split manifest exists: {loss_split_csv}", f"missing {loss_split_csv}")
        status &= _check(loss_manifest.is_file(), f"loss manifest exists: {loss_manifest}", f"missing {loss_manifest}")
        status &= _check(entropy_manifest.is_file(), f"entropy manifest exists: {entropy_manifest}", f"missing {entropy_manifest}")
        status &= _check(sae_meta.is_file(), f"sae meta exists: {sae_meta}", f"missing {sae_meta}")

        if loss_split_csv.is_file():
            ldf = pd.read_csv(loss_split_csv)
            status &= _check(
                len(ldf) == expected,
                f"loss split row count {split}: {len(ldf)}",
                f"loss split row mismatch {split}: got {len(ldf)} expected {expected}",
            )
            if "token_loss_path" in ldf.columns:
                missing = (~ldf["token_loss_path"].astype(str).map(lambda p: Path(p).is_file())).sum()
                status &= _check(missing == 0, f"loss files exist {split}", f"missing loss files {split}: {int(missing)}")

        if entropy_manifest.is_file():
            edf = pd.read_csv(entropy_manifest)
            status &= _check(
                len(edf) == expected,
                f"entropy row count {split}: {len(edf)}",
                f"entropy row mismatch {split}: got {len(edf)} expected {expected}",
            )
            if "entropy_curve_path" in edf.columns:
                missing = (~edf["entropy_curve_path"].astype(str).map(lambda p: Path(p).is_file())).sum()
                status &= _check(missing == 0, f"entropy files exist {split}", f"missing entropy files {split}: {int(missing)}")

            if args.check_contamination and "audio_path" in edf.columns:
                bad = edf["audio_path"].astype(str).str.contains("musiceval", case=False, na=False).sum()
                status &= _check(bad == 0, f"no musiceval contamination in entropy {split}", f"contamination in entropy {split}: {int(bad)} musiceval-like rows")

    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
