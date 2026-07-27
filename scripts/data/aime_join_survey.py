#!/usr/bin/env python3
"""Join disco-eth/AIME audio ids with disco-eth/AIME-survey pairwise labels.

AIME audio uses string ids like "04501"; survey uses int track ids 4501. They match as int(audio_id) == track_id.
Survey rows are pairwise: answer 1 = prefer track 1, answer 2 = prefer track 2.
Outputs:
  - aime_pairwise_survey.csv — full survey with normalized id columns
  - aime_track_winrate.csv — per-track wins / comparisons / win_rate (scalar heuristic)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--survey-parquet",
        type=Path,
        required=True,
        help="Path to AIME-survey train parquet (e.g. raw_hf/AIME-survey/data/train-00000-of-00001.parquet)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for output CSVs (e.g. data/manifests)",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sv = pd.read_parquet(args.survey_parquet)
    # HF column names use hyphens; normalize to underscores for CSV
    rename = {c: c.replace("-", "_") for c in sv.columns}
    sv = sv.rename(columns=rename)

    sv["track_1_id_str"] = sv["track_1_id"].map(lambda x: f"{int(x):05d}")
    sv["track_2_id_str"] = sv["track_2_id"].map(lambda x: f"{int(x):05d}")

    pairwise_path = args.out_dir / "aime_pairwise_survey.csv"
    sv.to_csv(pairwise_path, index=False)

    # Per-appearance: each row gives one win and one loss among the two tracks
    wcol = "question_type" if "question_type" in sv.columns else sv.columns[0]
    one = sv.assign(
        winner_id_str=lambda d: d.apply(
            lambda r: r["track_1_id_str"] if r["answer"] == 1 else r["track_2_id_str"],
            axis=1,
        ),
        loser_id_str=lambda d: d.apply(
            lambda r: r["track_2_id_str"] if r["answer"] == 1 else r["track_1_id_str"],
            axis=1,
        ),
    )
    wins = one[["winner_id_str", wcol]].rename(columns={"winner_id_str": "id"})
    wins["win"] = 1
    losses = one[["loser_id_str", wcol]].rename(columns={"loser_id_str": "id"})
    losses["win"] = 0
    long_df = pd.concat([wins, losses], ignore_index=True)

    agg = (
        long_df.groupby(["id", wcol], as_index=False)
        .agg(wins=("win", "sum"), n=("win", "count"))
        .assign(win_rate=lambda d: d["wins"] / d["n"])
    )
    overall = (
        long_df.groupby("id", as_index=False)
        .agg(wins=("win", "sum"), n=("win", "count"))
        .assign(win_rate=lambda d: d["wins"] / d["n"])
    )

    winrate_path = args.out_dir / "aime_track_winrate.csv"
    overall.to_csv(winrate_path, index=False)
    strat_path = args.out_dir / "aime_track_winrate_by_question_type.csv"
    agg.to_csv(strat_path, index=False)

    print(f"Wrote {pairwise_path} ({len(sv)} rows)")
    print(f"Wrote {winrate_path} ({len(overall)} tracks)")
    print(f"Wrote {strat_path} ({len(agg)} track x question_type)")


if __name__ == "__main__":
    main()
