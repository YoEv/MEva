"""
Compute Pearson/Spearman correlations between Audiobox Aesthetics scores
and human ground-truth scores.

Since Audiobox Aesthetics is zero-shot (no training), we compute on ALL
annotated clips, not just the test split.

Ground-truth sources:
  musiceval  - data/splits/musiceval/{train,val,test}.csv
               (concatenated; 'score' column = MOS)
  songeval   - raw_hf/SongEval/metadata.jsonl
               (mean Musicality across all annotators)
  aime       - data/manifests/aime_track_winrate_by_question_type.csv
               (Music Quality win-rate, all 1300 tracks)
  musicpref  - raw_hf/MusicPref/human_preference.csv
               (musicality win-rate, all 5030 clips)

Outputs:
  outputs/aesthetics/correlations_all.csv
  outputs/aesthetics/correlations_all.md
  (also per-split for musiceval: correlations_test.csv / .md)
"""
import os
import re
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PH7_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT  = os.path.dirname(PH7_ROOT)

RAW_HF    = os.path.join(PH7_ROOT, "raw_hf")
MANIFESTS = os.path.join(PH7_ROOT, "data", "manifests")
SPLITS    = os.path.join(PH7_ROOT, "data", "splits")
OUT_BASE  = os.path.join(PH7_ROOT, "outputs", "aesthetics")

AXES = ["CE", "CU", "PC", "PQ"]


# ── Ground-truth loaders (ALL annotated clips) ────────────────────────────────

def gt_musiceval_all():
    """All train+val+test MOS scores."""
    frames = []
    for split in ("train", "val", "test"):
        p = os.path.join(SPLITS, "musiceval", f"{split}.csv")
        df = pd.read_csv(p)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["audio_id"] = df["audio_path"].apply(
        lambda p: os.path.splitext(os.path.basename(p))[0])
    return df[["audio_id", "score", "split"]]


def gt_musiceval_test():
    """Test-split only (for per-split breakdown)."""
    df = pd.read_csv(os.path.join(SPLITS, "musiceval", "test.csv"))
    df["audio_id"] = df["audio_path"].apply(
        lambda p: os.path.splitext(os.path.basename(p))[0])
    df["split"] = "test"
    return df[["audio_id", "score", "split"]]


def gt_songeval():
    """Mean Musicality across all annotators, all 2399 clips."""
    rows = []
    with open(os.path.join(RAW_HF, "SongEval", "metadata.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            audio_id = os.path.splitext(os.path.basename(obj["file_name"]))[0]
            score = float(np.mean([a["Musicality"] for a in obj["annotation"]]))
            rows.append({"audio_id": audio_id, "score": score})
    return pd.DataFrame(rows)


def gt_aime():
    """Music Quality win-rate for all 1300 tracks."""
    df = pd.read_csv(
        os.path.join(MANIFESTS, "aime_track_winrate_by_question_type.csv"))
    mq = df[df["question_type"] == "Music Quality"][["id", "win_rate"]].copy()
    mq.rename(columns={"id": "audio_id", "win_rate": "score"}, inplace=True)
    mq["audio_id"] = mq["audio_id"].astype(int)
    return mq


def gt_musicpref():
    """
    Musicality win-rate for all 5030 clips.
    audio_id K = pair_id * 2 + side  (matches parquet path '{pair_id}_{side}.wav')
    """
    pref = pd.read_csv(os.path.join(RAW_HF, "MusicPref", "human_preference.csv"))
    records = {}
    for pair_id, row in enumerate(pref.itertuples(index=False)):
        id_a, id_b = pair_id * 2, pair_id * 2 + 1
        for aid in (id_a, id_b):
            records.setdefault(aid, {"wins": 0, "n": 0})
        records[id_a]["n"] += 1
        records[id_b]["n"] += 1
        result = row.musicality
        if result == "a_wins":
            records[id_a]["wins"] += 1
        elif result == "b_wins":
            records[id_b]["wins"] += 1
        else:
            records[id_a]["wins"] += 0.5
            records[id_b]["wins"] += 0.5
    rows = [{"audio_id": k, "score": v["wins"] / v["n"]}
            for k, v in records.items() if v["n"] > 0]
    return pd.DataFrame(rows)


# ── Correlation ───────────────────────────────────────────────────────────────

def correlate(gt_df, aes_df, axis):
    gt  = gt_df.copy();  gt["audio_id"]  = gt["audio_id"].astype(str)
    aes = aes_df.copy(); aes["audio_id"] = aes["audio_id"].astype(str)
    m   = gt.merge(aes[["audio_id", axis]], on="audio_id", how="inner").dropna()
    n   = len(m)
    if n < 2:
        return float("nan"), float("nan"), n
    r,   _ = pearsonr( m["score"], m[axis])
    rho, _ = spearmanr(m["score"], m[axis])
    return round(r, 4), round(rho, 4), n


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    configs = [
        # (label,       gt_loader,          dataset_key)
        ("musiceval",   gt_musiceval_all,   "musiceval"),
        ("musiceval_test", gt_musiceval_test, "musiceval"),
        ("songeval",    gt_songeval,        "songeval"),
        ("aime",        gt_aime,            "aime"),
        ("musicpref",   gt_musicpref,       "musicpref"),
    ]

    all_rows = []
    for label, gt_loader, ds_key in configs:
        aes_path = os.path.join(OUT_BASE, ds_key, "aesthetics_scores.csv")
        if not os.path.exists(aes_path):
            print(f"[{label}] aesthetics_scores.csv not found – skip")
            continue
        aes_df = pd.read_csv(aes_path)
        gt_df  = gt_loader()
        print(f"\n[{label}] GT rows: {len(gt_df)}, Aesthetics rows: {len(aes_df)}")

        for axis in AXES:
            r, rho, n = correlate(gt_df, aes_df, axis)
            print(f"  Aesthetics-{axis}: Pearson={r:>7.4f}, Spearman={rho:>7.4f}, n={n}")
            all_rows.append({"label": label, "method": f"Aesthetics-{axis}",
                              "pearson": r, "spearman": rho, "n": n})

    df = pd.DataFrame(all_rows)
    os.makedirs(OUT_BASE, exist_ok=True)

    csv_path = os.path.join(OUT_BASE, "correlations_all.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    md_path = os.path.join(OUT_BASE, "correlations_all.md")
    with open(md_path, "w") as f:
        f.write("| dataset | method | pearson | spearman | n |\n")
        f.write("|---------|--------|--------:|---------:|--:|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['label']} | {row['method']} "
                    f"| {row['pearson']} | {row['spearman']} | {row['n']} |\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
