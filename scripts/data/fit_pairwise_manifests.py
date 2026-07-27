#!/usr/bin/env python3
"""
Fit per-track 1-5 scores from pairwise preference labels for the three
pairwise datasets used in phase7_release -- now using Elo (see
`lib/elo_scoring.py`).  The hinge/ReLU scorer that used to
live in `lib/pairwise_relu_scores.py` was removed because it collapses the
single-pair-per-clip datasets into {win, tie, lose} bands.

Output CSV paths, filenames and **column names are identical** to the
previous version (``source, <id_col>, score_1to5, ...``), so
``gen_full_splits.py`` and downstream consumers need no changes to their
readers.  Each row gains a few new columns (``begin_s``/``end_s`` for AIME,
``listen_sec_used`` for Music Arena, plus diagnostic system/raw-score
columns) -- the readers just ignore extras.

Per-dataset recipes
-------------------
``--dataset musicpref --head musicality``
    System-level Elo on the musicality axis (7 systems) + per-track
    single-match adjustment (``track_score_from_system_elo``) + robust
    affine map to [1, 5].  Fidelity / text-audio-alignment heads are
    intentionally **removed**: the downstream evaluation only uses
    musicality, and mixing axes dilutes the Elo signal.

``--dataset aime --head music_quality``
    System-level Elo on the 13 generators (diagnostic) + per-track
    ``delta_Elo = 400 * log10(p_hat / (1 - p_hat))`` with Laplace-smoothed
    win-rate p_hat = (W + alpha) / (n + 2 * alpha).  Every AIME track
    appears in 12 Music-Quality pairs, so p_hat is a well-identified
    per-track statistic.  Final label = system_elo + delta_Elo, mapped
    robust-affinely to [1, 5].  The survey's 10-second window
    (``track_*_begin`` / ``track_*_end``) is propagated as
    ``begin_s``/``end_s`` for the downstream extractors to honour.

``--dataset musicarena``
    System-level Elo on the 7 systems + context-aware per-clip target
    (unified treatment of A_WINS / B_WINS / TIE / BOTH_BAD with
    ``context_blend_alpha``) + per-clip adjustment + robust affine map to
    [1, 5].  The system baseline is also compressed to a fixed span
    (``--system_span``) before adding the per-clip adjustment so the
    7-band system collapse goes away.  Audio paths in the CSV point at
    the source MP3s; actual rater-aligned WAV cropping happens inside
    ``gen_full_splits.py`` which reads ``listen_sec_used``/``duration`` /
    source path and emits a cropped WAV + split CSV.

Examples
--------
::

    python fit_pairwise_manifests.py --dataset musicpref --head musicality \\
        --human-preference-csv raw_hf/MusicPref/human_preference.csv \\
        --out data/manifests/pairwise_relu/musicpref_musicality_1to5.csv

    python fit_pairwise_manifests.py --dataset aime --head music_quality \\
        --survey-csv data/manifests/aime_pairwise_survey.csv \\
        --out data/manifests/pairwise_relu/aime_music_quality_1to5.csv

    python fit_pairwise_manifests.py --dataset musicarena \\
        --battle-glob 'raw_hf/MusicArena/battle_data/**/*.json' \\
        --out data/manifests/pairwise_relu/musicarena_1to5.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# lib is where elo_scoring.py lives.
_LIB = Path(__file__).resolve().parents[2] / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from elo_scoring import (  # noqa: E402
    DEFAULT_OUTCOME_MAP,
    fit_elo_grouped,
    map_to_mos_range,
    track_score_from_system_elo,
)

# Repo root = parent of phase7_release.
_REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# MusicPref loader (musicality-only Elo input)
# --------------------------------------------------------------------------

def _parse_musicpref_outcome(val: str) -> str:
    s = str(val).strip().lower()
    if "a_wins" in s and "b_wins" not in s:
        return "a_wins"
    if "b_wins" in s:
        return "b_wins"
    return "tie"


def _load_musicpref(csv_path: Path) -> Tuple[
    List[Tuple[str, str, str]],  # pairs (a, b, outcome)
    List[Tuple[str, str, str, str, str]],  # enriched (a, b, sys_a, sys_b, outcome)
    Dict[str, str],  # track -> system
]:
    df = pd.read_csv(csv_path)
    if "musicality" not in df.columns:
        raise SystemExit(
            f"{csv_path}: expected a 'musicality' column; MusicPref fidelity / "
            "text-audio-alignment heads were intentionally removed."
        )
    pairs: List[Tuple[str, str, str]] = []
    enriched: List[Tuple[str, str, str, str, str]] = []
    track_system: Dict[str, str] = {}
    for _, row in df.iterrows():
        a = str(row["audio_a"]).strip()
        b = str(row["audio_b"]).strip()
        outcome = _parse_musicpref_outcome(str(row["musicality"]))
        sys_a = str(row["system_a"])
        sys_b = str(row["system_b"])
        pairs.append((a, b, outcome))
        enriched.append((a, b, sys_a, sys_b, outcome))
        track_system[a] = sys_a
        track_system[b] = sys_b
    return pairs, enriched, track_system


# --------------------------------------------------------------------------
# AIME loader (Music Quality only; Laplace-smoothed winrate -> Elo offset)
# --------------------------------------------------------------------------

def _ts_to_sec(s: str) -> float:
    parts = str(s).split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = "0", parts[0], parts[1]
    else:
        raise ValueError(f"bad timestamp: {s}")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _load_aime_survey(survey_csv: Path) -> pd.DataFrame:
    """Load the merged AIME pairwise survey (produced by aime_join_survey.py)."""
    df = pd.read_csv(survey_csv)
    rename = {c: c.replace("-", "_") for c in df.columns}
    df = df.rename(columns=rename)
    if "question_type" not in df.columns:
        raise SystemExit(f"{survey_csv}: missing 'question_type' column")
    mq = df[df["question_type"] == "Music Quality"].copy()
    if len(mq) != 7800:
        print(
            f"[warn] expected 7800 'Music Quality' rows, got {len(mq)} -- "
            "continuing but double-check the survey file."
        )

    def _zfill(col: str) -> pd.Series:
        if col in mq.columns:
            return mq[col].astype(str).str.zfill(5)
        alt = col.replace("_str", "")
        return mq[alt].astype(int).astype(str).str.zfill(5)

    mq["track_a"] = _zfill("track_1_id_str")
    mq["track_b"] = _zfill("track_2_id_str")
    mq["outcome"] = mq["answer"].map({1: "a_wins", 2: "b_wins"})
    if mq["outcome"].isna().any():
        raise SystemExit("AIME Music Quality rows contain answer values outside {1, 2}")
    return mq


# --------------------------------------------------------------------------
# Music Arena loader
# --------------------------------------------------------------------------

def _normalise_musicarena_outcome(raw: str) -> str:
    s = str(raw).strip().upper()
    if s == "A":
        return "a_wins"
    if s == "B":
        return "b_wins"
    if s == "BOTH_BAD":
        return "both_bad"
    return "tie"


def _load_musicarena_battles(battle_glob: str, raw_root: Path) -> List[dict]:
    files = sorted(glob.glob(battle_glob, recursive=True))
    out: List[dict] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        rel_a = (obj.get("audio_a") or "").strip()
        rel_b = (obj.get("audio_b") or "").strip()
        if not rel_a or not rel_b:
            continue
        pa = raw_root / rel_a
        pb = raw_root / rel_b
        if not pa.is_file() or not pb.is_file():
            continue
        out.append({
            "uuid": obj.get("battle_uuid", ""),
            "audio_a_abs": pa,
            "audio_b_abs": pb,
            "system_a": obj["system_a"],
            "system_b": obj["system_b"],
            "preference": obj["preference"],
            "stem_a": Path(rel_a).stem,
            "stem_b": Path(rel_b).stem,
            "listen_a": float(obj.get("total_listening_time_a") or 0.0),
            "listen_b": float(obj.get("total_listening_time_b") or 0.0),
            "duration_a": float(obj.get("duration_a") or 0.0),
            "duration_b": float(obj.get("duration_b") or 0.0),
        })
    return out


# --------------------------------------------------------------------------
# Per-dataset orchestration
# --------------------------------------------------------------------------

def _build_musicpref_original_to_downloaded(parquet_dir: Path) -> Dict[str, str]:
    """Map MusicPref's per-pair ``audio`` filename (original name inside the
    parquet) to the sequential downloaded ``MusicPref2025_{i}.wav``.

    ``hf_ingest_smoke.py`` streams the parquets in sorted filename order and
    writes ``MusicPref2025_0.wav``, ``MusicPref2025_1.wav``, ... in the same
    order, so we reproduce that walk here.
    """
    parquets = sorted(parquet_dir.glob("*.parquet"))
    mapping: Dict[str, str] = {}
    idx = 0
    for p in parquets:
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            orig = str(row["audio"]["path"])
            mapping[orig] = f"MusicPref2025_{idx}.wav"
            idx += 1
    return mapping


def _run_musicpref(args: argparse.Namespace, root: Path) -> pd.DataFrame:
    if args.head != "musicality":
        raise SystemExit(
            "musicpref only supports --head musicality (fidelity was removed)."
        )
    csv_path = args.human_preference_csv or (
        root / "raw_hf/MusicPref/human_preference.csv"
    )
    csv_path = Path(csv_path).resolve()
    pairs, enriched, track_system = _load_musicpref(csv_path)

    parquet_dir = Path(args.musicpref_parquet_dir or (
        root / "raw_hf/MusicPref/data"
    )).resolve()
    orig_to_downloaded = _build_musicpref_original_to_downloaded(parquet_dir)
    print(f"[musicpref] parquet->downloaded-wav mapping: {len(orig_to_downloaded)} entries")

    sys_elo = fit_elo_grouped(
        pairs, group_of=lambda t: track_system[t],
        k_factor=args.k_system, n_passes=args.passes,
        n_seeds=args.seeds, seed=args.seed,
    )
    raw = track_score_from_system_elo(
        sys_elo, enriched, k_factor=args.k_track,
    )
    mos = map_to_mos_range(raw, low=args.low, high=args.high, robust=True)

    musicpref_prefix = args.musicpref_audio_subpath.strip("/")
    rows: List[dict] = []
    missing = 0
    for fn, s in sorted(mos.items()):
        mapped_name = orig_to_downloaded.get(fn, "")
        if mapped_name:
            audio_path = f"{musicpref_prefix}/{mapped_name}".replace("//", "/")
        else:
            missing += 1
            audio_path = ""
        rows.append({
            "source": "MusicPref2025_musicality",
            "filename": fn,
            "score_1to5": round(float(s), 6),
            "audio_path": audio_path,
            "system": track_system.get(fn, ""),
            "system_elo": float(sys_elo.get(track_system.get(fn, ""), float("nan"))),
            "score_raw": float(raw[fn]),
        })
    if missing:
        print(f"[musicpref] WARNING: {missing} tracks had no parquet entry "
              f"(audio_path left blank; gen_full_splits.py will drop them).")
    return pd.DataFrame(rows)


def _run_aime(args: argparse.Namespace, root: Path) -> pd.DataFrame:
    if args.head != "music_quality":
        raise SystemExit(
            "aime only supports --head music_quality (text_audio_alignment was removed)."
        )
    survey_csv = args.survey_csv or (
        root / "data/manifests/aime_pairwise_survey.csv"
    )
    mq = _load_aime_survey(Path(survey_csv).resolve())

    track_model: Dict[str, str] = {}
    track_window: Dict[str, Tuple[float, float]] = {}
    for _, r in mq.iterrows():
        ta, tb = r["track_a"], r["track_b"]
        track_model[ta] = r["model_1"]
        track_model[tb] = r["model_2"]
        if ta not in track_window:
            track_window[ta] = (
                _ts_to_sec(r["track_1_begin"]), _ts_to_sec(r["track_1_end"]),
            )
        if tb not in track_window:
            track_window[tb] = (
                _ts_to_sec(r["track_2_begin"]), _ts_to_sec(r["track_2_end"]),
            )
    pairs = [(r["track_a"], r["track_b"], r["outcome"]) for _, r in mq.iterrows()]

    print(
        f"[aime] pairs={len(pairs)}  tracks={len(track_model)}  "
        f"systems={len(set(track_model.values()))}"
    )
    sys_elo = fit_elo_grouped(
        pairs, group_of=lambda t: track_model[t],
        base_rating=args.base_rating, k_factor=args.k_system,
        n_passes=args.passes, n_seeds=args.seeds, seed=args.seed,
    )

    track_wins = {t: 0 for t in track_model}
    track_n = {t: 0 for t in track_model}
    for ta, tb, outcome in pairs:
        track_n[ta] += 1
        track_n[tb] += 1
        if outcome == "a_wins":
            track_wins[ta] += 1
        elif outcome == "b_wins":
            track_wins[tb] += 1

    alpha = args.laplace
    delta_elo: Dict[str, float] = {}
    p_hat: Dict[str, float] = {}
    for t in track_model:
        W, n = track_wins[t], track_n[t]
        p = (W + alpha) / (n + 2.0 * alpha)
        p_hat[t] = p
        delta_elo[t] = 400.0 * float(np.log10(p / (1.0 - p)))

    raw = {t: sys_elo[track_model[t]] + delta_elo[t] for t in track_model}
    mos = map_to_mos_range(raw, low=args.low, high=args.high, robust=True)

    rows: List[dict] = []
    for t, s in sorted(mos.items()):
        beg, end = track_window.get(t, (0.0, 10.0))
        rows.append({
            "source": "AIME2025_music_quality",
            "track_id": t,
            "score_1to5": round(float(s), 6),
            "score_raw": float(raw[t]),
            "delta_elo": float(delta_elo[t]),
            "p_hat": float(p_hat[t]),
            "model": track_model.get(t, ""),
            "system_elo": float(sys_elo.get(track_model[t], float("nan"))),
            "wins": int(track_wins[t]),
            "n": int(track_n[t]),
            "win_rate": float(track_wins[t] / track_n[t]) if track_n[t] else float("nan"),
            "begin_s": float(beg),
            "end_s": float(end),
        })
    return pd.DataFrame(rows)


def _run_musicarena(args: argparse.Namespace, root: Path) -> pd.DataFrame:
    if args.head is not None:
        raise SystemExit("musicarena does not take --head")
    bg = args.battle_glob or str(
        root / "raw_hf/MusicArena/battle_data/**/*.json"
    )
    raw_root = root / "raw_hf/MusicArena"
    battles = _load_musicarena_battles(bg, raw_root)
    print(f"[musicarena] battles with both audios: {len(battles)}")
    if not battles:
        raise SystemExit("No usable Music Arena battles found.")

    pairs: List[Tuple[str, str, str]] = []
    enriched: List[Tuple[str, str, str, str, str]] = []
    track_system: Dict[str, str] = {}
    track_abs: Dict[str, Path] = {}
    track_listen: Dict[str, float] = {}
    track_dur: Dict[str, float] = {}
    for b in battles:
        ta, tb = b["stem_a"], b["stem_b"]
        outcome = _normalise_musicarena_outcome(b["preference"])
        pairs.append((ta, tb, outcome))
        enriched.append((ta, tb, b["system_a"], b["system_b"], outcome))
        track_system[ta] = b["system_a"]
        track_system[tb] = b["system_b"]
        track_abs[ta] = b["audio_a_abs"]
        track_abs[tb] = b["audio_b_abs"]
        track_listen[ta] = b["listen_a"]
        track_listen[tb] = b["listen_b"]
        track_dur[ta] = b["duration_a"]
        track_dur[tb] = b["duration_b"]

    outcome_map = dict(DEFAULT_OUTCOME_MAP)
    outcome_map["both_bad"] = (args.both_bad_score, args.both_bad_score)

    # System-level ELO: use very few passes (default=1) to prevent both_bad
    # drain accumulation.  Each both_bad match drains exactly 2K from the
    # rating pool per pass (constant, independent of expected score), so with
    # passes=80 and 489 both_bad battles all system ELOs collapse to ~-150k.
    # With passes=1 each system has ~437 matches — more than enough signal for
    # a stable relative ordering.  The both_bad signal is kept (not changed to
    # tie) so high-BB-rate systems still rank lower.
    sys_passes = args.passes_system if args.passes_system is not None else 1
    sys_elo = fit_elo_grouped(
        pairs, group_of=lambda t: track_system[t],
        k_factor=args.k_system, n_passes=sys_passes,
        n_seeds=args.seeds, seed=args.seed,
        outcome_map=outcome_map,
    )

    # Context-aware soft target (see build_musicarena_scores.py for the full derivation).
    alpha_ctx = float(np.clip(args.context_blend_alpha, 0.0, 1.0))
    sys_min = float(min(sys_elo.values()))
    sys_max = float(max(sys_elo.values()))
    sys_span = max(sys_max - sys_min, 1e-6)

    result_centered = {
        "a_wins":   (+0.5, -0.5),
        "b_wins":   (-0.5, +0.5),
        "tie":      ( 0.0,  0.0),
        "both_bad": (-1.0, -1.0),
    }

    def context_target_fn(sys_a, sys_b, r_a, r_b, outcome):
        q = ((r_a + r_b) / 2.0 - sys_min) / sys_span
        ctx = float(np.clip(q - 0.5, -0.5, 0.5))
        ra_c, rb_c = result_centered.get(outcome, (0.0, 0.0))
        s_a = 0.5 + alpha_ctx * ra_c + (1.0 - alpha_ctx) * ctx
        s_b = 0.5 + alpha_ctx * rb_c + (1.0 - alpha_ctx) * ctx
        return (s_a, s_b)

    residual = track_score_from_system_elo(
        sys_elo, enriched, k_factor=args.k_track,
        outcome_map=outcome_map, return_adjustment=True,
        target_fn=context_target_fn,
    )

    # System-span compression: rescale system baseline to a fixed span
    # before adding the per-clip residual, so the 7-band system collapse
    # goes away.  Set --system_span 0 to disable.
    if args.system_span > 0.0:
        sys_baseline = map_to_mos_range(
            sys_elo, low=0.0, high=float(args.system_span), robust=True,
        )
    else:
        sys_baseline = {s: float(r) for s, r in sys_elo.items()}

    # Per-system z-score residual: normalise each clip's residual by the
    # system's own residual std before adding the system baseline back.
    # This makes per-clip spreads comparable across systems (e.g. acestep
    # std=0.20 vs elevenlabs std=0.62 no longer compete for the same label
    # range).  Clips with only one appearance keep their raw residual.
    if args.zscore_residual:
        from collections import defaultdict
        sys_resids: Dict[str, List[float]] = defaultdict(list)
        for t, r in residual.items():
            sys_resids[track_system[t]].append(r)
        sys_res_std: Dict[str, float] = {
            s: float(np.std(v)) if len(v) > 1 else 1.0
            for s, v in sys_resids.items()
        }
        raw = {
            t: sys_baseline[track_system[t]]
               + residual[t] / max(sys_res_std[track_system[t]], 1e-6)
            for t in track_system
        }
    else:
        raw = {
            t: sys_baseline[track_system[t]] + residual[t]
            for t in track_system
        }

    mos = map_to_mos_range(raw, low=args.low, high=args.high, robust=True)

    # Min-clips-per-system filter: compute clip counts after listen filter,
    # then exclude systems that fall below the threshold.
    min_clips = int(args.min_clips_per_system)
    max_listen = float(args.max_listen_sec)
    if min_clips > 0:
        from collections import Counter
        sys_clip_counts: Counter = Counter()
        for t in track_system:
            dur = track_dur[t] if track_dur[t] > 0 else float("inf")
            heard = min(track_listen[t], dur)
            if min(heard, max_listen) >= float(args.min_listen_sec):
                sys_clip_counts[track_system[t]] += 1
        dropped_sys = {s for s, n in sys_clip_counts.items() if n < min_clips}
        if dropped_sys:
            print(f"[min-clips-per-system={min_clips}] dropping {len(dropped_sys)} system(s): "
                  f"{sorted(dropped_sys)}")
    else:
        dropped_sys = set()

    rows: List[dict] = []
    for t, s in sorted(mos.items()):
        if track_system[t] in dropped_sys:
            continue
        dur = track_dur[t] if track_dur[t] > 0 else float("inf")
        heard = min(track_listen[t], dur)
        listen_sec_used = float(min(heard, max_listen))
        if listen_sec_used < float(args.min_listen_sec):
            # Snap judgements: drop clips with < min_listen_sec evidence.
            continue
        src_abs = track_abs[t]
        try:
            rel = src_abs.relative_to(_REPO)
            audio_path = str(rel).replace("\\", "/")
        except ValueError:
            audio_path = str(src_abs)
        rows.append({
            "source": "MusicArena2025",
            "audio_path": audio_path,
            "score_1to5": round(float(s), 6),
            "score_raw": float(raw[t]),
            "score_raw_residual": float(residual[t]),
            "system": track_system[t],
            "system_elo": float(sys_elo[track_system[t]]),
            "listen_sec_used": float(listen_sec_used),
            "duration_sec": float(track_dur[t]),
            "id": t,
        })
    if not rows:
        raise SystemExit("All Music Arena clips dropped by --min_listen_sec")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset",
        choices=("musicpref", "aime", "musicarena"),
        required=True,
    )
    ap.add_argument(
        "--head",
        choices=("musicality", "music_quality"),
        default=None,
        help=(
            "Required for musicpref (only 'musicality') and aime (only "
            "'music_quality'); omit for musicarena.  Fidelity and "
            "text-audio-alignment heads were removed."
        ),
    )

    # -- Input paths --------------------------------------------------------
    ap.add_argument("--human-preference-csv", type=Path,
                    help="MusicPref human_preference.csv")
    ap.add_argument("--survey-csv", type=Path,
                    help="AIME merged pairwise survey (w/ begin/end timestamps)")
    ap.add_argument("--battle-glob", type=str,
                    help="Glob for MusicArena battle JSON (quote for **)")

    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=None)

    # -- Elo hyperparams ----------------------------------------------------
    ap.add_argument("--k-system", dest="k_system", type=float, default=24.0)
    ap.add_argument("--k-track", dest="k_track", type=float, default=16.0)
    ap.add_argument("--passes", type=int, default=4,
                    help="Passes for track-level ELO residual fit. "
                         "Kept low (default=4) to reduce per-clip residual variance.")
    ap.add_argument("--passes-system", dest="passes_system", type=int, default=None,
                    help="Passes for system-level ELO fit (musicarena only). "
                         "Defaults to 1 to prevent both_bad drain accumulation.")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--base-rating", dest="base_rating", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low", type=float, default=1.0)
    ap.add_argument("--high", type=float, default=5.0)

    # -- AIME-specific ------------------------------------------------------
    ap.add_argument("--laplace", type=float, default=1.0,
                    help="AIME Laplace smoothing alpha: p_hat=(W+a)/(n+2a).")

    # -- Music Arena-specific ----------------------------------------------
    ap.add_argument("--both-bad-score", dest="both_bad_score",
                    type=float, default=-0.5,
                    help="Target score (both sides) for BOTH_BAD outcomes.")
    ap.add_argument("--context-blend-alpha", dest="context_blend_alpha",
                    type=float, default=0.5,
                    help="Mix rater's hard label (alpha=1) with matchup "
                         "pair-quality context (alpha=0) for the 4-outcome "
                         "soft target.")
    ap.add_argument("--system-span", dest="system_span",
                    type=float, default=50.0,
                    help="Compressed Elo span for the system baseline "
                         "(0 disables compression).")
    ap.add_argument("--max-listen-sec", dest="max_listen_sec",
                    type=float, default=180.0,
                    help="Cap on the rater-aligned audio window written "
                         "to the CSV's listen_sec_used column.")
    ap.add_argument("--min-listen-sec", dest="min_listen_sec",
                    type=float, default=2.0,
                    help="Drop clips whose rater listened < this many seconds.")
    ap.add_argument("--min-clips-per-system", dest="min_clips_per_system",
                    type=int, default=60,
                    help="Drop systems with fewer clips than this after listen "
                         "filtering. Prevents saturated/sparse system labels "
                         "from polluting training.")
    ap.add_argument("--zscore-residual", dest="zscore_residual",
                    action="store_true", default=True,
                    help="Normalise per-clip residual by per-system std before "
                         "adding system baseline, making residuals comparable "
                         "across systems with different score spreads.")
    ap.add_argument("--no-zscore-residual", dest="zscore_residual",
                    action="store_false")

    # -- MusicPref audio_path prefix in the CSV ----------------------------
    ap.add_argument(
        "--musicpref-audio-subpath",
        default="datasets/musicprefs/audio",
        help="Path prefix for musicpref audio_path column "
             "(relative to project root).",
    )
    ap.add_argument(
        "--musicpref-parquet-dir", type=Path, default=None,
        help="Directory of MusicPref parquet files used to map the pair "
             "CSV's original filenames to the downloaded "
             "MusicPref2025_{i}.wav names.  Defaults to "
             "raw_hf/MusicPref/data.",
    )

    args = ap.parse_args()

    root = (args.project_root or _REPO).resolve()

    if args.dataset == "musicpref":
        df = _run_musicpref(args, root)
    elif args.dataset == "aime":
        df = _run_aime(args, root)
    else:
        df = _run_musicarena(args, root)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} ({len(df)} rows)  "
        f"score_1to5: min={df['score_1to5'].min():.2f}  "
        f"max={df['score_1to5'].max():.2f}  "
        f"mean={df['score_1to5'].mean():.2f}"
    )


if __name__ == "__main__":
    main()
