"""Elo rating utilities for pairwise preference data.

Elo maps a pairwise comparison record to a continuous rating per item by
iteratively updating:

    E_ij = 1 / (1 + 10**((R_j - R_i) / 400))       # expected score for i
    R_i' = R_i + K * (S_ij - E_ij)                  # S_ij in {1, 0.5, 0}

Repeat over all matches, many times, with randomized order to wash out the
path-dependence.

For MusicPref / AIME / Music Arena every audio file appears in only one or a
handful of pairs, so item-level Elo (or Bradley-Terry, or a ReLU hinge)
mathematically collapses into at most three rating bins (win / tie / loss).
Two remedies used by the score-builders:
  1. Run Elo at the *system* level (7 systems, hundreds of matches each --
     the rating signal is very well identified there).
  2. Add a per-track adjustment on top of the system baseline via
     ``track_score_from_system_elo``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Literal, Mapping, Sequence, Tuple

import numpy as np

# "both_bad" is used by Music Arena: both clips are bad. It is NOT the same
# as "tie". Tie means "both equally good"; both_bad means "both equally bad".
# We encode it by giving both sides a sub-0.5 target score, so the Elo
# update `K*(S - E)` pulls both clips' ratings down (roughly equally).
Outcome = Literal["a_wins", "b_wins", "tie", "both_bad"]
Pair = Tuple[str, str, Outcome]

# Default target-score mapping. Override by passing `outcome_map=...` to the
# Elo fitters. For Music Arena the user's requested ordering is
#   win > tie > lose > both_bad
# so both_bad must push the clip *below* what losing would (|adj| > K*E).
# Setting S = -0.5 gives |adj| = K*(1 - E) >= K/2, strictly larger than a
# normal loss's K*E for any realistic E < 1. Elo updates are additive in
# (S - E), so negative target scores are fine; they just represent a
# "worse than losing" outcome.
DEFAULT_OUTCOME_MAP: Mapping[str, Tuple[float, float]] = {
    "a_wins":   (1.0, 0.0),
    "b_wins":   (0.0, 1.0),
    "tie":      (0.5, 0.5),
    "both_bad": (-0.5, -0.5),
}


def _expected(r_i: float, r_j: float) -> float:
    diff = (r_j - r_i) / 400.0
    if diff > 100:
        return 0.0
    elif diff < -100:
        return 1.0
    return 1.0 / (1.0 + 10.0 ** diff)


def _outcome_scores(
    outcome: Outcome,
    outcome_map: Mapping[str, Tuple[float, float]] = DEFAULT_OUTCOME_MAP,
) -> Tuple[float, float]:
    if outcome in outcome_map:
        return outcome_map[outcome]
    if outcome == "a_wins":
        return 1.0, 0.0
    if outcome == "b_wins":
        return 0.0, 1.0
    return 0.5, 0.5


def fit_elo(
    pairs: Sequence[Pair],
    *,
    base_rating: float = 1500.0,
    k_factor: float = 24.0,
    n_passes: int = 40,
    n_seeds: int = 8,
    seed: int = 0,
    init_ratings: "dict[str, float] | None" = None,
    outcome_map: Mapping[str, Tuple[float, float]] = DEFAULT_OUTCOME_MAP,
    verbose: bool = False,
) -> dict[str, float]:
    """Fit item-level Elo by repeated randomized sweeps.

    Averaging across `n_seeds` independent permutation sequences reduces
    order-bias. Returns `{item: avg_rating}`.

    `init_ratings` (optional) seeds each item's starting rating (items not
    in the dict fall back to `base_rating`). This is how we inject a
    model-level prior into a per-track Elo fit (AIME).
    """
    items = sorted({a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    idx = {it: i for i, it in enumerate(items)}
    n = len(items)
    acc = np.zeros(n, dtype=np.float64)

    init_vec = np.full(n, base_rating, dtype=np.float64)
    if init_ratings:
        for it, v in init_ratings.items():
            if it in idx:
                init_vec[idx[it]] = float(v)

    for s in range(n_seeds):
        rng = np.random.default_rng(seed + s)
        r = init_vec.copy()
        order = np.arange(len(pairs))
        for p in range(n_passes):
            rng.shuffle(order)
            for k in order:
                a, b, outcome = pairs[k]
                i, j = idx[a], idx[b]
                s_i, s_j = _outcome_scores(outcome, outcome_map)
                e_i = _expected(r[i], r[j])
                # Note: for zero-sum outcomes (a_wins/b_wins/tie) this is
                # classical Elo; for both_bad (s_i + s_j < 1) both players'
                # ratings drop, which is the desired "absolute badness"
                # signal. Elo is no longer strictly zero-sum but still
                # converges under repeated sweeps.
                r[i] += k_factor * (s_i - e_i)
                r[j] += k_factor * (s_j - (1.0 - e_i))
        acc += r
        if verbose:
            print(f"  seed {s}: rating span = {r.max() - r.min():.1f}")

    acc /= n_seeds
    return {items[i]: float(acc[i]) for i in range(n)}


def fit_elo_grouped(
    pairs: Sequence[Tuple[str, str, Outcome]],
    group_of: Callable[[str], str],
    **kwargs,
) -> dict[str, float]:
    """Fit Elo at the group level by translating each item-pair to a
    group-pair. Handy for system-level ratings.
    """
    g_pairs: list[Pair] = [
        (group_of(a), group_of(b), outcome) for a, b, outcome in pairs
    ]
    return fit_elo(g_pairs, **kwargs)


def track_score_from_system_elo(
    system_elo: dict[str, float],
    pairs: Iterable[Tuple[str, str, str, str, Outcome]],
    *,
    k_factor: float = 16.0,
    outcome_map: Mapping[str, Tuple[float, float]] = DEFAULT_OUTCOME_MAP,
    return_adjustment: bool = False,
    target_fn: "Callable[[str, str, float, float, str], Tuple[float, float]] | None" = None,
) -> dict[str, float]:
    """Attach to each track: its system's Elo + a single-match adjustment.

    `pairs` items are `(track_a, track_b, system_a, system_b, outcome)`.
    The adjustment is one Elo update centered on the system baseline, so
    the final score is continuous-valued per track rather than collapsed
    into {+K, 0, -K} bins.

    If `return_adjustment=True`, returns only the per-track adjustment
    `K*(S - E)` (i.e. the score with the system Elo baseline removed) --
    handy when you want a system-residualised label.

    If `target_fn` is provided, it overrides `outcome_map` on a per-pair
    basis and is called as `target_fn(sys_a, sys_b, sys_elo_a,
    sys_elo_b, outcome) -> (s_a, s_b)`.  Lets callers compute context-
    dependent soft targets (e.g. "soften TIE / BOTH_BAD toward the
    matchup's expected pair quality").
    """
    out: dict[str, float] = {}
    for a, b, sys_a, sys_b, outcome in pairs:
        r_a = system_elo.get(sys_a, 1500.0)
        r_b = system_elo.get(sys_b, 1500.0)
        if target_fn is not None:
            s_a, s_b = target_fn(sys_a, sys_b, r_a, r_b, outcome)
        else:
            s_a, s_b = _outcome_scores(outcome, outcome_map)
        e_a = _expected(r_a, r_b)
        adj_a = k_factor * (s_a - e_a)
        adj_b = k_factor * (s_b - (1.0 - e_a))
        if return_adjustment:
            out[a] = adj_a
            out[b] = adj_b
        else:
            out[a] = r_a + adj_a
            out[b] = r_b + adj_b
    return out


def map_to_mos_range(
    scores: dict[str, float],
    *,
    low: float = 1.0,
    high: float = 5.0,
    robust: bool = True,
    q_lo: float = 0.02,
    q_hi: float = 0.98,
) -> dict[str, float]:
    """Affine-rescale to `[low, high]`. Robust mode uses quantiles to
    avoid a couple of outliers squashing the middle of the distribution.
    """
    vals = np.asarray(list(scores.values()), dtype=np.float64)
    if robust:
        mn = float(np.quantile(vals, q_lo))
        mx = float(np.quantile(vals, q_hi))
    else:
        mn, mx = float(vals.min()), float(vals.max())
    if mx <= mn:
        mid = (low + high) / 2
        return {k: mid for k in scores}
    scale = (high - low) / (mx - mn)
    return {k: float(np.clip(low + scale * (v - mn), low, high)) for k, v in scores.items()}
