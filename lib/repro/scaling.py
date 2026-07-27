from typing import Callable, Dict

import numpy as np

DEFAULT_MOS_LOW, DEFAULT_MOS_HIGH = 1.0, 5.0


def rescale_direct(pred_val: np.ndarray, human_val: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    p_min, p_max = float(np.nanmin(pred_val)), float(np.nanmax(pred_val))
    if p_max <= p_min:
        return np.full_like(pred_test, np.nanmean(human_val))
    scale = (DEFAULT_MOS_HIGH - DEFAULT_MOS_LOW) / (p_max - p_min)
    out = DEFAULT_MOS_LOW + scale * (pred_test - p_min)
    return np.clip(out, DEFAULT_MOS_LOW, DEFAULT_MOS_HIGH)


def rescale_center_align(pred_val: np.ndarray, human_val: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    return pred_test + float(np.nanmean(human_val) - np.nanmean(pred_val))


def rescale_mean_std_align(pred_val: np.ndarray, human_val: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    mu_p = np.nanmean(pred_val)
    mu_h = np.nanmean(human_val)
    std_p = np.nanstd(pred_val)
    std_h = np.nanstd(human_val)
    if std_p < 1e-12:
        return np.full_like(pred_test, mu_h)
    return (pred_test - mu_p) * (std_h / std_p) + mu_h


def rescale_max_min_align(pred_val: np.ndarray, human_val: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    p_min, p_max = float(np.nanmin(pred_val)), float(np.nanmax(pred_val))
    h_min, h_max = float(np.nanmin(human_val)), float(np.nanmax(human_val))
    if p_max <= p_min:
        return np.full_like(pred_test, np.nanmean(human_val))
    out = h_min + (h_max - h_min) * (pred_test - p_min) / (p_max - p_min)
    return np.clip(out, DEFAULT_MOS_LOW, DEFAULT_MOS_HIGH)


def rescale_linear_regression(pred_val: np.ndarray, human_val: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    mask = np.isfinite(pred_val) & np.isfinite(human_val)
    if mask.sum() < 2:
        return np.full_like(pred_test, np.nanmean(human_val))
    coef = np.polyfit(pred_val[mask], human_val[mask], 1)
    out = coef[0] * pred_test + coef[1]
    return np.clip(out, DEFAULT_MOS_LOW, DEFAULT_MOS_HIGH)


def rescale_quantile_map(pred_val: np.ndarray, human_val: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    pred_val = np.asarray(pred_val, dtype=float).flatten()
    human_val = np.asarray(human_val, dtype=float).flatten()
    pred_test = np.asarray(pred_test, dtype=float).flatten()
    m_val = np.isfinite(pred_val) & np.isfinite(human_val)
    m_test = np.isfinite(pred_test)
    if m_val.sum() < 2:
        return np.clip(np.full_like(pred_test, np.nanmean(human_val)), DEFAULT_MOS_LOW, DEFAULT_MOS_HIGH)
    pv = np.sort(pred_val[m_val])
    hv = np.sort(human_val[m_val])
    pv_u, counts = np.unique(pv, return_counts=True)
    if len(pv_u) == 1:
        return np.clip(np.full_like(pred_test, float(np.nanmean(hv))), DEFAULT_MOS_LOW, DEFAULT_MOS_HIGH)
    cum = np.cumsum(counts).astype(float)
    p_pv = (cum - 0.5 * counts) / float(cum[-1])
    p_hv = (np.arange(len(hv), dtype=float) + 0.5) / float(len(hv))
    out = np.full_like(pred_test, np.nan, dtype=float)
    pct = np.interp(pred_test[m_test], pv_u, p_pv, left=p_pv[0], right=p_pv[-1])
    out[m_test] = np.interp(pct, p_hv, hv, left=hv[0], right=hv[-1])
    out[~m_test] = float(np.nanmean(hv))
    return np.clip(out, DEFAULT_MOS_LOW, DEFAULT_MOS_HIGH)


RESCALE_METHODS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "direct": rescale_direct,
    "center_align": rescale_center_align,
    "mean_std_align": rescale_mean_std_align,
    "max_min_align": rescale_max_min_align,
    "linear_regression": rescale_linear_regression,
    "quantile_map": rescale_quantile_map,
}


def apply_rescale(method: str, pred_val: np.ndarray, human_val: np.ndarray, pred_test: np.ndarray) -> np.ndarray:
    if method not in RESCALE_METHODS:
        raise ValueError(f"Unknown rescale method: {method}")
    return RESCALE_METHODS[method](pred_val, human_val, pred_test)
