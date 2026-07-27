from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")


def safe_pearson_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return float(r), float(rho)


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str = "Pred vs. Human",
    xlabel: str = "Human MOS",
    ylabel: str = "Predicted",
) -> None:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    r, rho = safe_pearson_spearman(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="y=x")
    if len(y_true) >= 2:
        coef = np.polyfit(y_true, y_pred, 1)
        xx = np.array([float(y_true.min()), float(y_true.max())])
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy, "b-", lw=2, label="linear fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nPearson r={r:.4f}  Spearman rho={rho:.4f}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
