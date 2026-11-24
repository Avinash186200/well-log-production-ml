"""Metrics for ranking drilling candidates, plus error on the original scale.

- Spearman rank correlation between predicted and actual 12-month oil (ties get average ranks).
- Top-decile capture: the actual oil of the wells the model ranks in its top 10%, divided by
  the actual oil of the true top 10%. 1.0 means the model picked the best wells; picking at
  random gives about mean(y) / mean(true top decile).
- MAE in barrels, and the bias ratio sum(pred) / sum(actual) (1.0 = right on average).
- Quantile calibration: the share of actual values below each predicted quantile, and the
  coverage of the central 80% interval.
- Pad-level bootstrap standard errors: wells on one pad are correlated, so pads (not wells)
  are resampled.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def spearman(pred, y) -> float:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    rp, ry = rankdata(pred), rankdata(y)
    if rp.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rp, ry)[0, 1])


def top_k(pred, frac: float) -> np.ndarray:
    """Indices of the top `frac` of rows by prediction (ties broken by row order)."""
    pred = np.asarray(pred, float)
    k = max(1, int(round(frac * len(pred))))
    return np.argsort(-pred, kind="stable")[:k]


def top_decile_capture(pred, y, frac: float = 0.1) -> float:
    y = np.asarray(y, float)
    k = len(top_k(pred, frac))
    best = np.sort(y)[::-1][:k].sum()
    return float(y[top_k(pred, frac)].sum() / best) if best > 0 else float("nan")


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def bias_ratio(pred, y) -> float:
    return float(np.sum(pred) / np.sum(y))


def ranking_metrics(pred, y) -> dict[str, float]:
    return {"spearman": spearman(pred, y), "top_decile_capture": top_decile_capture(pred, y),
            "mae_bbl": mae(pred, y), "bias_ratio": bias_ratio(pred, y), "n": int(len(y))}


def quantile_calibration(q: dict[float, np.ndarray], y) -> dict[str, float]:
    """q maps a lower-tail probability (0.1, 0.5, 0.9) to predicted quantiles.

    Industry naming: P90 is the value exceeded with 90% probability, i.e. the 10th percentile;
    P10 is the 90th percentile. Keys here are percentiles; labels in reports use P-names.
    """
    y = np.asarray(y, float)
    out = {f"share_below_q{int(a * 100):02d}": float(np.mean(y < v)) for a, v in q.items()}
    lo, hi = q[min(q)], q[max(q)]
    out["coverage_central"] = float(np.mean((y >= lo) & (y <= hi)))
    out["mean_width_bbl"] = float(np.mean(hi - lo))
    return out


def pad_bootstrap_se(pred, y, pads, fn, n_boot: int = 1000, seed: int = 0) -> float:
    """Standard error of fn(pred, y) from resampling whole pads with replacement."""
    pred, y, pads = np.asarray(pred, float), np.asarray(y, float), np.asarray(pads)
    uniq, inv = np.unique(pads, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = fn(pred[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    return float(np.std(vals, ddof=1))
