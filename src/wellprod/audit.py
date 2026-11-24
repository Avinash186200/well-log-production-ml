"""Data-readiness audit: is there enough clean data to try, and what will bite?

Run before any model. It answers, with counts:

- header: how many values are missing or impossible per column;
- logs: which curves exist under which names and units, how many wells survive each extra
  curve requirement, how many unit fixes and spikes there were;
- target: zeros, skew, outliers;
- leakage: columns recorded after completion (from the data dictionary) and columns whose
  rank correlation with the target is suspiciously high (from the data);
- spatial clustering: wells per pad, nearest-neighbour distances, how fast the target's
  correlation between wells decays with distance, and Moran's I.

The same log-inventory code runs on the synthetic basin and on the public FORCE 2020 logs.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .curves import CANONICAL, CanonicalLogs, coverage
from .features import LOG_CURVES, MIN_COVERAGE

SURVIVAL_ORDER = ("GR", "RHOB", "RT", "NPHI", "DT")
LEAK_RHO = 0.9
DIST_BANDS_KM = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)


# ---------------------------------------------------------------- log inventory --------------
def inventory_row(well_id: str, logs: CanonicalLogs, top: float, base: float) -> dict:
    """One row per well describing its canonical curves over [top, base]."""
    cov = coverage(logs, top, base)
    d = logs.depth
    row = {"well_id": well_id, "n_samples": int(len(d)),
           "step_m": float(np.median(np.diff(d))) if len(d) > 1 else np.nan,
           "top_m": float(top), "base_m": float(base),
           "unknown_curves": ";".join(sorted(logs.unknown)),
           "duplicate_aliases": ";".join(sorted(logs.duplicates))}
    for c in CANONICAL:
        fx = logs.fixes.get(c)
        row[f"cov_{c}"] = round(cov[c], 4)
        row[f"raw_{c}"] = fx.raw_mnemonic if fx else ""
        row[f"unit_{c}"] = fx.raw_unit if fx else ""
        row[f"conv_{c}"] = fx.conversion if fx else ""
        row[f"mismatch_{c}"] = bool(fx.unit_mismatch) if fx else False
        row[f"spikes_{c}"] = int(fx.n_spikes) if fx else 0
    return row


def summarise_inventory(inv: pd.DataFrame, curves=LOG_CURVES, survival=SURVIVAL_ORDER,
                        min_coverage: float = MIN_COVERAGE) -> dict:
    n = len(inv)
    out: dict = {"wells": int(n), "min_coverage": min_coverage, "curves": {}}
    for c in curves:
        has = inv[f"raw_{c}"] != ""
        present = inv[f"cov_{c}"] >= min_coverage
        out["curves"][c] = {
            "wells_with_curve": int(has.sum()),
            "wells_covering_zone": int(present.sum()),
            "share_covering_zone": round(float(present.mean()), 4) if n else 0.0,
            "wells_partial": int((has & ~present).sum()),
            "aliases": dict(sorted(Counter(inv.loc[has, f"raw_{c}"]).items())),
            "units": dict(sorted(Counter(inv.loc[has, f"unit_{c}"]).items())),
            "conversions": dict(sorted(Counter(
                inv.loc[has & (inv[f"conv_{c}"] != "none"), f"conv_{c}"]).items())),
            "declared_unit_wrong": int(inv[f"mismatch_{c}"].sum()),
            "wells_with_spikes": int((inv[f"spikes_{c}"] > 0).sum()),
            "spike_samples": int(inv[f"spikes_{c}"].sum()),
        }
    chain, mask = [], np.ones(n, dtype=bool)
    for c in survival:
        mask = mask & (inv[f"cov_{c}"] >= min_coverage).to_numpy()
        chain.append({"requires": "+".join(survival[:survival.index(c) + 1]),
                      "wells": int(mask.sum())})
    out["survival"] = chain
    unknown = Counter(m for s in inv["unknown_curves"] for m in s.split(";") if m)
    out["unknown_curves"] = dict(sorted(unknown.items()))
    return out


# ---------------------------------------------------------------- spatial --------------------
def pair_distances(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.triu_indices(len(xy), k=1)
    d = np.sqrt(((xy[i] - xy[j]) ** 2).sum(-1))
    return i, j, d


def correlogram(xy: np.ndarray, v: np.ndarray, bands=DIST_BANDS_KM) -> list[dict]:
    """Mean product of standardised values over well pairs in each distance band: a simple
    estimate of how correlated two wells' targets are at that distance."""
    z = (v - v.mean()) / v.std()
    i, j, d = pair_distances(xy)
    out = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = (d >= lo) & (d < hi)
        out.append({"from_km": lo, "to_km": hi, "pairs": int(m.sum()),
                    "correlation": round(float(np.mean(z[i[m]] * z[j[m]])), 4)
                    if m.any() else None})
    return out


def morans_i(xy: np.ndarray, v: np.ndarray, k: int = 5, n_perm: int = 199,
             seed: int = 0) -> dict:
    """Moran's I with row-standardised k-nearest-neighbour weights, and a one-sided
    permutation p-value (share of random relabellings with I at least as large)."""
    n = len(v)
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1, kind="stable")[:, :k]

    def stat(x):
        z = x - x.mean()
        return float((z * z[nn].mean(axis=1)).sum() / (z * z).sum())

    obs = stat(v)
    rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(v)) for _ in range(n_perm)])
    return {"k": k, "I": round(obs, 4), "expected_if_random": round(-1.0 / (n - 1), 4),
            "permutations": n_perm,
            "p_value": round(float((1 + (perm >= obs).sum()) / (n_perm + 1)), 4)}


def nearest_neighbour_km(xy: np.ndarray) -> np.ndarray:
    d = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.sqrt(d.min(axis=1))


# ---------------------------------------------------------------- full audit -----------------
def _q(s: pd.Series, qs=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)) -> dict:
    return {f"q{int(q * 100):02d}": round(float(s.quantile(q)), 2) for q in qs}


def audit(wells: pd.DataFrame, logs: dict[str, CanonicalLogs],
          dictionary: dict[str, tuple[str, str]], target: str = "cum12_oil_bbl") -> dict:
    inv = pd.DataFrame([inventory_row(w, logs[w], t, b) for w, t, b in
                        zip(wells["well_id"], wells["zone_top_m"], wells["zone_base_m"])])
    res: dict = {"wells": int(len(wells)), "pads": int(wells["pad_id"].nunique()),
                 "fields": int(wells["field_id"].nunique())}

    # header
    cols = [c for c in wells.columns if c in dictionary]
    res["missing_share"] = {c: round(float(wells[c].isna().mean()), 4) for c in cols
                            if wells[c].isna().any()}
    res["impossible_values"] = {"lateral_m <= 0": int((wells["lateral_m"] <= 0).sum())}
    comp_out = {}
    for c in ("lateral_m", "proppant_kg_m", "fluid_m3_m", "stage_spacing_m"):
        v = wells[c].where(wells[c] > 0)
        med = v.median()
        mad = (v - med).abs().median() * 1.4826
        comp_out[c] = int(((v - med).abs() / mad > 4).sum())
    res["completion_outliers_robust_z_gt_4"] = comp_out

    # logs
    res["logs"] = summarise_inventory(inv)

    # target
    y = wells[target].astype(float)
    pos = y[y > 0]
    ly = np.log(pos)
    rz = (ly - ly.median()).abs() / ((ly - ly.median()).abs().median() * 1.4826)
    res["target"] = {"n": int(len(y)), "zero_share": round(float((y == 0).mean()), 4),
                     "quantiles_bbl": _q(y), "mean_bbl": round(float(y.mean()), 1),
                     "skew": round(float(y.skew()), 3),
                     "log_outliers_robust_z_gt_3_5": int((rz > 3.5).sum())}

    # leakage
    by_dict = sorted(c for c in cols if dictionary[c][0] == "post-completion")
    rhos = {}
    for c in cols:
        if c == target or dictionary[c][0] in ("id", "target"):
            continue
        s = wells[[c, target]].dropna()
        if len(s) > 10 and s[c].nunique() > 1:
            rhos[c] = round(float(s[c].rank().corr(s[target].rank())), 4)
    by_data = sorted(c for c, r in rhos.items() if abs(r) > LEAK_RHO)
    res["leakage"] = {"post_completion_by_dictionary": by_dict,
                      "rank_corr_with_target": dict(sorted(rhos.items())),
                      "flag_threshold_abs_rho": LEAK_RHO, "flagged_by_data": by_data,
                      "flagged_by_data_not_dictionary": sorted(set(by_data) - set(by_dict)),
                      "excluded_from_features": by_dict}

    # spatial
    xy = wells[["x_km", "y_km"]].to_numpy(float)
    per_pad = wells.groupby("pad_id").size()
    nn = nearest_neighbour_km(xy)
    # ranks, not log1p: the zeros would dominate a log-scale correlation
    ry = y.rank().to_numpy(float)
    res["spatial"] = {"wells_per_pad": {"min": int(per_pad.min()),
                                        "median": float(per_pad.median()),
                                        "max": int(per_pad.max())},
                      "nearest_neighbour_km": {"median": round(float(np.median(nn)), 3),
                                               "q90": round(float(np.quantile(nn, 0.9)), 3)},
                      "correlogram_rank_target": correlogram(xy, ry),
                      "morans_i_rank_target": morans_i(xy, ry)}
    return {"summary": res, "inventory": inv}
