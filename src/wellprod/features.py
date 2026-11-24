"""Per-well features: log-curve summaries over the target zone, petrophysical proxies, and the
planned completion design. One row per well, never one column per depth.

Petrophysics (standard textbook forms, constants fixed in advance, not fitted to the data):

- shale volume from the gamma-ray index: Vsh = (GR - GR_clean) / (GR_shale - GR_clean), clipped
  to [0, 1];
- density porosity: phi_d = (rho_ma - RHOB) / (rho_ma - rho_fl), with rho_ma = 2.65 g/cm3
  (quartz) and rho_fl = 1.0 g/cm3; averaged with neutron porosity when NPHI exists;
- water saturation from Archie: Sw = (Rw / (phi^m * Rt))^(1/n), a = 1, m = n = 2;
- net pay: samples with Vsh < 0.4, phi > 0.06 and Sw < 0.6; hydrocarbon pore thickness
  HCPV = sum over pay of phi * (1 - Sw) * step.

Missing curves give NaN features (LightGBM handles NaN natively); nothing is imputed from
other wells, so a feature row depends on that well's own data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .curves import CANONICAL, CanonicalLogs, coverage
from .synth import RW

GR_CLEAN, GR_SHALE = 25.0, 125.0
RHO_MA, RHO_FL = 2.65, 1.0
VSH_CUT, PHI_CUT, SW_CUT = 0.4, 0.06, 0.6
MIN_COVERAGE = 0.8      # a curve counts as present if it covers >= 80% of the zone

LOG_CURVES = ("GR", "RHOB", "NPHI", "RT", "DT")
COMPLETION = ("lateral_m", "proppant_kg_m", "fluid_m3_m", "stage_spacing_m", "completion_year")
LOCATION = ("x_km", "y_km", "tvd_m")
PETRO = ("vsh_mean", "phi_mean", "sw_mean", "ntg", "net_pay_m", "hcpv_m", "zone_thickness_m")


def vshale(gr: np.ndarray, clean: float = GR_CLEAN, shale: float = GR_SHALE) -> np.ndarray:
    return np.clip((gr - clean) / (shale - clean), 0.0, 1.0)


def porosity(rhob: np.ndarray, nphi: np.ndarray | None) -> np.ndarray:
    phi_d = np.clip((RHO_MA - rhob) / (RHO_MA - RHO_FL), 0.0, 0.45)
    if nphi is None:
        return phi_d
    nphi = np.clip(nphi, 0.0, 0.6)
    both = np.isfinite(phi_d) & np.isfinite(nphi)
    return np.where(both, 0.5 * (phi_d + nphi), phi_d)


def archie_sw(rt: np.ndarray, phi: np.ndarray, rw: float = RW) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = np.sqrt(rw / (np.maximum(phi, 1e-3) ** 2 * rt))
    return np.clip(sw, 0.0, 1.0)


def log_features(logs: CanonicalLogs, top: float, base: float) -> dict[str, float]:
    """Summary statistics and petrophysical proxies over [top, base]."""
    d = logs.depth
    z = (d >= top - 1e-9) & (d <= base + 1e-9)
    step = float(np.median(np.diff(d))) if len(d) > 1 else np.nan
    cov = coverage(logs, top, base)
    out: dict[str, float] = {"zone_thickness_m": float(base - top)}
    present = {}
    for c in LOG_CURVES:
        out[f"cov_{c}"] = cov[c]
        v = logs.curves.get(c)
        ok = v is not None and cov[c] >= MIN_COVERAGE
        present[c] = ok
        vv = v[z] if ok else np.array([np.nan])
        if c == "RT" and ok:
            vv = np.log10(vv)
        name = "log10RT" if c == "RT" else c
        fin = vv[np.isfinite(vv)]
        for q, lab in ((10, "p10"), (50, "p50"), (90, "p90")):
            out[f"{name}_{lab}"] = float(np.percentile(fin, q)) if len(fin) else np.nan
        out[f"{name}_mean"] = float(fin.mean()) if len(fin) else np.nan
        out[f"{name}_std"] = float(fin.std()) if len(fin) > 1 else np.nan
    # petrophysics needs GR, RHOB, RT; NPHI is optional
    if present["GR"] and present["RHOB"] and present["RT"]:
        gr, rhob, rt = (logs.curves[c][z] for c in ("GR", "RHOB", "RT"))
        nphi = logs.curves["NPHI"][z] if present["NPHI"] else None
        vsh = vshale(gr)
        phi = porosity(rhob, nphi)
        sw = archie_sw(rt, phi)
        fin = np.isfinite(vsh) & np.isfinite(phi) & np.isfinite(sw)
        pay = fin & (vsh < VSH_CUT) & (phi > PHI_CUT) & (sw < SW_CUT)
        out["vsh_mean"] = float(vsh[fin].mean()) if fin.any() else np.nan
        out["phi_mean"] = float(phi[fin].mean()) if fin.any() else np.nan
        out["sw_mean"] = float(sw[fin].mean()) if fin.any() else np.nan
        out["ntg"] = float((fin & (vsh < VSH_CUT)).sum() / max(fin.sum(), 1))
        out["net_pay_m"] = float(pay.sum() * step)
        out["hcpv_m"] = float(np.sum((phi * (1 - sw))[pay]) * step)
    else:
        for k in PETRO:
            if k != "zone_thickness_m":
                out[k] = np.nan
    return out


def log_feature_names() -> list[str]:
    names = ["zone_thickness_m"]
    for c in LOG_CURVES:
        names.append(f"cov_{c}")
        name = "log10RT" if c == "RT" else c
        names += [f"{name}_{s}" for s in ("p10", "p50", "p90", "mean", "std")]
    names += [k for k in PETRO if k != "zone_thickness_m"]
    return names


def clean_completion(wells: pd.DataFrame) -> pd.DataFrame:
    """Completion and location columns with impossible values set to NaN (counted by the
    audit). A lateral of 0 m or less is a data-entry error, not a short well."""
    out = wells[list(COMPLETION) + list(LOCATION)].astype(float).copy()
    out.loc[out["lateral_m"] <= 0, "lateral_m"] = np.nan
    return out


def feature_table(wells: pd.DataFrame, logs: dict[str, CanonicalLogs],
                  use_location: bool = True) -> pd.DataFrame:
    """Model features, indexed like `wells`. Post-completion columns are never included."""
    rows = [log_features(logs[w], t, b) for w, t, b in
            zip(wells["well_id"], wells["zone_top_m"], wells["zone_base_m"])]
    lf = pd.DataFrame(rows, index=wells.index)[log_feature_names()]
    comp = clean_completion(wells)
    if not use_location:
        comp = comp.drop(columns=["x_km", "y_km"])
    return pd.concat([lf, comp], axis=1)


__all__ = ["CANONICAL", "feature_table", "log_features", "log_feature_names", "vshale",
           "porosity", "archie_sw", "clean_completion"]
