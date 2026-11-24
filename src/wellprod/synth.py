"""Synthetic basin: spatially correlated geology, pads of horizontal wells, raw LAS logs with
real-world defects, completion design, and a skewed, zero-inflated 12-month oil target.

The generator knows the truth, so the evaluation can be checked against it:

- geology (porosity, net-to-gross, water saturation, zone thickness, depth) are smooth random
  fields over the map, so nearby wells look alike;
- a fracture-intensity field that changes production but is NOT visible in the logs, plus a
  pad effect shared by wells on one pad. Both make neighbours' outcomes informative, which is
  exactly what makes random K-fold CV optimistic;
- pads nearer a field centre were drilled first, so the last pads (the held-out set) are newer
  and further out, like real step-out drilling;
- logs come as LAS text with vendor-specific mnemonics, wrong or mixed units, spikes, NULL
  gaps, missing curves on older wells and partial coverage of the target zone;
- completion design trends up with year, and some fields are missing or mis-entered;
- post-completion columns (first-30-day rate, 6-month cumulative, ...) are present on purpose,
  because they leak the target; the audit must flag them.

Every random draw comes from one `numpy.random.Generator` seeded by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .las import write_las

# Each data column, when it is known, and what it is. The audit uses `available`.
DATA_DICTIONARY: dict[str, tuple[str, str]] = {
    "well_id": ("id", "well identifier"),
    "pad_id": ("id", "drilling pad (wells on one pad share a surface location)"),
    "field_id": ("id", "field"),
    "x_km": ("planned", "easting of the lateral midpoint, km"),
    "y_km": ("planned", "northing of the lateral midpoint, km"),
    "tvd_m": ("planned", "true vertical depth of the target zone, m"),
    "zone_top_m": ("planned", "top of the target zone in the pilot log, m MD"),
    "zone_base_m": ("planned", "base of the target zone in the pilot log, m MD"),
    "completion_year": ("planned", "year the well was completed"),
    "lateral_m": ("planned", "completed lateral length, m"),
    "proppant_kg_m": ("planned", "proppant per metre of lateral, kg/m"),
    "fluid_m3_m": ("planned", "frac fluid per metre of lateral, m3/m"),
    "stage_spacing_m": ("planned", "distance between frac stages, m"),
    "ip30_bopd": ("post-completion", "average oil rate over the first 30 producing days"),
    "peak_month_oil_bbl": ("post-completion", "highest monthly oil volume"),
    "cum6_oil_bbl": ("post-completion", "first-6-month cumulative oil"),
    "water_cut_3m": ("post-completion", "water cut over the first 3 months"),
    "cum12_oil_bbl": ("target", "first-12-month cumulative oil, bbl"),
}

# Each vendor's names and units for (GR, RHOB, NPHI, RT, DT) plus extra curves it records.
VENDORS = (
    {"GR": ("GR", "GAPI"), "RHOB": ("RHOB", "G/C3"), "NPHI": ("NPHI", "V/V"),
     "RT": ("RDEP", "OHMM"), "DT": ("DTC", "US/F"), "extra": ("CALI", "IN")},
    {"GR": ("GRC", "GAPI"), "RHOB": ("ZDEN", "G/CM3"), "NPHI": ("TNPH", "V/V"),
     "RT": ("ILD", "OHMM"), "DT": ("DT", "US/F"), "extra": ("SP", "MV")},
    {"GR": ("SGR", "API"), "RHOB": ("DEN", "G/CC"), "NPHI": ("NEU", "PU"),
     "RT": ("LLD", "OHM.M"), "DT": ("AC", "US/FT"), "extra": ("PEF", "B/E")},
)

RW = 0.05             # formation water resistivity, ohm.m (also used by the feature code)
HCPV_REF = 0.8        # reference hydrocarbon pore thickness, m, for the production law


@dataclass(frozen=True)
class SynthConfig:
    n_fields: int = 9
    pads_per_field: tuple[int, int] = (12, 18)
    wells_per_pad: tuple[int, int] = (3, 8)
    area_km: tuple[float, float] = (120.0, 80.0)
    min_field_gap_km: float = 24.0
    field_radius_km: float = 9.0
    length_scale_km: float = 7.0
    first_year: int = 2014
    last_year: int = 2024
    holdout_from_year: int = 2022
    step_m: float = 0.5
    log_margin_m: float = 15.0


def _grf(rng: np.random.Generator, length_km: float, n_feat: int = 300):
    """A smooth stationary random field on the map with unit variance (random Fourier features
    of a squared-exponential kernel with the given length scale)."""
    w = rng.normal(0.0, 1.0 / length_km, size=(n_feat, 2))
    b = rng.uniform(0.0, 2 * np.pi, size=n_feat)
    a = rng.normal(0.0, 1.0, size=n_feat)

    def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = np.outer(x, w[:, 0]) + np.outer(y, w[:, 1]) + b
        return np.sqrt(2.0 / n_feat) * np.cos(z) @ a

    return f


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _field_centres(rng, cfg: SynthConfig) -> np.ndarray:
    r = cfg.field_radius_km
    out: list[tuple[float, float]] = []
    while len(out) < cfg.n_fields:
        c = (rng.uniform(r, cfg.area_km[0] - r), rng.uniform(r, cfg.area_km[1] - r))
        if all(np.hypot(c[0] - o[0], c[1] - o[1]) >= cfg.min_field_gap_km for o in out):
            out.append(c)
    return np.array(out)


def _beds(rng, n: int, ntg: float, mean_bed: int) -> np.ndarray:
    """Boolean sand flags for n samples, in beds of random thickness."""
    sand = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        k = max(1, int(rng.exponential(mean_bed)))
        sand[i:i + k] = rng.random() < ntg
        i += k
    return sand


@dataclass
class Basin:
    wells: pd.DataFrame          # one row per well: header, completion, leakage cols, target
    las: dict[str, str]          # well_id -> LAS text of the pilot log
    truth: pd.DataFrame          # well_id -> hidden truth (expected value, hcpv, fracture)


def simulate_basin(seed: int, cfg: SynthConfig | None = None) -> Basin:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    g = {k: _grf(rng, cfg.length_scale_km) for k in
         ("phi", "ntg", "sw", "thick", "depth", "frac")}
    centres = _field_centres(rng, cfg)
    span = cfg.last_year - cfg.first_year

    rows, truth_rows, las = [], [], {}
    wid = 0
    for f_idx, (cx, cy) in enumerate(centres):
        vendor_bias = rng.integers(0, len(VENDORS))
        prop_style = rng.normal(0.0, 0.12)            # some fields frac harder than others
        n_pads = rng.integers(cfg.pads_per_field[0], cfg.pads_per_field[1] + 1)
        for p_idx in range(n_pads):
            rr = cfg.field_radius_km * np.sqrt(rng.random())
            th = rng.uniform(0, 2 * np.pi)
            px, py = cx + rr * np.cos(th), cy + rr * np.sin(th)
            # (rr / R)^2 is uniform on [0, 1]: years spread evenly, outer pads drilled later
            year = int(np.clip(round(cfg.first_year + span * (rr / cfg.field_radius_km) ** 2
                                     + rng.normal(0, 1.0)), cfg.first_year, cfg.last_year))
            pad_id = f"F{f_idx + 1:02d}-P{p_idx + 1:02d}"
            pad_eff = rng.normal(0.0, 0.2)
            vendor = VENDORS[vendor_bias if rng.random() < 0.7 else rng.integers(0, 3)]
            azi = rng.uniform(0, np.pi)
            n_w = rng.integers(cfg.wells_per_pad[0], cfg.wells_per_pad[1] + 1)
            for k in range(n_w):
                wid += 1
                well_id = f"W{wid:04d}"
                # lateral midpoints fan out from the pad, ~0.3 km apart, ~1 km along azimuth
                off = (k - (n_w - 1) / 2) * 0.3
                x = px + 1.0 * np.cos(azi) - off * np.sin(azi)
                y = py + 1.0 * np.sin(azi) + off * np.cos(azi)
                rec, tr, text = _one_well(rng, cfg, g, well_id, pad_id, f_idx, x, y, year,
                                          pad_eff, prop_style, vendor)
                rows.append(rec)
                truth_rows.append(tr)
                las[well_id] = text
    wells = pd.DataFrame(rows)
    wells["split"] = np.where(wells["completion_year"] >= cfg.holdout_from_year, "holdout", "dev")
    return Basin(wells=wells, las=las, truth=pd.DataFrame(truth_rows))


def _one_well(rng, cfg, g, well_id, pad_id, f_idx, x, y, year, pad_eff, prop_style, vendor):
    xa, ya = np.array([x]), np.array([y])
    phi_m = float(np.clip(0.10 + 0.03 * g["phi"](xa, ya)[0], 0.04, 0.20))
    ntg = float(_sigmoid(0.2 + 0.9 * g["ntg"](xa, ya)[0]))
    sw_m = float(np.clip(0.40 - 0.10 * g["sw"](xa, ya)[0], 0.15, 0.85))
    thick = float(np.clip(30.0 + 8.0 * g["thick"](xa, ya)[0], 12.0, 55.0))
    tvd = float(2600.0 + 250.0 * g["depth"](xa, ya)[0] + 3.0 * x)
    frac = float(g["frac"](xa, ya)[0])

    # --- pilot log: the zone plus a margin of shale above and below -------------------------
    step = cfg.step_m
    top = round(tvd + rng.normal(0, 5), 1)
    base = round(top + thick, 1)
    depth = np.arange(top - cfg.log_margin_m, base + cfg.log_margin_m + 1e-9, step)
    n = len(depth)
    in_zone = (depth >= top) & (depth <= base)
    sand = _beds(rng, n, ntg, mean_bed=4) & in_zone
    vsh = np.where(sand, rng.uniform(0.0, 0.25, n), rng.uniform(0.6, 1.0, n))
    phi = np.where(sand, np.clip(phi_m * rng.normal(1.0, 0.15, n), 0.01, 0.3),
                   rng.uniform(0.02, 0.05, n))
    sw = np.where(sand, np.clip(sw_m * rng.normal(1.0, 0.1, n), 0.08, 1.0), 1.0)
    # true hydrocarbon pore thickness in the zone (m): what the oil-in-place depends on
    hcpv = float(np.sum((phi * (1 - sw))[in_zone & sand]) * step)

    gr = 25.0 + 100.0 * vsh + rng.normal(0, 4, n)
    rho_ma = np.where(sand, 2.65, 2.60)
    rho_fl = sw * 1.0 + (1 - sw) * 0.8
    rhob = phi * rho_fl + (1 - phi) * rho_ma + rng.normal(0, 0.015, n)
    nphi = phi + 0.25 * vsh + rng.normal(0, 0.015, n)
    rt_sand = RW / (phi ** 2 * np.maximum(sw, 0.05) ** 2)
    rt = np.where(sand, rt_sand, rng.uniform(2.0, 4.0, n)) * np.exp(rng.normal(0, 0.1, n))
    rt = np.clip(rt, 0.2, 2000.0)
    dt = 55.0 + 180.0 * phi + 25.0 * vsh + rng.normal(0, 2, n)
    extra = rng.normal(0, 1, n).cumsum() * 0.2 + 10.0
    curves = {"GR": gr, "RHOB": rhob, "NPHI": nphi, "RT": rt, "DT": dt}

    # --- defects ------------------------------------------------------------------------------
    units = {k: vendor[k][1] for k in curves}
    if year < 2017 and rng.random() < 0.5:
        curves.pop("NPHI")                                  # older wells: no neutron log
    if rng.random() < 0.6:
        curves.pop("DT")                                    # sonic run on a minority of wells
    if rng.random() < 0.04:
        curves["RHOB"] = curves["RHOB"] * 1000.0             # density in kg/m3 ...
        units["RHOB"] = "K/M3" if rng.random() < 0.5 else units["RHOB"]   # ... label not always
    if "NPHI" in curves and units["NPHI"] != "PU" and rng.random() < 0.05:
        curves["NPHI"] = curves["NPHI"] * 100.0              # percent, still labelled V/V
    if units["NPHI"] == "PU" and "NPHI" in curves:
        curves["NPHI"] = curves["NPHI"] * 100.0              # vendor 3 records percent properly
    if "DT" in curves and rng.random() < 0.1:
        curves["DT"] = curves["DT"] * 3.28084
        units["DT"] = "US/M"
    if rng.random() < 0.2:                                   # spikes on a few samples
        for k in list(curves):
            m = rng.random(n) < 0.01
            curves[k] = np.where(m, curves[k] * rng.choice([0.2, 5.0], n), curves[k])
    if rng.random() < 0.05:                                  # a NULL gap in density
        i0 = rng.integers(0, max(n - 30, 1))
        curves["RHOB"] = curves["RHOB"].copy()
        curves["RHOB"][i0:i0 + rng.integers(10, 30)] = np.nan
    start = 0
    if rng.random() < 0.05:                                  # log starts inside the zone
        start = int(np.searchsorted(depth, top + rng.uniform(5, 15)))

    cols = [("DEPT", "M", "Measured depth")]
    data = [depth]
    for k, v in curves.items():
        cols.append((vendor[k][0], units[k], k))
        data.append(v)
    cols.append((vendor["extra"][0], vendor["extra"][1], "auxiliary"))
    data.append(extra)
    arr = np.column_stack(data)[start:]
    text = write_las({"WELL": ("", well_id), "FLD": ("", f"F{f_idx + 1:02d}"),
                      "XKM": ("KM", f"{x:.3f}"), "YKM": ("KM", f"{y:.3f}")}, cols, arr)

    # --- completion design ----------------------------------------------------------------------
    t = year - cfg.first_year
    lateral = float(np.clip(rng.normal(1800 + 60 * t, 350), 700, 3400))
    prop = float(np.clip(rng.normal((900 + 70 * t) * np.exp(prop_style), 220), 300, 2400))
    fluid = float(prop / 120.0 * rng.normal(1.0, 0.15))
    stage = float(np.clip(rng.normal(70 - 3 * t, 10), 30, 120))

    # --- production law (truth) -----------------------------------------------------------------
    lp = np.log(prop / 1200.0)
    log_mu = (11.0 + 0.8 * np.log(max(hcpv, 0.05) / HCPV_REF) + 0.7 * np.log(lateral / 2000.0)
              + 0.35 * lp - 0.25 * lp ** 2 - 0.2 * lp * (phi_m - 0.10) / 0.03
              - 0.002 * (stage - 60) + 0.3 * frac + pad_eff)
    mu = float(np.exp(log_mu))
    quality = np.log(max(hcpv, 0.05) / HCPV_REF)
    p_zero = float(_sigmoid(-3.2 - 1.2 * quality)) + 0.02
    produced = rng.random() >= p_zero
    shape = 8.0                                              # Gamma noise, CV = 1/sqrt(8)
    cum12 = float(rng.gamma(shape, mu / shape)) if produced else 0.0
    expected = mu * (1.0 - p_zero)

    rec = {"well_id": well_id, "pad_id": pad_id, "field_id": f"F{f_idx + 1:02d}",
           "x_km": round(x, 3), "y_km": round(y, 3), "tvd_m": round(tvd, 1),
           "zone_top_m": top, "zone_base_m": base, "completion_year": year,
           "lateral_m": round(lateral, 1), "proppant_kg_m": round(prop, 1),
           "fluid_m3_m": round(fluid, 2), "stage_spacing_m": round(stage, 1)}
    # missing and mis-entered completion data
    if rng.random() < 0.05:
        rec["proppant_kg_m"] = np.nan
    if rng.random() < 0.15:
        rec["fluid_m3_m"] = np.nan
    if rng.random() < 0.02:
        rec["tvd_m"] = np.nan
    if rng.random() < 0.01:
        rec["lateral_m"] = 0.0
    # post-completion columns (leak the target)
    rec["ip30_bopd"] = round(cum12 * rng.uniform(0.15, 0.2) / 30.0, 1)
    rec["peak_month_oil_bbl"] = round(cum12 * rng.normal(0.16, 0.02), 0)
    rec["cum6_oil_bbl"] = round(cum12 * rng.normal(0.62, 0.03), 0)
    rec["water_cut_3m"] = round(float(np.clip(sw_m + rng.normal(0, 0.08), 0, 1)), 3) \
        if produced else np.nan
    rec["cum12_oil_bbl"] = round(cum12, 0)
    truth = {"well_id": well_id, "expected_cum12": expected, "hcpv_m": hcpv,
             "fracture": frac, "p_zero": p_zero}
    return rec, truth, text
