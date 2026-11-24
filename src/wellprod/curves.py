"""Mnemonic aliases, unit checks and spike flags: raw LAS curves -> canonical curves.

Vendors and vintages name the same measurement differently (gamma ray as GR, GRC, SGR, ...)
and record it in different units (bulk density in g/cm3 or kg/m3, neutron porosity as a
fraction or in percent). `canonicalise` maps every raw curve to one canonical name, converts
it to one unit, and records what it did, so the audit can count each fix instead of hiding it.

Units are judged from the VALUES as well as the declared unit, because the declared unit is
sometimes wrong (a curve labelled V/V whose median is 25 is in percent).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# canonical name -> raw mnemonics seen for it (upper case). When a file has several aliases
# for one measurement, the one with the most samples wins; ties go to the first listed.
ALIASES: dict[str, tuple[str, ...]] = {
    "GR": ("GR", "GRC", "SGR", "GRD", "GAM", "CGR"),
    "RHOB": ("RHOB", "ZDEN", "DEN", "RHOZ", "DENS"),
    "NPHI": ("NPHI", "TNPH", "NEU", "NPOR", "CNC"),
    "RT": ("RDEP", "ILD", "LLD", "RD", "RT", "AT90", "RESD"),
    "DT": ("DTC", "DT", "AC", "DTCO"),
    "CALI": ("CALI", "CAL", "HCAL"),
}
CANONICAL = tuple(ALIASES)
_LOOKUP = {raw: canon for canon, raws in ALIASES.items() for raw in raws}

# Physically plausible ranges in canonical units; samples outside are flagged as spikes.
# Organic-rich shales can read well above 300 gAPI, so GR is allowed up to 500.
PLAUSIBLE = {"GR": (0.0, 500.0), "RHOB": (1.5, 3.1), "NPHI": (-0.05, 0.8),
             "RT": (0.05, 5000.0), "DT": (35.0, 200.0), "CALI": (4.0, 30.0)}
UNITS = {"GR": "gAPI", "RHOB": "g/cm3", "NPHI": "v/v", "RT": "ohm.m", "DT": "us/ft",
         "CALI": "in"}


def canonical_name(raw: str) -> str | None:
    return _LOOKUP.get(raw.strip().upper())


@dataclass
class CurveFix:
    canonical: str
    raw_mnemonic: str
    raw_unit: str
    conversion: str = "none"         # e.g. "kg/m3 -> g/cm3"
    unit_mismatch: bool = False      # declared unit disagrees with the values
    n_spikes: int = 0                # samples outside PLAUSIBLE after conversion (set to NaN)


@dataclass
class CanonicalLogs:
    depth: np.ndarray
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    fixes: dict[str, CurveFix] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)      # raw mnemonics with no alias
    duplicates: list[str] = field(default_factory=list)   # extra aliases not used
    empty: list[str] = field(default_factory=list)        # curves with no finite sample


def _convert(canon: str, values: np.ndarray, unit: str) -> tuple[np.ndarray, str, bool]:
    """Return (values in canonical units, conversion label, declared-unit mismatch)."""
    u = unit.strip().lower().replace(" ", "")
    med = float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan
    if canon == "RHOB":
        declared_kg = u in ("kg/m3", "k/m3", "kg/m^3")
        by_value_kg = med > 100.0
        if by_value_kg:
            return values / 1000.0, "kg/m3 -> g/cm3", not declared_kg
        return values, "none", declared_kg
    if canon == "NPHI":
        declared_pct = u in ("%", "pu", "p.u.", "percent")
        by_value_pct = med > 1.5
        if by_value_pct:
            return values / 100.0, "percent -> v/v", not declared_pct
        return values, "none", declared_pct
    if canon == "DT":
        declared_si = u in ("us/m", "usec/m")
        by_value_si = med > 230.0
        if by_value_si:
            return values / 3.28084, "us/m -> us/ft", not declared_si
        return values, "none", declared_si
    return values, "none", False


def canonicalise(depth: np.ndarray, raw: dict[str, tuple[str, np.ndarray]]) -> CanonicalLogs:
    """Map raw curves {mnemonic: (unit, values)} to canonical curves in canonical units.

    Spikes (outside PLAUSIBLE after conversion) become NaN and are counted per curve.
    """
    out = CanonicalLogs(depth=np.asarray(depth, dtype=float))
    by_canon: dict[str, list[str]] = {}
    n_finite = {m: int(np.isfinite(np.asarray(v, dtype=float)).sum())
                for m, (_, v) in raw.items()}
    for mnem in raw:
        canon = canonical_name(mnem)
        if n_finite[mnem] == 0:
            out.empty.append(mnem)            # declared but no data (common in exports)
        elif canon is None:
            out.unknown.append(mnem)
        else:
            by_canon.setdefault(canon, []).append(mnem)
    for canon, mnems in by_canon.items():
        # the alias with the most data wins; ties go to the order in ALIASES
        order = ALIASES[canon]
        mnems = sorted(mnems, key=lambda m: (-n_finite[m], order.index(m.strip().upper())))
        chosen = mnems[0]
        out.duplicates += mnems[1:]
        unit, vals = raw[chosen]
        vals = np.asarray(vals, dtype=float).copy()
        vals, conv, mismatch = _convert(canon, vals, unit)
        lo, hi = PLAUSIBLE[canon]
        bad = np.isfinite(vals) & ((vals < lo) | (vals > hi))
        vals[bad] = np.nan
        out.curves[canon] = vals
        out.fixes[canon] = CurveFix(canon, chosen, unit, conv, mismatch, int(bad.sum()))
    return out


def from_las(las) -> CanonicalLogs:
    """Canonical curves from a parsed LasFile (first curve = depth)."""
    depth = las.data[:, 0]
    raw = {c.mnemonic: (c.unit, las.data[:, i]) for i, c in enumerate(las.curves) if i > 0}
    return canonicalise(depth, raw)


def coverage(logs: CanonicalLogs, top: float, base: float) -> dict[str, float]:
    """Share of the [top, base] interval with a finite sample, per canonical curve.

    The denominator is the interval length divided by the median depth step, so a log that
    starts below `top` (or stops above `base`) loses coverage too.
    """
    d = logs.depth
    if len(d) < 2:
        return {c: 0.0 for c in CANONICAL}
    step = float(np.median(np.diff(d)))
    n_expected = max(int(round((base - top) / step)) + 1, 1)
    inside = (d >= top - 1e-9) & (d <= base + 1e-9)
    res = {}
    for c in CANONICAL:
        v = logs.curves.get(c)
        res[c] = 0.0 if v is None else min(float(np.isfinite(v[inside]).sum()) / n_expected, 1.0)
    return res
