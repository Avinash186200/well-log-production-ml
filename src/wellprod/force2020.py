"""The public FORCE 2020 well-log dataset: download with a checksum, audit the real logs.

Source: Bormann, Peter; Aursand, Peder; Dilib, Fahad; Manral, Surrender; Dischington, Peter
(2020). "FORCE 2020 Well well log and lithofacies dataset for machine learning competition"
(v1). Zenodo. https://doi.org/10.5281/zenodo.4351156. Licence CC BY 4.0. Contains data under
the Norwegian licence for Open Government data (NLOD) 2.0, distributed by the Norwegian
government.
118 wells from the Norwegian North Sea, LAS 2.0, with interpreted lithology per sample.

It has logs and lithology, but NO production data, so it is used here for the parts that
need real logs: the data-readiness audit (curve coverage, aliases, units, spikes, spacing)
and a check of the gamma-ray sand flag against the interpreted lithology. The production
model runs on the synthetic basin.

`summarise_zip` reads the downloaded zip (about 170 MB; it is never committed) and writes one
row per well to `samples/force2020_wells.csv` (small, committed, CC BY 4.0 attribution in
`samples/NOTICE.txt`). `summary_from_sample` turns that table into `results/force2020.json`,
so CI can rebuild the reported numbers from the committed sample without any download.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import inventory_row, nearest_neighbour_km, summarise_inventory
from .curves import CANONICAL, from_las
from .features import LOG_CURVES, vshale
from .las import LasError, read_las

URL = ("https://zenodo.org/api/records/4351156/files/"
       "LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip/content")
FILENAME = "LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip"
SIZE = 169_785_172
MD5 = "da0d72d5783a780ac71779453d9fbfbf"          # published by Zenodo
SHA256 = "042ef1d473100dc12768cae7107a8cead071a394f137af3bc1ddaf8a5d1b8dda"   # computed locally
SANDSTONE = 30000                                    # FORCE lithology code for sandstone
LITHOLOGY = {30000: "sandstone", 65030: "sandstone/shale", 65000: "shale", 80000: "marl",
             74000: "dolomite", 70000: "limestone", 70032: "chalk", 88000: "halite",
             86000: "anhydrite", 99000: "tuff", 90000: "coal", 93000: "basement"}
CLEAN_VSH = 0.4                                      # GR index below this = "clean"


def file_hashes(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def verify(path: Path) -> None:
    md5, sha = file_hashes(path)
    if md5 != MD5 or sha != SHA256:
        raise ValueError(f"checksum mismatch for {path}: md5 {md5}, sha256 {sha}")


def download(dest_dir: Path) -> Path:
    """Download the zip into dest_dir (if not already there) and verify both checksums."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / FILENAME
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)     # fixed https URL, checked below
        tmp.replace(path)
    verify(path)
    return path


def well_row(name: str, text: str) -> dict:
    """One well's audit row. The audit interval is the logged gamma-ray interval (first to
    last finite GR sample), because the files have no target zone."""
    las = read_las(text)
    logs = from_las(las)
    d = logs.depth
    gr = logs.curves.get("GR")
    fin = np.flatnonzero(np.isfinite(gr)) if gr is not None else np.array([], dtype=int)
    top, base = (float(d[fin[0]]), float(d[fin[-1]])) if len(fin) else (np.nan, np.nan)
    row = inventory_row(name, logs, top, base)
    row["empty_curves"] = ";".join(sorted(logs.empty))
    for k in ("X_LOC", "Y_LOC"):
        v = las.curve(k) if k in las.mnemonics else np.array([np.nan])
        row[k.lower()[0] + "_km"] = round(float(np.nanmedian(v)) / 1000.0, 3) \
            if np.isfinite(v).any() else np.nan
    for c in CANONICAL:
        v = logs.curves.get(c)
        row[f"{c}_p50"] = round(float(np.nanmedian(v)), 4) \
            if v is not None and np.isfinite(v).any() else np.nan
    # lithology check over the GR interval: does "low GR" mean "sandstone"?
    lith_name = "FORCE_2020_LITHOFACIES_LITHOLOGY"
    if lith_name in las.mnemonics and len(fin):
        z = (d >= top) & (d <= base)
        lith = las.curve(lith_name)[z]
        g = gr[z]
        ok = np.isfinite(lith) & np.isfinite(g)
        lo, hi = np.percentile(g[np.isfinite(g)], [5, 95])
        clean = vshale(g, lo, hi) < CLEAN_VSH
        sand = lith == SANDSTONE
        codes, counts = np.unique(lith[ok & clean].astype(int), return_counts=True)
        row.update(n_lith=int(ok.sum()), n_sand_lith=int((ok & sand).sum()),
                   n_clean_gr=int((ok & clean).sum()),
                   n_clean_and_sand=int((ok & clean & sand).sum()),
                   clean_by_lithology=";".join(f"{c}:{n}" for c, n in zip(codes, counts)))
    else:
        row.update(n_lith=0, n_sand_lith=0, n_clean_gr=0, n_clean_and_sand=0,
                   clean_by_lithology="")
    return row


def summarise_zip(zip_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Per-well rows for every LAS file in the zip, plus a list of files that failed."""
    rows, failed = [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.lower().endswith(".las")):
            well = Path(name).stem
            try:
                rows.append(well_row(well, z.read(name).decode("latin-1")))
            except (LasError, ValueError) as exc:
                failed.append({"file": Path(name).name, "error": str(exc)[:200]})
    return pd.DataFrame(rows), failed


def _spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(a.corr(b))


def summary_from_sample(df: pd.DataFrame) -> dict:
    """The reported FORCE 2020 numbers, computed only from the committed per-well sample."""
    df = df.copy()
    for c in df.columns:
        if c.startswith(("raw_", "unit_", "conv_", "unknown", "duplicate", "empty", "clean_by")):
            df[c] = df[c].fillna("").astype(str)
    inv = summarise_inventory(df, curves=LOG_CURVES)
    loose = summarise_inventory(df, curves=LOG_CURVES, min_coverage=0.5)["survival"]
    by_lith: dict[str, int] = {}
    for s in df["clean_by_lithology"]:
        for item in filter(None, s.split(";")):
            code, n = item.split(":")
            name = LITHOLOGY.get(int(code), f"code {code}")
            by_lith[name] = by_lith.get(name, 0) + int(n)
    xy = df[["x_km", "y_km"]].to_numpy(float)
    ok = np.isfinite(xy).all(axis=1)
    nn = nearest_neighbour_km(xy[ok])
    interval = (df["base_m"] - df["top_m"]).dropna()
    lith = df[df["n_lith"] > 0]
    n_s, n_c, n_cs = (int(lith[k].sum()) for k in ("n_sand_lith", "n_clean_gr",
                                                  "n_clean_and_sand"))
    empty = pd.Series([m for s in df["empty_curves"] for m in s.split(";") if m])
    return {
        "source": {"zenodo_record": 4351156, "doi": "10.5281/zenodo.4351156",
                   "licence": "CC BY 4.0 (underlying government data: NLOD 2.0)",
                   "zip_bytes": SIZE, "md5": MD5, "sha256": SHA256},
        "wells": int(len(df)),
        "depth_step_m": {str(k): int(v) for k, v in
                         sorted(df["step_m"].round(4).value_counts().items())},
        "gr_interval_m": {"min": round(float(interval.min()), 1),
                          "median": round(float(interval.median()), 1),
                          "max": round(float(interval.max()), 1)},
        "logs": inv,
        "survival_at_min_coverage_0_5": loose,
        "declared_but_empty_curves": {str(k): int(v) for k, v in
                                      empty.value_counts().sort_index().items()},
        "spatial": {"wells_with_location": int(ok.sum()),
                    "nearest_neighbour_km": {"min": round(float(nn.min()), 3),
                                             "median": round(float(np.median(nn)), 3),
                                             "q90": round(float(np.quantile(nn, 0.9)), 3)},
                    "wells_with_neighbour_within_2km": int((nn < 2.0).sum())},
        "gr_sand_flag_vs_lithology": {
            "wells": int(len(lith)), "samples": int(lith["n_lith"].sum()),
            "sandstone_samples": n_s, "clean_gr_samples": n_c,
            "recall_sandstone": round(n_cs / n_s, 4) if n_s else None,
            "precision_sandstone": round(n_cs / n_c, 4) if n_c else None,
            "spearman_well_sand_fraction": round(_spearman(
                lith["n_sand_lith"] / lith["n_lith"], lith["n_clean_gr"] / lith["n_lith"]), 4),
            "clean_gr_samples_by_lithology": dict(sorted(by_lith.items(),
                                                         key=lambda kv: -kv[1])),
            "rule": f"clean if GR index < {CLEAN_VSH}, GR endpoints = per-well P05/P95"},
    }
