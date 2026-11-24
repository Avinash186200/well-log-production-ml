import numpy as np
import pandas as pd

from wellprod.curves import canonicalise
from wellprod.features import (archie_sw, feature_table, log_feature_names, log_features,
                               porosity, vshale)
from wellprod.synth import DATA_DICTIONARY, RW


def test_petrophysics_formulas():
    np.testing.assert_allclose(vshale(np.array([25.0, 75.0, 125.0, 300.0])), [0, 0.5, 1, 1])
    np.testing.assert_allclose(porosity(np.array([2.65, 2.32]), None), [0.0, 0.2])
    np.testing.assert_allclose(porosity(np.array([2.32]), np.array([0.3])), [0.25])
    # Archie with a = 1, m = n = 2: Sw = sqrt(Rw / (phi^2 Rt))
    np.testing.assert_allclose(archie_sw(np.array([RW / (0.2 ** 2 * 0.5 ** 2)]),
                                         np.array([0.2])), [0.5])


def _clean_well():
    d = np.arange(0.0, 20.0, 0.5)
    zone = (d >= 5) & (d <= 15)
    sand = zone & (d < 10)
    phi = np.where(sand, 0.2, 0.03)
    sw = np.where(sand, 0.4, 1.0)
    raw = {"GR": ("GAPI", np.where(sand, 30.0, 110.0)),
           "RHOB": ("G/C3", 2.65 - phi * 1.65),
           "RDEP": ("OHMM", RW / (phi ** 2 * sw ** 2))}
    return canonicalise(d, raw), sand.sum()


def test_log_features_recover_known_pay():
    logs, n_sand = _clean_well()
    f = log_features(logs, 5.0, 15.0)
    assert f["net_pay_m"] == n_sand * 0.5
    np.testing.assert_allclose(f["hcpv_m"], n_sand * 0.5 * 0.2 * 0.6, rtol=1e-9)
    assert np.isnan(f["NPHI_mean"]) and f["cov_NPHI"] == 0.0
    assert set(f) == set(log_feature_names())


def test_missing_core_curve_gives_nan_petrophysics():
    logs, _ = _clean_well()
    del logs.curves["RHOB"]
    f = log_features(logs, 5.0, 15.0)
    assert np.isnan(f["hcpv_m"]) and np.isnan(f["phi_mean"])
    assert np.isfinite(f["GR_mean"])


def test_features_exclude_post_completion_columns_and_are_row_independent(small_basin):
    basin, logs = small_basin
    X = feature_table(basin.wells, logs)
    leak = {c for c, (when, _) in DATA_DICTIONARY.items() if when in ("post-completion", "target")}
    assert not leak & set(X.columns)
    # a well's features depend on that well only: computing on a subset changes nothing
    sub = basin.wells.iloc[::3]
    pd.testing.assert_frame_equal(feature_table(sub, logs), X.loc[sub.index])


def test_impossible_lateral_becomes_nan(small_basin):
    basin, logs = small_basin
    w = basin.wells.head(3).copy()
    w.loc[w.index[0], "lateral_m"] = 0.0
    X = feature_table(w, logs)
    assert np.isnan(X["lateral_m"].iloc[0]) and np.isfinite(X["lateral_m"].iloc[1])
