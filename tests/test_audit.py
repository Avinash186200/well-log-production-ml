import numpy as np

from wellprod.audit import audit, correlogram, morans_i
from wellprod.synth import DATA_DICTIONARY


def test_audit_counts_and_flags(small_basin):
    basin, logs = small_basin
    w = basin.wells
    res = audit(w, logs, DATA_DICTIONARY)["summary"]
    assert res["wells"] == len(w)
    chain = [s["wells"] for s in res["logs"]["survival"]]
    assert chain == sorted(chain, reverse=True) and chain[0] <= len(w)
    lk = res["leakage"]
    assert set(lk["post_completion_by_dictionary"]) == {
        "ip30_bopd", "peak_month_oil_bbl", "cum6_oil_bbl", "water_cut_3m"}
    assert {"ip30_bopd", "cum6_oil_bbl"} <= set(lk["flagged_by_data"])
    assert lk["flagged_by_data_not_dictionary"] == []
    assert res["impossible_values"]["lateral_m <= 0"] == int((w["lateral_m"] <= 0).sum())
    assert res["target"]["zero_share"] == round(float((w["cum12_oil_bbl"] == 0).mean()), 4)
    gr = res["logs"]["curves"]["GR"]
    assert sum(gr["aliases"].values()) == gr["wells_with_curve"]


def test_morans_i_sees_smooth_fields_not_noise():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 50, (300, 2))
    smooth = np.sin(xy[:, 0] / 8) + np.cos(xy[:, 1] / 8)
    noise = rng.normal(size=300)
    a, b = morans_i(xy, smooth, n_perm=99), morans_i(xy, noise, n_perm=99)
    assert a["I"] > 0.5 and a["p_value"] == 0.01
    assert abs(b["I"]) < 0.15


def test_correlogram_decays_for_smooth_field():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 60, (300, 2))
    v = np.sin(xy[:, 0] / 10)
    bands = correlogram(xy, v, bands=(0, 2, 40))
    assert bands[0]["correlation"] > bands[1]["correlation"]
    assert bands[0]["pairs"] + bands[1]["pairs"] <= 300 * 299 // 2
