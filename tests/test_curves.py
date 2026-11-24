import numpy as np

from wellprod.curves import canonical_name, canonicalise, coverage

D = np.arange(0.0, 10.0, 0.5)


def test_aliases_map_to_one_canonical_name():
    assert {canonical_name(m) for m in ("GR", "grc", "SGR")} == {"GR"}
    assert {canonical_name(m) for m in ("RDEP", "ILD", "LLD")} == {"RT"}
    assert canonical_name("FOO") is None


def test_alias_with_most_data_wins_and_the_other_is_listed():
    gr = np.full(len(D), 60.0)
    sparse = np.where(np.arange(len(D)) < 3, 50.0, np.nan)
    logs = canonicalise(D, {"GR": ("GAPI", sparse), "SGR": ("GAPI", gr), "XYZ": ("", gr),
                            "CALI": ("IN", np.full(len(D), np.nan))})
    assert logs.fixes["GR"].raw_mnemonic == "SGR"
    assert logs.duplicates == ["GR"] and logs.unknown == ["XYZ"] and logs.empty == ["CALI"]


def test_units_judged_by_values_and_wrong_labels_flagged():
    rhob = np.full(len(D), 2450.0)
    nphi = np.full(len(D), 22.0)
    dt = np.full(len(D), 300.0)
    logs = canonicalise(D, {"RHOB": ("G/C3", rhob), "NPHI": ("V/V", nphi),
                            "DT": ("US/M", dt), "NEU": ("PU", nphi)})
    np.testing.assert_allclose(logs.curves["RHOB"], 2.45)
    assert logs.fixes["RHOB"].conversion == "kg/m3 -> g/cm3" and logs.fixes["RHOB"].unit_mismatch
    np.testing.assert_allclose(logs.curves["NPHI"], 0.22)
    assert logs.fixes["NPHI"].unit_mismatch               # percent labelled V/V
    np.testing.assert_allclose(logs.curves["DT"], 300.0 / 3.28084)
    assert not logs.fixes["DT"].unit_mismatch             # correctly labelled us/m


def test_correct_percent_label_is_not_a_mismatch():
    logs = canonicalise(D, {"NEU": ("PU", np.full(len(D), 20.0))})
    assert logs.fixes["NPHI"].conversion == "percent -> v/v"
    assert not logs.fixes["NPHI"].unit_mismatch


def test_spikes_become_nan_and_are_counted():
    gr = np.full(len(D), 60.0)
    gr[[2, 5]] = [900.0, -5.0]
    logs = canonicalise(D, {"GR": ("GAPI", gr)})
    assert logs.fixes["GR"].n_spikes == 2
    assert np.isnan(logs.curves["GR"][[2, 5]]).all()


def test_coverage_counts_missing_top_of_zone():
    depth = np.arange(5.0, 20.0, 0.5)           # log starts at 5 m
    logs = canonicalise(depth, {"GR": ("GAPI", np.full(len(depth), 60.0))})
    cov = coverage(logs, 0.0, 10.0)              # zone 0-10 m, 21 expected samples
    assert abs(cov["GR"] - 11 / 21) < 1e-12
    assert cov["RHOB"] == 0.0
