import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wellprod.las import read_las
from wellprod.synth import DATA_DICTIONARY, simulate_basin

from conftest import SMALL


def test_same_seed_same_basin():
    a, b = simulate_basin(5, SMALL), simulate_basin(5, SMALL)
    pd.testing.assert_frame_equal(a.wells, b.wells)
    assert a.las == b.las
    assert not simulate_basin(6, SMALL).wells.equals(a.wells)


def test_split_by_completion_year_and_whole_pads(small_basin):
    basin, _ = small_basin
    w = basin.wells
    assert (w.loc[w.split == "holdout", "completion_year"] >= SMALL.holdout_from_year).all()
    assert (w.loc[w.split == "dev", "completion_year"] < SMALL.holdout_from_year).all()
    # a pad is drilled in one year, so it never straddles the split
    assert (w.groupby("pad_id")["split"].nunique() == 1).all()


def test_every_column_is_in_the_data_dictionary(small_basin):
    basin, _ = small_basin
    assert set(basin.wells.columns) - {"split"} == set(DATA_DICTIONARY)


def test_target_is_skewed_zero_inflated_and_non_negative(small_basin):
    basin, _ = small_basin
    y = basin.wells["cum12_oil_bbl"]
    assert (y >= 0).all()
    assert 0.02 < (y == 0).mean() < 0.3
    assert y.mean() > y.median()


def test_post_completion_columns_leak_the_target(small_basin):
    basin, _ = small_basin
    w = basin.wells
    for c in ("ip30_bopd", "cum6_oil_bbl"):
        assert spearmanr(w[c], w["cum12_oil_bbl"]).statistic > 0.9


def test_las_files_parse_and_carry_the_well_name(small_basin):
    basin, _ = small_basin
    for wid, text in list(basin.las.items())[:20]:
        las = read_las(text)
        assert las.well_value("WELL") == wid
        assert las.mnemonics[0] == "DEPT"
        assert np.isfinite(las.data[:, 0]).all()


def test_truth_expected_value_ranks_wells_better_than_noise(small_basin):
    basin, _ = small_basin
    rho = spearmanr(basin.truth["expected_cum12"], basin.wells["cum12_oil_bbl"]).statistic
    assert rho > 0.5
