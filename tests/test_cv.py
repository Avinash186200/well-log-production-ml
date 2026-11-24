import numpy as np
import pandas as pd
import pytest

from wellprod.cv import (kfold_grouped, kfold_random, make_splits, nearest_train_distance,
                         spatial_blocks)


def _check_partition(splits, n):
    tests = np.concatenate([te for _, te in splits])
    assert sorted(tests) == list(range(n))
    for tr, te in splits:
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == n


def test_random_kfold_partitions_rows():
    sp = kfold_random(23, 5, 0)
    _check_partition(sp, 23)
    assert [len(te) for _, te in sp] == [5, 5, 5, 4, 4]
    assert all((a[1] == b[1]).all() for a, b in zip(sp, kfold_random(23, 5, 0)))


def test_grouped_kfold_keeps_groups_whole_and_balanced():
    groups = np.repeat(np.arange(20), np.arange(1, 21) % 6 + 1)
    sp = kfold_grouped(groups, 5, 3)
    _check_partition(sp, len(groups))
    for tr, te in sp:
        assert not set(groups[tr]) & set(groups[te])
    sizes = [len(te) for _, te in sp]
    assert max(sizes) - min(sizes) <= 6


def test_too_few_groups_raise():
    with pytest.raises(ValueError):
        kfold_grouped(np.array([1, 1, 2]), 3, 0)


def test_spatial_blocks_are_square_cells():
    b = spatial_blocks(np.array([0.5, 9.9, 10.1, 25.0]), np.array([0.0, 0.0, 0.0, 31.0]), 10.0)
    assert list(b) == ["0_0", "0_0", "1_0", "2_3"]


def test_grouped_schemes_push_test_wells_further_from_training(small_basin):
    basin, _ = small_basin
    w = basin.wells.reset_index(drop=True)
    med = {}
    for s in ("random", "pad", "block", "field"):
        sp = make_splits(s, w, 5, 0, 10.0)
        _check_partition(sp, len(w))
        med[s] = float(np.median(nearest_train_distance(w, sp)))
    assert med["random"] < med["pad"] < med["block"] < med["field"]


def test_field_scheme_is_leave_one_field_out(small_basin):
    basin, _ = small_basin
    w = basin.wells.reset_index(drop=True)
    sp = make_splits("field", w, 5, 0, 10.0)
    assert len(sp) == w["field_id"].nunique()
    for _, te in sp:
        assert w.loc[te, "field_id"].nunique() == 1


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        make_splits("time", pd.DataFrame(), 5, 0, 10.0)


def test_block_folds_never_split_a_pad(small_basin):
    basin, _ = small_basin
    w = basin.wells.reset_index(drop=True)
    for tr, te in make_splits("block", w, 5, 0, 10.0):
        assert not set(w.loc[tr, "pad_id"]) & set(w.loc[te, "pad_id"])
