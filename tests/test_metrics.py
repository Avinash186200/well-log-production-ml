import numpy as np
from scipy.stats import spearmanr

from wellprod.metrics import (bias_ratio, mae, pad_bootstrap_se, quantile_calibration,
                              spearman, top_decile_capture, top_k)


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    y = np.round(rng.gamma(2, 1, 200), 1)
    y[:20] = 0.0
    p = y + rng.normal(0, 1, 200)
    assert abs(spearman(p, y) - spearmanr(p, y).statistic) < 1e-12
    assert np.isnan(spearman(np.ones(5), np.arange(5)))


def test_top_decile_capture():
    y = np.arange(1.0, 101.0)
    assert top_decile_capture(y, y) == 1.0
    assert top_decile_capture(-y, y) == sum(range(1, 11)) / sum(range(91, 101))
    assert len(top_k(np.zeros(25), 0.1)) == 2          # round(2.5) = 2 (banker's rounding)
    assert list(top_k(np.array([1.0, 3.0, 3.0, 0.0]), 0.5)) == [1, 2]   # ties by row order


def test_mae_and_bias_ratio():
    assert mae([1, 2, 3], [2, 2, 5]) == 1.0
    assert bias_ratio([1, 1], [2, 2]) == 0.5


def test_quantile_calibration_counts_shares():
    y = np.arange(10.0)
    q = {0.1: np.full(10, 1.0), 0.5: np.full(10, 5.0), 0.9: np.full(10, 9.0)}
    c = quantile_calibration(q, y)
    assert c["share_below_q10"] == 0.1 and c["share_below_q50"] == 0.5
    assert c["share_below_q90"] == 0.9 and c["coverage_central"] == 0.9
    assert c["mean_width_bbl"] == 8.0


def test_pad_bootstrap_is_deterministic_and_positive():
    rng = np.random.default_rng(1)
    y = rng.gamma(2, 1, 120)
    p = y + rng.normal(0, 1, 120)
    pads = np.repeat(np.arange(30), 4)
    a = pad_bootstrap_se(p, y, pads, spearman, n_boot=200, seed=0)
    assert a == pad_bootstrap_se(p, y, pads, spearman, n_boot=200, seed=0)
    assert 0 < a < 0.5
