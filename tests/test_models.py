import numpy as np
import pandas as pd

from wellprod.models import LGBM, KNNLocation, TypeCurve, fit_quantiles, grid, param_key

P = {"num_leaves": 4, "min_data_in_leaf": 10, "num_rounds": 50, "tweedie_variance_power": 1.5}


def test_type_curve_scales_field_median_per_metre():
    w = pd.DataFrame({"field_id": ["A", "A", "B"], "lateral_m": [1000.0, 2000.0, 1000.0]})
    tc = TypeCurve().fit(w, [10_000.0, 30_000.0, 5_000.0])
    new = pd.DataFrame({"field_id": ["A", "B", "C", "A"],
                        "lateral_m": [1500.0, 2000.0, 1000.0, np.nan]})
    # A: median(10, 15) = 12.5 bbl/m; B: 5 bbl/m; C unknown: median(10, 15, 5) = 10 bbl/m;
    # missing lateral: field median of the target (A: 20,000)
    np.testing.assert_allclose(tc.predict(new), [18_750.0, 10_000.0, 10_000.0, 20_000.0])


def test_knn_averages_nearest_wells():
    w = pd.DataFrame({"x_km": [0.0, 1.0, 10.0], "y_km": [0.0, 0.0, 0.0]})
    m = KNNLocation(2).fit(w, [1.0, 3.0, 100.0])
    np.testing.assert_allclose(m.predict(pd.DataFrame({"x_km": [0.2], "y_km": [0.0]})), [2.0])


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    mu = np.exp(1.0 + 0.8 * X["a"])
    y = np.where(rng.random(n) < 0.1, 0.0, rng.gamma(4, mu / 4))
    return X, y


def test_lgbm_is_deterministic_and_non_negative():
    X, y = _data()
    for obj in ("tweedie", "log"):
        a = LGBM(obj, P).fit(X, y).predict(X)
        b = LGBM(obj, P).fit(X, y).predict(X)
        assert (a == b).all() and (a >= 0).all()


def test_tweedie_learns_the_signal_and_log_target_underpredicts_the_mean():
    X, y = _data(2000)
    tw = LGBM("tweedie", P).fit(X, y).predict(X)
    lg = LGBM("log", P).fit(X, y).predict(X)
    assert np.corrcoef(tw, X["a"])[0, 1] > 0.8
    assert abs(tw.sum() / y.sum() - 1) < 0.05
    assert lg.sum() / y.sum() < 0.9


def test_quantiles_never_cross():
    X, y = _data()
    q = fit_quantiles(X, y, P)(X)
    assert (q[0.1] <= q[0.5]).all() and (q[0.5] <= q[0.9]).all()


def test_shap_contributions_sum_to_raw_score():
    X, y = _data()
    m = LGBM("tweedie", P).fit(X, y)
    c = m.contributions(X)
    raw = m.booster.predict(X.to_numpy(float), raw_score=True)
    np.testing.assert_allclose(c.sum(1), raw, rtol=1e-9, atol=1e-9)


def test_grid_is_full_product_with_unique_keys():
    g = grid()
    assert len(g) == 16 and len({param_key(p) for p in g}) == 16
