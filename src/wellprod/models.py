"""Models: two baselines a reservoir engineer would use, and LightGBM.

- `TypeCurve`: per-field median oil per metre of lateral times the well's lateral length
  (the usual "type curve scaled by lateral" estimate). Unknown field -> all-field median.
- `KNNLocation`: the mean 12-month oil of the k nearest training wells by map distance
  ("look at the offset wells").
- `LGBM` with three objectives:
  * `tweedie`: predicts the mean directly; handles exact zeros and a long right tail;
  * `log`: L2 on log1p(y), back-transformed with expm1. This predicts something closer to a
    geometric mean than the mean (zeros pull it down further), so it under-predicts totals
    (see `bias_ratio`). No smearing correction is applied; one would fix most of the total
    bias but not the ranking;
  * `quantile`: one model per percentile (10th, 50th, 90th) for P90 / P50 / P10 ranges.

LightGBM runs single-threaded with `deterministic=True` and a fixed seed, so results repeat
exactly on the same library versions.
"""
from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_PARAMS = {"learning_rate": 0.05, "feature_fraction": 0.8, "lambda_l2": 1.0,
               "num_threads": 1, "deterministic": True, "force_col_wise": True, "seed": 7,
               "verbosity": -1}

GRID = {"num_leaves": (4, 15), "min_data_in_leaf": (10, 40), "num_rounds": (100, 300),
        "tweedie_variance_power": (1.2, 1.5)}
KNN_K = (3, 5, 10, 20)


def grid() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]


def param_key(p: dict) -> str:
    return ",".join(f"{k}={p[k]}" for k in sorted(p))


class TypeCurve:
    name = "type curve (field median per metre x lateral)"

    def fit(self, wells: pd.DataFrame, y):
        lat = wells["lateral_m"].where(wells["lateral_m"] > 0)
        per_m = pd.Series(np.asarray(y, float), index=wells.index) / lat
        df = pd.DataFrame({"field": wells["field_id"], "per_m": per_m,
                           "y": np.asarray(y, float)})
        self.field_per_m = df.dropna().groupby("field")["per_m"].median().to_dict()
        self.field_y = df.groupby("field")["y"].median().to_dict()
        self.all_per_m = float(df["per_m"].median())
        self.all_y = float(df["y"].median())
        return self

    def predict(self, wells: pd.DataFrame) -> np.ndarray:
        out = []
        for f, lat in zip(wells["field_id"], wells["lateral_m"]):
            if np.isfinite(lat) and lat > 0:
                out.append(self.field_per_m.get(f, self.all_per_m) * lat)
            else:
                out.append(self.field_y.get(f, self.all_y))
        return np.array(out)


class KNNLocation:
    def __init__(self, k: int):
        self.k = k
        self.name = f"kNN on location (k={k})"

    def fit(self, wells: pd.DataFrame, y):
        self.xy = wells[["x_km", "y_km"]].to_numpy(float)
        self.y = np.asarray(y, float)
        return self

    def predict(self, wells: pd.DataFrame) -> np.ndarray:
        q = wells[["x_km", "y_km"]].to_numpy(float)
        d = ((q[:, None, :] - self.xy[None, :, :]) ** 2).sum(-1)
        k = min(self.k, len(self.y))
        nn = np.argsort(d, axis=1, kind="stable")[:, :k]
        return self.y[nn].mean(axis=1)


class LGBM:
    def __init__(self, objective: str, params: dict, alpha: float | None = None):
        if objective not in ("tweedie", "log", "quantile"):
            raise ValueError(objective)
        self.objective, self.params, self.alpha = objective, dict(params), alpha

    def _lgb_params(self) -> dict:
        p = {**BASE_PARAMS, "num_leaves": self.params["num_leaves"],
             "min_data_in_leaf": self.params["min_data_in_leaf"]}
        if self.objective == "tweedie":
            p.update(objective="tweedie",
                     tweedie_variance_power=self.params["tweedie_variance_power"])
        elif self.objective == "log":
            p.update(objective="regression")
        else:
            p.update(objective="quantile", alpha=self.alpha)
        return p

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        target = np.log1p(y) if self.objective == "log" else y
        self.columns = list(X.columns)
        ds = lgb.Dataset(X.to_numpy(float), label=target, feature_name=self.columns,
                         free_raw_data=True)
        # keep_training_booster skips the save-to-string-and-reload step at the end of training
        self.booster = lgb.train(self._lgb_params(), ds,
                                 num_boost_round=self.params["num_rounds"],
                                 keep_training_booster=True)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(X[self.columns].to_numpy(float))
        if self.objective == "log":
            return np.expm1(raw).clip(min=0.0)
        return raw.clip(min=0.0) if self.objective == "quantile" else raw

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP values on the raw score (log scale for tweedie), one column per feature
        plus the expected value in the last column."""
        return self.booster.predict(X[self.columns].to_numpy(float), pred_contrib=True)


def fit_quantiles(X: pd.DataFrame, y, params: dict, alphas=(0.1, 0.5, 0.9)):
    """One quantile model per alpha. Predictions are sorted per row so they never cross."""
    models = {a: LGBM("quantile", params, alpha=a).fit(X, y) for a in alphas}

    def predict(Xn: pd.DataFrame) -> dict[float, np.ndarray]:
        raw = np.column_stack([models[a].predict(Xn) for a in alphas])
        raw.sort(axis=1)
        return {a: raw[:, i] for i, a in enumerate(alphas)}

    return predict
