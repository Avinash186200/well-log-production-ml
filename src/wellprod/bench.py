"""The full offline run: simulate, audit, build features, cross-validate four ways on the
development wells, choose settings by spatial-block CV, score the held-out wells once, and
write results/*.json plus RESULTS.md.

Order of operations (nothing below a line looks at anything the line above it hides):

1. simulate the basin; parse every LAS file; split wells into development (completed before
   2022) and held-out (2022 onwards: the newest, mostly outer pads);
2. audit and feature choices use development wells only;
3. every hyperparameter (LightGBM grid, kNN k) is chosen by pooled out-of-fold Spearman under
   spatial-block CV on development wells; the same grid is also scored under random, pad and
   leave-field-out CV, only to measure how optimistic each one is;
4. final models are refitted on all development wells and score the held-out wells once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import audit
from .curves import from_las
from .cv import SCHEMES, make_splits, nearest_train_distance
from .features import feature_table
from .force2020 import summary_from_sample
from .las import read_las
from .metrics import (pad_bootstrap_se, quantile_calibration, ranking_metrics, spearman,
                      top_decile_capture, mae)
from .models import KNN_K, LGBM, KNNLocation, TypeCurve, fit_quantiles, grid, param_key
from .synth import DATA_DICTIONARY, SynthConfig, simulate_basin

SEED = 7
K_FOLDS = 5
CV_SEED = 0
BLOCK_KM = 10.0
SELECT_SCHEME = "block"
N_BOOT = 1000
TARGET = "cum12_oil_bbl"
RESULT_FILES = ("data", "audit", "cv", "holdout", "force2020")
ROOT = Path(__file__).resolve().parents[2]
FORCE_SAMPLE = ROOT / "samples" / "force2020_wells.csv"


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _rm(m: dict) -> dict:
    return {k: (v if k == "n" else _r(v, 1 if k == "mae_bbl" else 4)) for k, v in m.items()}


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8",
                    newline="\n")


def load(seed: int = SEED, cfg: SynthConfig | None = None):
    basin = simulate_basin(seed, cfg)
    logs = {w: from_las(read_las(t)) for w, t in basin.las.items()}
    return basin, logs


def _oof(make, X, wells, y, splits, uses_wells: bool) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        m = make()
        if uses_wells:
            m.fit(wells.iloc[tr], y[tr])
            pred[te] = m.predict(wells.iloc[te])
        else:
            m.fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
    return pred


def run(out: Path, seed: int = SEED, cfg: SynthConfig | None = None, log=print,
        grid_override: list[dict] | None = None, n_boot: int = N_BOOT,
        force_sample: Path | None = FORCE_SAMPLE) -> dict:
    cfg = cfg or SynthConfig()
    out.mkdir(parents=True, exist_ok=True)
    basin, logs = load(seed, cfg)
    wells = basin.wells
    dev_m = (wells["split"] == "dev").to_numpy()
    wd = wells[dev_m].reset_index(drop=True)
    wh = wells[~dev_m].reset_index(drop=True)
    yd, yh = wd[TARGET].to_numpy(float), wh[TARGET].to_numpy(float)
    log(f"basin: {len(wells)} wells, {len(wd)} development, {len(wh)} held-out")

    data = {"seed": seed, "config": {k: v for k, v in cfg.__dict__.items()},
            "wells": int(len(wells)), "pads": int(wells["pad_id"].nunique()),
            "fields": int(wells["field_id"].nunique())}
    for name, w in (("dev", wd), ("holdout", wh)):
        data[name] = {"wells": int(len(w)), "pads": int(w["pad_id"].nunique()),
                      "fields": int(w["field_id"].nunique()),
                      "completion_years": [int(w["completion_year"].min()),
                                           int(w["completion_year"].max())],
                      "median_lateral_m": _r(w["lateral_m"].where(w["lateral_m"] > 0).median(),
                                             1),
                      "median_proppant_kg_m": _r(w["proppant_kg_m"].median(), 1)}
    # how far the held-out wells are from development wells (a step-out measure)
    xy_d, xy_h = wd[["x_km", "y_km"]].to_numpy(), wh[["x_km", "y_km"]].to_numpy()
    nn_h = np.sqrt(((xy_h[:, None] - xy_d[None]) ** 2).sum(-1)).min(1)
    data["holdout"]["nearest_dev_well_km_median"] = _r(np.median(nn_h), 3)
    data["holdout"]["new_pads_share"] = _r(np.mean(~wh["pad_id"].isin(wd["pad_id"])), 4)
    # completion designs outside the development range: trees cannot extrapolate past it
    data["holdout"]["share_above_dev_max"] = {
        c: _r(np.mean(wh[c] > wd[c].max()), 4) for c in ("lateral_m", "proppant_kg_m")}

    # ---- audit (development wells only) ------------------------------------------------------
    aud = audit(wd, {w: logs[w] for w in wd["well_id"]}, DATA_DICTIONARY, TARGET)
    _write_json(out / "audit.json", aud["summary"])
    log("audit done")

    # ---- features ------------------------------------------------------------------------------
    Xd = feature_table(wd, logs)
    Xh = feature_table(wh, logs)
    Xd_nl = Xd.drop(columns=["x_km", "y_km"])
    Xh_nl = Xh.drop(columns=["x_km", "y_km"])
    data["features"] = {"n": int(Xd.shape[1]), "names": list(Xd.columns)}
    _write_json(out / "data.json", data)

    # ---- cross-validation on development wells -------------------------------------------------
    g = grid_override or grid()
    splits = {s: make_splits(s, wd, K_FOLDS, CV_SEED, BLOCK_KM) for s in SCHEMES}
    cv: dict = {"k_folds": K_FOLDS, "cv_seed": CV_SEED, "block_km": BLOCK_KM,
                "select_scheme": SELECT_SCHEME, "criterion": "pooled out-of-fold Spearman",
                "schemes": {}}
    oof_store: dict = {}
    for s in SCHEMES:
        sp = splits[s]
        nd = nearest_train_distance(wd, sp)
        ent = {"folds": len(sp), "test_wells_per_fold": [int(len(te)) for _, te in sp],
               "nearest_train_well_km": {"median": _r(np.median(nd), 3),
                                         "q10": _r(np.quantile(nd, 0.1), 3)},
               "tweedie_grid": [], "knn": []}
        for p in g:
            pred = _oof(lambda p=p: LGBM("tweedie", p), Xd, wd, yd, sp, False)
            ent["tweedie_grid"].append({"params": p, **_rm(ranking_metrics(pred, yd))})
            oof_store[(s, "tweedie", param_key(p))] = pred
        for k in KNN_K:
            pred = _oof(lambda k=k: KNNLocation(k), Xd, wd, yd, sp, True)
            ent["knn"].append({"k": k, **_rm(ranking_metrics(pred, yd))})
            oof_store[(s, "knn", k)] = pred
        pred = _oof(TypeCurve, Xd, wd, yd, sp, True)
        ent["type_curve"] = _rm(ranking_metrics(pred, yd))
        # choice under THIS scheme (first best in grid order, so ties go to the simpler model)
        best_t = max(range(len(g)), key=lambda i: (ent["tweedie_grid"][i]["spearman"], -i))
        best_k = max(range(len(KNN_K)), key=lambda i: (ent["knn"][i]["spearman"], -i))
        ent["chosen_tweedie"] = g[best_t]
        ent["chosen_knn_k"] = KNN_K[best_k]
        cv["schemes"][s] = ent
        log(f"cv {s}: best tweedie Spearman {ent['tweedie_grid'][best_t]['spearman']}")

    sel = cv["schemes"][SELECT_SCHEME]
    p_sel, k_sel = sel["chosen_tweedie"], sel["chosen_knn_k"]
    p_rand = cv["schemes"]["random"]["chosen_tweedie"]
    cv["selected"] = {"tweedie": p_sel, "knn_k": k_sel, "random_cv_would_pick": p_rand}

    # the selected settings, scored under every scheme: the optimism table
    table = {}
    for s in SCHEMES:
        sp = splits[s]
        oofs = {"type_curve": _oof(TypeCurve, Xd, wd, yd, sp, True),
                "knn": oof_store[(s, "knn", k_sel)],
                "tweedie": oof_store[(s, "tweedie", param_key(p_sel))],
                "log_target": _oof(lambda: LGBM("log", p_sel), Xd, wd, yd, sp, False),
                "tweedie_no_location": _oof(lambda: LGBM("tweedie", p_sel), Xd_nl, wd, yd, sp,
                                            False)}
        row = {}
        for m, pr in oofs.items():
            row[m] = _rm(ranking_metrics(pr, yd))
            # pad bootstrap over the pooled out-of-fold predictions (models are not refitted,
            # so this is the sampling error of the wells, not of the model fitting)
            row[m]["se_spearman"] = _r(pad_bootstrap_se(pr, yd, wd["pad_id"], spearman,
                                                        n_boot))
            if s == "field":
                # mean Spearman inside each held-out field: ranking skill that does not depend
                # on getting each field's level right (pooled Spearman penalises wrong levels)
                row[m]["within_field_mean_spearman"] = _r(float(np.nanmean(
                    [spearman(pr[te], yd[te]) for _, te in sp])))
        table[s] = row
    cv["selected_under_each_scheme"] = table

    # quantile models: out-of-fold calibration under the selection scheme
    qpred = {a: np.full(len(yd), np.nan) for a in (0.1, 0.5, 0.9)}
    for tr, te in splits[SELECT_SCHEME]:
        pr = fit_quantiles(Xd.iloc[tr], yd[tr], p_sel)(Xd.iloc[te])
        for a in qpred:
            qpred[a][te] = pr[a]
    # the ceiling: rank by the generator's true expected value (no CV needed, nothing fitted)
    cv["oracle_dev"] = _rm(ranking_metrics(
        basin.truth.set_index("well_id").loc[wd["well_id"], "expected_cum12"].to_numpy(), yd))
    cv["top_decile_wells"] = int(round(0.1 * len(yd)))
    cv["tweedie_share_of_oracle_spearman"] = {
        s: _r(table[s]["tweedie"]["spearman"] / cv["oracle_dev"]["spearman"]) for s in SCHEMES}
    cv["quantile_oof_" + SELECT_SCHEME] = {k: _r(v) for k, v in
                                           quantile_calibration(qpred, yd).items()}
    _write_json(out / "cv.json", cv)
    log("cv done")

    # ---- held-out wells, scored once -----------------------------------------------------------
    pads_h = wh["pad_id"].to_numpy()
    truth = basin.truth.set_index("well_id")
    preds = {
        "type_curve": TypeCurve().fit(wd, yd).predict(wh),
        "knn": KNNLocation(k_sel).fit(wd, yd).predict(wh),
        "tweedie": None,
        "tweedie_random_cv_pick": LGBM("tweedie", p_rand).fit(Xd, yd).predict(Xh),
        "log_target": LGBM("log", p_sel).fit(Xd, yd).predict(Xh),
        "tweedie_no_location": LGBM("tweedie", p_sel).fit(Xd_nl, yd).predict(Xh_nl),
        "oracle": truth.loc[wh["well_id"], "expected_cum12"].to_numpy(),
    }
    final = LGBM("tweedie", p_sel).fit(Xd, yd)
    preds["tweedie"] = final.predict(Xh)
    ho = {"wells": int(len(wh)), "pads": int(len(np.unique(pads_h))),
          "zero_share": _r(np.mean(yh == 0)), "mean_bbl": _r(np.mean(yh), 1),
          "top_decile_wells": int(round(0.1 * len(wh))), "bootstrap": {
              "n_boot": n_boot, "unit": "pad", "seed": 0}, "models": {}}
    for name, pr in preds.items():
        m = _rm(ranking_metrics(pr, yh))
        m["se_spearman"] = _r(pad_bootstrap_se(pr, yh, pads_h, spearman, n_boot))
        m["se_top_decile_capture"] = _r(pad_bootstrap_se(pr, yh, pads_h, top_decile_capture,
                                                         n_boot))
        m["se_mae_bbl"] = _r(pad_bootstrap_se(pr, yh, pads_h, mae, n_boot), 1)
        ho["models"][name] = m
    ho["random_pick_capture_expected"] = _r(np.mean(yh) / np.mean(np.sort(yh)[::-1][
        :ho["top_decile_wells"]]))
    ho["tweedie_share_of_oracle_spearman"] = _r(ho["models"]["tweedie"]["spearman"]
                                                / ho["models"]["oracle"]["spearman"])
    qh = fit_quantiles(Xd, yd, p_sel)(Xh)
    ho["quantiles"] = {k: _r(v) for k, v in quantile_calibration(qh, yh).items()}
    contrib = final.contributions(Xh)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=final.columns).sort_values(ascending=False)
    ho["mean_abs_shap_top10"] = {k: _r(v) for k, v in imp.head(10).items()}
    _write_json(out / "holdout.json", ho)
    log("held-out done")

    force = None
    if force_sample is not None and Path(force_sample).exists():
        force = summary_from_sample(pd.read_csv(force_sample))
        _write_json(out / "force2020.json", force)
    elif force_sample is not None:
        log(f"FORCE 2020 sample not found at {force_sample} (run from a source checkout); "
            "skipping force2020.json")
    res = {"data": data, "audit": aud["summary"], "cv": cv, "holdout": ho, "force2020": force}
    (out / "RESULTS.md").write_text(render(res), encoding="utf-8", newline="\n")
    return res


# ------------------------------------------------------------------ report -----------------------
NAMES = {"type_curve": "type curve (field median per m x lateral)",
         "knn": "kNN on location", "tweedie": "LightGBM Tweedie (selected)",
         "tweedie_random_cv_pick": "LightGBM Tweedie (random-CV pick)",
         "log_target": "LightGBM on log1p(y)", "tweedie_no_location": "LightGBM Tweedie, no x/y",
         "oracle": "oracle: true expected value"}
SCHEME_NAMES = {"random": "random 5-fold", "pad": "grouped by pad (5-fold)",
                "block": f"spatial {BLOCK_KM:g} km blocks (5-fold)",
                "field": "leave one field out"}


def _f(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float) and abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render(res: dict) -> str:
    d, a, cv, ho, fz = res["data"], res["audit"], res["cv"], res["holdout"], res["force2020"]
    L = ["# Results", "",
         "Synthetic basin from `wellprod.synth` (seed "
         f"{d['seed']}) unless marked FORCE 2020. Rebuild with `make bench`.", "",
         "## Data", "",
         "| split | wells | pads | fields | completion years |", "|---|---|---|---|---|"]
    for s in ("dev", "holdout"):
        e = d[s]
        L.append(f"| {s} | {e['wells']} | {e['pads']} | {e['fields']} | "
                 f"{e['completion_years'][0]}-{e['completion_years'][1]} |")
    ext = d["holdout"]["share_above_dev_max"]
    L += ["", "| split | median lateral m | median proppant kg/m |", "|---|---|---|"]
    for s in ("dev", "holdout"):
        L.append(f"| {s} | {d[s]['median_lateral_m']:,.0f} | "
                 f"{d[s]['median_proppant_kg_m']:,.0f} |")
    L += ["", f"Held-out wells: {d['holdout']['new_pads_share']:.0%} are on pads with no "
          f"development well; median distance to the nearest development well "
          f"{d['holdout']['nearest_dev_well_km_median']} km; share above the development "
          f"maximum: lateral {ext['lateral_m']}, proppant {ext['proppant_kg_m']}.", "",
          "## Audit (development wells)", "",
          f"- target: zero share {a['target']['zero_share']}, median "
          f"{a['target']['quantiles_bbl']['q50']:,.0f} bbl, p99 "
          f"{a['target']['quantiles_bbl']['q99']:,.0f} bbl, skew {a['target']['skew']}",
          f"- leakage by dictionary: {', '.join(a['leakage']['post_completion_by_dictionary'])}",
          f"- leakage by data (|rank corr| > {a['leakage']['flag_threshold_abs_rho']}): "
          f"{', '.join(a['leakage']['flagged_by_data']) or 'none'}",
          f"- Moran's I of the target's rank, k=5: {a['spatial']['morans_i_rank_target']['I']} "
          f"(p = {a['spatial']['morans_i_rank_target']['p_value']})", "",
          "| requirement | wells |", "|---|---|"]
    for s in a["logs"]["survival"]:
        L.append(f"| {s['requires']} | {s['wells']} |")
    L += ["", "| curve | aliases | wells covering zone | unit conversions | declared unit "
          "wrong | wells with spikes |", "|---|---|---|---|---|---|"]
    for c, e in a["logs"]["curves"].items():
        conv = sum(e["conversions"].values())
        L.append(f"| {c} | {', '.join(e['aliases'])} | {e['wells_covering_zone']} | {conv} | "
                 f"{e['declared_unit_wrong']} | {e['wells_with_spikes']} |")
    L += ["", "Correlogram of the target's rank:", "", "| distance km | pairs | correlation |",
          "|---|---|---|"]
    for b in a["spatial"]["correlogram_rank_target"]:
        L.append(f"| {b['from_km']}-{b['to_km']} | {b['pairs']} | {_f(b['correlation'])} |")

    L += ["", "## Cross-validation on development wells", "",
          f"Selected by {cv['criterion']} under {SCHEME_NAMES[cv['select_scheme']]}: "
          f"Tweedie `{param_key(cv['selected']['tweedie'])}`, kNN k = "
          f"{cv['selected']['knn_k']}. Random CV would pick "
          f"`{param_key(cv['selected']['random_cv_would_pick'])}`.", "",
          "| CV scheme | median distance test -> nearest train well (km) | model | Spearman | "
          "s.e. | top-decile capture | MAE bbl | bias ratio |",
          "|---|---|---|---|---|---|---|---|"]
    for s, row in cv["selected_under_each_scheme"].items():
        dist = cv["schemes"][s]["nearest_train_well_km"]["median"]
        for m, e in row.items():
            L.append(f"| {SCHEME_NAMES[s]} | {dist} | {NAMES[m]} | {_f(e['spearman'])} | "
                     f"{_f(e['se_spearman'])} | {_f(e['top_decile_capture'])} | "
                     f"{_f(e['mae_bbl'])} | {_f(e['bias_ratio'])} |")
    o = cv["oracle_dev"]
    L.append(f"| any (nothing fitted) | - | {NAMES['oracle']} | {_f(o['spearman'])} | - | "
             f"{_f(o['top_decile_capture'])} | {_f(o['mae_bbl'])} | {_f(o['bias_ratio'])} |")
    L += ["", "Leave one field out, mean Spearman inside each held-out field (no penalty for "
          "getting a field's level wrong):", "", "| model | pooled Spearman | within-field mean "
          "Spearman |", "|---|---|---|"]
    for m, e in cv["selected_under_each_scheme"]["field"].items():
        L.append(f"| {NAMES[m]} | {_f(e['spearman'])} | {_f(e['within_field_mean_spearman'])} |")
    sh = cv["tweedie_share_of_oracle_spearman"]
    L += ["", f"n = {o['n']} wells per row; top decile = {cv['top_decile_wells']} wells. "
          "LightGBM Tweedie Spearman as a share of the oracle's: " +
          ", ".join(f"{SCHEME_NAMES[k]} {v}" for k, v in sh.items()) + "."]
    q = cv[f"quantile_oof_{cv['select_scheme']}"]
    L += ["", f"Quantile models, out-of-fold under {SCHEME_NAMES[cv['select_scheme']]}: share "
          f"below P90 (10th pct) {q['share_below_q10']}, below P50 {q['share_below_q50']}, "
          f"below P10 (90th pct) {q['share_below_q90']}, P90-P10 coverage "
          f"{q['coverage_central']}.", "",
          "Tweedie grid, pooled out-of-fold Spearman by scheme:", "",
          "| params | " + " | ".join(SCHEME_NAMES[s] for s in cv["schemes"]) + " |",
          "|---|" + "---|" * len(cv["schemes"])]
    grids = [cv["schemes"][s]["tweedie_grid"] for s in cv["schemes"]]
    for i, e in enumerate(grids[0]):
        L.append(f"| {param_key(e['params'])} | " +
                 " | ".join(_f(gg[i]["spearman"]) for gg in grids) + " |")

    L += ["", "## Held-out wells (scored once)", "",
          f"{ho['wells']} wells on {ho['pads']} pads; zero share {ho['zero_share']}; top decile "
          f"= {ho['top_decile_wells']} wells. Standard errors: pad-level bootstrap, "
          f"{ho['bootstrap']['n_boot']} resamples.", "",
          "| model | Spearman | s.e. | top-decile capture | s.e. | MAE bbl | s.e. | bias ratio |",
          "|---|---|---|---|---|---|---|---|"]
    for m, e in ho["models"].items():
        L.append(f"| {NAMES[m]} | {_f(e['spearman'])} | {_f(e['se_spearman'])} | "
                 f"{_f(e['top_decile_capture'])} | {_f(e['se_top_decile_capture'])} | "
                 f"{_f(e['mae_bbl'])} | {_f(e['se_mae_bbl'])} | {_f(e['bias_ratio'])} |")
    qq = ho["quantiles"]
    if ho["models"]["tweedie_random_cv_pick"] == ho["models"]["tweedie"]:
        L += ["", "Random CV picked the same settings, so its row repeats the selected one."]
    L += ["", f"LightGBM Tweedie Spearman as a share of the oracle's: "
          f"{ho['tweedie_share_of_oracle_spearman']}."]
    L += ["", f"Random picking would capture about {ho['random_pick_capture_expected']} of the "
          "top decile's oil.", "",
          f"Quantiles on held-out: share below P90 {qq['share_below_q10']}, below P50 "
          f"{qq['share_below_q50']}, below P10 {qq['share_below_q90']}; P90-P10 coverage "
          f"{qq['coverage_central']}; mean width {qq['mean_width_bbl']:,.0f} bbl.", "",
          "Mean |SHAP| on held-out (log scale), top 10:", "", "| feature | mean abs SHAP |",
          "|---|---|"]
    for k, v in ho["mean_abs_shap_top10"].items():
        L.append(f"| {k} | {v} |")

    if fz is not None:
        L += ["", "## FORCE 2020 public logs (real data, audit only)", "",
              f"{fz['wells']} wells, {fz['source']['licence']}, DOI {fz['source']['doi']}. "
              "Interval: first to last gamma-ray sample.", "",
              "| requirement | wells (coverage >= 0.8) | wells (coverage >= 0.5) |",
              "|---|---|---|"]
        for s8, s5 in zip(fz["logs"]["survival"], fz["survival_at_min_coverage_0_5"]):
            L.append(f"| {s8['requires']} | {s8['wells']} | {s5['wells']} |")
        gl = fz["gr_sand_flag_vs_lithology"]
        L += ["", f"GR clean flag vs interpreted lithology ({gl['samples']:,} samples): recall "
              f"of sandstone {gl['recall_sandstone']}, precision {gl['precision_sandstone']}, "
              f"per-well sand-fraction Spearman {gl['spearman_well_sand_fraction']}.",
              f"Nearest-neighbour well distance: median "
              f"{fz['spatial']['nearest_neighbour_km']['median']} km."]
    return "\n".join(L) + "\n"
