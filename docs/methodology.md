# Methodology

The rules behind every number in the README, and what was decided where.

## Data

### Synthetic basin (`wellprod.synth`, seed 7)

- 9 fields placed at least 24 km apart on a 120 x 80 km map; 12 to 18 pads per field within 9 km of the centre; 3 to 8 wells per pad, lateral midpoints 0.3 km apart, about 1 km from the pad along a random azimuth.
- Geology: six smooth random fields (random Fourier features of a squared-exponential kernel, length scale 7 km) for mean porosity, net-to-gross, water saturation, zone thickness, depth, and fracture intensity. The fracture field changes production but is not in any log, so it can only be learnt through location.
- Pilot log per well: the target zone plus 15 m of shale above and below, 0.5 m steps. Sand and shale beds of random thickness; GR, RHOB, NPHI, RT and DT from the bed properties with textbook relations (RT from Archie with Rw = 0.05 ohm.m) and noise.
- Defects: three vendors with different mnemonics and units; no neutron log on half of the pre-2017 wells; sonic on about 40% of wells; 4% of density curves in kg/m3 (half still labelled g/cm3); 5% of neutron curves in percent labelled V/V; 10% of sonic curves in us/m; spikes on 1% of samples in 20% of wells; NULL gaps in density in 5% of wells; 5% of logs start 5 to 15 m inside the zone; missing proppant (5%), fluid (15%), depth (2%); 1% of laterals entered as 0.
- Completion trends with year (longer laterals, more proppant, tighter stages) with a field-level style.
- Production: log of the mean = 11 + 0.8 log(HCPV / 0.8 m) + 0.7 log(lateral / 2000 m) + a concave proppant effect that is stronger in low-porosity rock - a small stage-spacing effect + 0.3 x fracture + a pad effect (sd 0.2). A well is a zero (dry or failed) with probability sigmoid(-3.2 - 1.2 log(HCPV / 0.8)) + 0.02; otherwise 12-month oil is Gamma with that mean and shape 8. The oracle is the true expected value, mean x (1 - P(zero)).
- Post-completion columns (first-30-day rate, peak month, 6-month cumulative, 3-month water cut) are generated from the outcome on purpose, as leakage bait.
- Pads near a field centre are drilled first: year = 2014 + 10 (r / R)^2 + noise, so years are spread evenly and outer pads are later.

### Split

| split | rule | wells | pads |
|---|---|---|---|
| development | completed 2014-2021 | 578 | 112 |
| held-out | completed 2022-2024 | 140 | 25 |

A pad is completed in one year, so no pad straddles the split, and all held-out wells are on pads with no development well (`results/data.json` -> `holdout.new_pads_share`). The cut-off year was set in the generator config before any model was run.

### FORCE 2020 (real logs)

Source: Bormann, Aursand, Dilib, Manral and Dischington (2020), "FORCE 2020 Well well log and lithofacies dataset for machine learning competition" (v1), Zenodo, https://doi.org/10.5281/zenodo.4351156. Licence CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). It contains data under the Norwegian licence for Open Government data (NLOD) 2.0 (https://data.norge.no/nlod/en/2.0), distributed by the Norwegian government; the Zenodo record gives the licence as "NOLD 2.0" and does not name the agency. `wellprod force-download` fetches the 169,785,172-byte zip and checks the MD5 published by Zenodo (`da0d72d5...`) and a SHA-256 I computed after the first download (`042ef1d4...`; full values in `force2020.py`). `force-sample` parses all 118 LAS files and writes one row per well to `samples/force2020_wells.csv`; `force-report` (and `make bench`) build `results/force2020.json` from that file only, so CI needs no download. All 118 files parsed; none failed.

The FORCE data has no production, so it is used for the audit and the lithology check only.

## Readiness audit

Run on the development wells only (`results/audit.json`).

- A curve "covers the zone" when finite samples fill at least 80% of the expected number of samples between zone top and base (interval length / median step + 1). A log that starts inside the zone, or has a NULL gap, loses coverage.
- Survival chain: wells covering the zone with GR, then also RHOB, RT, NPHI, DT (the order of how essential each is for the petrophysics).
- Units are judged from the median value as well as the label: density with a median above 100 is kg/m3; neutron with a median above 1.5 is percent; sonic with a median above 230 is us/m. A label that disagrees with the values is counted as `declared_unit_wrong`.
- Spikes: samples outside a plausible range after unit conversion (GR 0-500 gAPI, RHOB 1.5-3.1 g/cm3, NPHI -0.05-0.8, RT 0.05-5000 ohm.m, DT 35-200 us/ft) become NaN and are counted. Range checks catch gross spikes only; a spike that stays in range is not caught.
- When a file has several aliases for one curve, the one with the most finite samples wins (ties by a fixed precedence); curves with no finite sample are listed as empty.
- Leakage: a column is a candidate if the data dictionary says it is known only after completion, or if its absolute rank correlation with the target exceeds 0.9. Candidates from the dictionary are excluded from features; a data-only flag (none here) would need a human decision.
- Spatial: correlogram = mean product of standardised target ranks over well pairs in each distance band; Moran's I with row-standardised 5-nearest-neighbour weights and a one-sided permutation p-value (199 permutations; 0.005 is the smallest possible). Both use ranks because the zeros would dominate a log-scale correlation.

## Features

Computed per well over its own target zone; nothing is pooled across wells, so the feature row of a held-out well uses only that well's data (`test_features_exclude_post_completion_columns_and_are_row_independent`).

- For each of GR, RHOB, NPHI, log10 RT, DT: coverage, P10, P50, P90, mean, standard deviation. A curve under 80% coverage gives NaN statistics.
- Petrophysics when GR, RHOB and RT cover the zone: Vsh from the GR index (endpoints 25 and 125 gAPI), porosity from density (matrix 2.65, fluid 1.0 g/cm3) averaged with neutron where present, Archie Sw (a = 1, m = n = 2, Rw = 0.05), net-to-gross (Vsh < 0.4), net pay (Vsh < 0.4, phi > 0.06, Sw < 0.6), hydrocarbon pore thickness. The constants are fixed in advance; they match the generator, which on real data would be the petrophysicist's job.
- Completion: lateral length (0 or less -> NaN), proppant and fluid per metre, stage spacing, completion year. Location: x, y, true vertical depth.
- 45 features in all (`results/data.json` -> `features`). LightGBM handles NaN natively; nothing is imputed.

## Validation and model selection

- Four CV schemes on the 578 development wells, all with seed 0: random 5-fold; 5-fold grouped by pad; 5-fold grouped by 10 km x 10 km map cells, each pad assigned to the cell of its centroid so no pad is split; leave one field out (9 folds). Grouped folds are filled by putting each group (largest first, after a seeded shuffle) into the fold with the fewest wells.
- Metrics are computed on the pooled out-of-fold predictions (one prediction per well), so every scheme is scored on the same 578 wells. Per-fold averages would not be comparable: in leave-field-out each fold is one field, and a ranking inside one field has a narrower spread than across fields. Pooling also penalises a model that gets a new field's level wrong even if it ranks wells inside that field well, so leave-field-out also reports the mean within-field Spearman (`within_field_mean_spearman`).
- Selection: the LightGBM Tweedie grid (num_leaves 4 or 15, min_data_in_leaf 10 or 40, rounds 100 or 300, variance power 1.2 or 1.5; learning rate 0.05, feature fraction 0.8, L2 1.0) and kNN k (3, 5, 10, 20) are chosen by the highest pooled out-of-fold Spearman under spatial-block CV; ties go to the earlier grid entry. The grid is scored under all four schemes, but only the block-CV score chooses.
- The block size (10 km) was set from the development wells' correlogram, which is near zero beyond 10 km.
- The log1p model and the quantile models reuse the selected Tweedie tree settings; they are not tuned separately.
- LightGBM runs with one thread, `deterministic=True`, `force_col_wise=True` and seed 7, so results repeat across machines with the same library versions. The full rerun test passed on Windows and in a `python:3.11-slim` Linux container.

## Held-out evaluation

The final models are refitted on all development wells and score the held-out wells once, in one run, for a fixed list of models: type curve, kNN (k from block CV), LightGBM Tweedie (block-CV settings), LightGBM Tweedie with the settings random CV would have picked (shown only to measure what the wrong CV would have cost; here it is the same setting), LightGBM on log1p(y), LightGBM Tweedie without x and y, and the oracle.

History: the generator settings (zero-rate reference, pad years, the 2022 cut-off) were last changed while looking only at the target's shape and the split sizes, before any model was fitted. A first exploratory grid run looked at development wells only. The first full run produced the held-out numbers reported here; later runs added reporting only (rank-based correlogram, oracle on development wells, bootstrap s.e. for CV rows, completion medians by split, shares of the oracle) and a LightGBM training flag that leaves predictions unchanged (the full rerun test passed before and after). The models, grid, selection rule, features and split did not change after the held-out wells were first scored. One CV change came after held-out scoring, prompted by an audit: spatial blocks were first assigned per well, which split 7 of 112 development pads across cells and left 14 block-CV test wells with a pad-mate in training. Blocks are now assigned by pad centroid. The full run was repeated, which rescored the held-out wells; block CV still selected the same LightGBM settings and kNN k, so the held-out numbers did not change. Block-CV rows changed slightly (LightGBM Spearman 0.711 to 0.709, kNN 0.477 to 0.458).

## Metrics and standard errors

- Spearman: rank correlation with average ranks for ties (the zeros tie).
- Top-decile capture: actual oil of the model's top 10% (ties broken by row order) / actual oil of the true top 10%. 58 wells in CV, 14 on held-out.
- MAE in barrels; bias ratio = sum of predictions / sum of actuals.
- Quantile calibration: share of actuals strictly below each predicted percentile (y < q). With exact zeros, a predicted percentile of 0 counts zero wells as below it, so the lower-tail shares are slightly understated where P90 is 0; coverage of the P90-P10 range. Predicted quantiles are sorted per well so they never cross.
- Standard errors: 1,000 bootstrap resamples of whole pads, seed 0. For held-out rows the models are fixed and only wells are resampled. For CV rows the pooled out-of-fold predictions are resampled; the models are not refitted, so this leaves out refitting noise.
- Mean |SHAP|: LightGBM's built-in TreeSHAP (`pred_contrib=True`) on the held-out wells, on the raw (log) scale.

## Leakage controls

| risk | control | where tested |
|---|---|---|
| post-completion columns as features | excluded by the data dictionary; flagged again by rank correlation | `test_audit_counts_and_flags`, `test_features_exclude_post_completion_columns_and_are_row_independent` |
| held-out wells influencing choices | the audit, the grid search and kNN k use development wells only; held-out scored once | `test_small_run_end_to_end` checks the audit's well count equals the development count and that the selection is the best block-CV score |
| neighbours on both sides of a split | grouped, block and field schemes; distances to the nearest training well reported per scheme | `test_grouped_schemes_push_test_wells_further_from_training` |
| features borrowing other wells' data | per-well features only, fixed petrophysical constants, no cross-well imputation | `test_features_exclude_post_completion_columns_and_are_row_independent` |
| a pad split across development and held-out | split by completion year, one year per pad | `test_split_by_completion_year_and_whole_pads` |
