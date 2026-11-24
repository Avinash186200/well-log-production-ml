# Results

Synthetic basin from `wellprod.synth` (seed 7) unless marked FORCE 2020. Rebuild with `make bench`.

## Data

| split | wells | pads | fields | completion years |
|---|---|---|---|---|
| dev | 578 | 112 | 9 | 2014-2021 |
| holdout | 140 | 25 | 9 | 2022-2024 |

| split | median lateral m | median proppant kg/m |
|---|---|---|
| dev | 2,003 | 1,162 |
| holdout | 2,380 | 1,558 |

Held-out wells: 100% are on pads with no development well; median distance to the nearest development well 2.738 km; share above the development maximum: lateral 0.0143, proppant 0.0.

## Audit (development wells)

- target: zero share 0.1176, median 42,176 bbl, p99 225,594 bbl, skew 1.79
- leakage by dictionary: cum6_oil_bbl, ip30_bopd, peak_month_oil_bbl, water_cut_3m
- leakage by data (|rank corr| > 0.9): cum6_oil_bbl, ip30_bopd, peak_month_oil_bbl
- Moran's I of the target's rank, k=5: 0.4989 (p = 0.005)

| requirement | wells |
|---|---|
| GR | 551 |
| GR+RHOB | 539 |
| GR+RHOB+RT | 539 |
| GR+RHOB+RT+NPHI | 431 |
| GR+RHOB+RT+NPHI+DT | 174 |

| curve | aliases | wells covering zone | unit conversions | declared unit wrong | wells with spikes |
|---|---|---|---|---|---|
| GR | GR, GRC, SGR | 551 | 0 | 0 | 29 |
| RHOB | DEN, RHOB, ZDEN | 539 | 17 | 12 | 92 |
| NPHI | NEU, NPHI, TNPH | 439 | 108 | 22 | 46 |
| RT | ILD, LLD, RDEP | 551 | 0 | 0 | 0 |
| DT | AC, DT, DTC | 219 | 26 | 0 | 31 |

Correlogram of the target's rank:

| distance km | pairs | correlation |
|---|---|---|
| 0.0-0.5 | 563 | 0.452 |
| 0.5-1.0 | 873 | 0.502 |
| 1.0-2.0 | 1231 | 0.489 |
| 2.0-5.0 | 4594 | 0.413 |
| 5.0-10.0 | 8442 | 0.343 |
| 10.0-20.0 | 6807 | 0.009 |
| 20.0-40.0 | 42858 | -0.073 |

## Cross-validation on development wells

Selected by pooled out-of-fold Spearman under spatial 10 km blocks (5-fold): Tweedie `min_data_in_leaf=40,num_leaves=15,num_rounds=100,tweedie_variance_power=1.2`, kNN k = 10. Random CV would pick `min_data_in_leaf=40,num_leaves=15,num_rounds=100,tweedie_variance_power=1.2`.

| CV scheme | median distance test -> nearest train well (km) | model | Spearman | s.e. | top-decile capture | MAE bbl | bias ratio |
|---|---|---|---|---|---|---|---|
| random 5-fold | 0.3 | type curve (field median per m x lateral) | 0.581 | 0.045 | 0.649 | 28,747 | 0.896 |
| random 5-fold | 0.3 | kNN on location | 0.641 | 0.044 | 0.755 | 25,780 | 0.981 |
| random 5-fold | 0.3 | LightGBM Tweedie (selected) | 0.760 | 0.025 | 0.822 | 23,406 | 0.961 |
| random 5-fold | 0.3 | LightGBM on log1p(y) | 0.627 | 0.032 | 0.638 | 33,361 | 0.642 |
| random 5-fold | 0.3 | LightGBM Tweedie, no x/y | 0.749 | 0.024 | 0.825 | 24,278 | 0.958 |
| grouped by pad (5-fold) | 1.676 | type curve (field median per m x lateral) | 0.554 | 0.045 | 0.564 | 30,330 | 0.927 |
| grouped by pad (5-fold) | 1.676 | kNN on location | 0.609 | 0.043 | 0.667 | 28,287 | 0.975 |
| grouped by pad (5-fold) | 1.676 | LightGBM Tweedie (selected) | 0.747 | 0.025 | 0.758 | 24,150 | 0.950 |
| grouped by pad (5-fold) | 1.676 | LightGBM on log1p(y) | 0.636 | 0.033 | 0.709 | 32,147 | 0.636 |
| grouped by pad (5-fold) | 1.676 | LightGBM Tweedie, no x/y | 0.737 | 0.024 | 0.750 | 24,862 | 0.955 |
| spatial 10 km blocks (5-fold) | 4.696 | type curve (field median per m x lateral) | 0.521 | 0.051 | 0.516 | 31,412 | 0.922 |
| spatial 10 km blocks (5-fold) | 4.696 | kNN on location | 0.458 | 0.058 | 0.529 | 32,484 | 1.040 |
| spatial 10 km blocks (5-fold) | 4.696 | LightGBM Tweedie (selected) | 0.709 | 0.029 | 0.767 | 25,893 | 0.943 |
| spatial 10 km blocks (5-fold) | 4.696 | LightGBM on log1p(y) | 0.559 | 0.038 | 0.682 | 34,128 | 0.658 |
| spatial 10 km blocks (5-fold) | 4.696 | LightGBM Tweedie, no x/y | 0.703 | 0.029 | 0.795 | 26,183 | 0.938 |
| leave one field out | 17.873 | type curve (field median per m x lateral) | -0.061 | 0.054 | 0.252 | 37,382 | 0.776 |
| leave one field out | 17.873 | kNN on location | -0.190 | 0.079 | 0.173 | 49,103 | 1.108 |
| leave one field out | 17.873 | LightGBM Tweedie (selected) | 0.667 | 0.032 | 0.676 | 28,529 | 1.004 |
| leave one field out | 17.873 | LightGBM on log1p(y) | 0.534 | 0.038 | 0.599 | 35,190 | 0.594 |
| leave one field out | 17.873 | LightGBM Tweedie, no x/y | 0.664 | 0.033 | 0.658 | 28,035 | 0.933 |
| any (nothing fitted) | - | oracle: true expected value | 0.841 | - | 0.866 | 17,792 | 0.998 |

Leave one field out, mean Spearman inside each held-out field (no penalty for getting a field's level wrong):

| model | pooled Spearman | within-field mean Spearman |
|---|---|---|
| type curve (field median per m x lateral) | -0.061 | 0.234 |
| kNN on location | -0.190 | 0.056 |
| LightGBM Tweedie (selected) | 0.667 | 0.546 |
| LightGBM on log1p(y) | 0.534 | 0.381 |
| LightGBM Tweedie, no x/y | 0.664 | 0.546 |

n = 578 wells per row; top decile = 58 wells. LightGBM Tweedie Spearman as a share of the oracle's: random 5-fold 0.9034, grouped by pad (5-fold) 0.8887, spatial 10 km blocks (5-fold) 0.8436, leave one field out 0.7934.

Quantile models, out-of-fold under spatial 10 km blocks (5-fold): share below P90 (10th pct) 0.1817, below P50 0.5277, below P10 (90th pct) 0.8478, P90-P10 coverage 0.6661.

Tweedie grid, pooled out-of-fold Spearman by scheme:

| params | random 5-fold | grouped by pad (5-fold) | spatial 10 km blocks (5-fold) | leave one field out |
|---|---|---|---|---|
| min_data_in_leaf=10,num_leaves=4,num_rounds=100,tweedie_variance_power=1.2 | 0.750 | 0.736 | 0.701 | 0.659 |
| min_data_in_leaf=10,num_leaves=4,num_rounds=100,tweedie_variance_power=1.5 | 0.752 | 0.742 | 0.700 | 0.659 |
| min_data_in_leaf=10,num_leaves=4,num_rounds=300,tweedie_variance_power=1.2 | 0.743 | 0.745 | 0.692 | 0.671 |
| min_data_in_leaf=10,num_leaves=4,num_rounds=300,tweedie_variance_power=1.5 | 0.748 | 0.749 | 0.697 | 0.664 |
| min_data_in_leaf=40,num_leaves=4,num_rounds=100,tweedie_variance_power=1.2 | 0.750 | 0.734 | 0.701 | 0.651 |
| min_data_in_leaf=40,num_leaves=4,num_rounds=100,tweedie_variance_power=1.5 | 0.750 | 0.740 | 0.699 | 0.651 |
| min_data_in_leaf=40,num_leaves=4,num_rounds=300,tweedie_variance_power=1.2 | 0.755 | 0.748 | 0.705 | 0.668 |
| min_data_in_leaf=40,num_leaves=4,num_rounds=300,tweedie_variance_power=1.5 | 0.752 | 0.751 | 0.705 | 0.664 |
| min_data_in_leaf=10,num_leaves=15,num_rounds=100,tweedie_variance_power=1.2 | 0.756 | 0.747 | 0.703 | 0.660 |
| min_data_in_leaf=10,num_leaves=15,num_rounds=100,tweedie_variance_power=1.5 | 0.748 | 0.749 | 0.697 | 0.660 |
| min_data_in_leaf=10,num_leaves=15,num_rounds=300,tweedie_variance_power=1.2 | 0.735 | 0.726 | 0.693 | 0.655 |
| min_data_in_leaf=10,num_leaves=15,num_rounds=300,tweedie_variance_power=1.5 | 0.724 | 0.732 | 0.679 | 0.652 |
| min_data_in_leaf=40,num_leaves=15,num_rounds=100,tweedie_variance_power=1.2 | 0.760 | 0.747 | 0.709 | 0.667 |
| min_data_in_leaf=40,num_leaves=15,num_rounds=100,tweedie_variance_power=1.5 | 0.756 | 0.744 | 0.707 | 0.643 |
| min_data_in_leaf=40,num_leaves=15,num_rounds=300,tweedie_variance_power=1.2 | 0.743 | 0.743 | 0.705 | 0.681 |
| min_data_in_leaf=40,num_leaves=15,num_rounds=300,tweedie_variance_power=1.5 | 0.742 | 0.744 | 0.693 | 0.639 |

## Held-out wells (scored once)

140 wells on 25 pads; zero share 0.0786; top decile = 14 wells. Standard errors: pad-level bootstrap, 1000 resamples.

| model | Spearman | s.e. | top-decile capture | s.e. | MAE bbl | s.e. | bias ratio |
|---|---|---|---|---|---|---|---|
| type curve (field median per m x lateral) | 0.345 | 0.102 | 0.455 | 0.092 | 41,508 | 5,567 | 0.811 |
| kNN on location | 0.361 | 0.115 | 0.397 | 0.101 | 41,692 | 5,836 | 0.808 |
| LightGBM Tweedie (selected) | 0.587 | 0.066 | 0.666 | 0.070 | 34,286 | 4,664 | 0.893 |
| LightGBM Tweedie (random-CV pick) | 0.587 | 0.066 | 0.666 | 0.070 | 34,286 | 4,664 | 0.893 |
| LightGBM on log1p(y) | 0.523 | 0.079 | 0.652 | 0.112 | 45,300 | 6,209 | 0.606 |
| LightGBM Tweedie, no x/y | 0.595 | 0.061 | 0.750 | 0.072 | 33,032 | 4,364 | 0.894 |
| oracle: true expected value | 0.763 | 0.050 | 0.941 | 0.042 | 22,380 | 2,600 | 0.981 |

Random CV picked the same settings, so its row repeats the selected one.

LightGBM Tweedie Spearman as a share of the oracle's: 0.7687.

Random picking would capture about 0.3374 of the top decile's oil.

Quantiles on held-out: share below P90 0.1, below P50 0.5214, below P10 0.8143; P90-P10 coverage 0.7143; mean width 78,129 bbl.

Mean |SHAP| on held-out (log scale), top 10:

| feature | mean abs SHAP |
|---|---|
| RHOB_mean | 0.1894 |
| hcpv_m | 0.1377 |
| lateral_m | 0.1029 |
| x_km | 0.0783 |
| RHOB_p10 | 0.0742 |
| proppant_kg_m | 0.0723 |
| log10RT_mean | 0.06 |
| zone_thickness_m | 0.051 |
| tvd_m | 0.05 |
| completion_year | 0.0429 |

## FORCE 2020 public logs (real data, audit only)

118 wells, CC BY 4.0 (underlying government data: NLOD 2.0), DOI 10.5281/zenodo.4351156. Interval: first to last gamma-ray sample.

| requirement | wells (coverage >= 0.8) | wells (coverage >= 0.5) |
|---|---|---|
| GR | 118 | 118 |
| GR+RHOB | 34 | 82 |
| GR+RHOB+RT | 34 | 82 |
| GR+RHOB+RT+NPHI | 13 | 53 |
| GR+RHOB+RT+NPHI+DT | 12 | 51 |

GR clean flag vs interpreted lithology (1,429,611 samples): recall of sandstone 0.7389, precision 0.2975, per-well sand-fraction Spearman 0.1005.
Nearest-neighbour well distance: median 7.73 km.
