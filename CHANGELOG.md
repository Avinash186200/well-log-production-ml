# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-24

### Added

- Synthetic basin generator: 9 fields of pads of horizontal wells, smooth random geology fields, a hidden fracture field and pad effect, step-out drilling order, LAS 2.0 pilot logs with vendor aliases, mixed and mislabelled units, spikes, NULL gaps, missing curves and partial zone coverage, completion design with a technology trend, a zero-inflated Gamma 12-month oil target, and post-completion leakage columns.
- LAS 2.0 reader and writer that raises on unsupported files instead of half-reading them.
- Curve canonicalisation: alias map, value-based unit checks with mislabel counts, plausible-range spike flags, zone coverage.
- Data-readiness audit: missing and impossible values, curve survival chain, unit fixes, spikes, target shape, leakage candidates by data dictionary and by rank correlation, wells per pad, correlogram and Moran's I.
- Per-well features: zone statistics for five curves, shale volume, porosity, Archie saturation, net pay, hydrocarbon pore thickness, completion and location.
- Random, pad-grouped, spatial-block and leave-field-out CV; selection by spatial-block CV on development wells; held-out wells scored once.
- Type curve and kNN-on-location baselines; LightGBM with Tweedie, log1p and quantile objectives; Spearman, top-decile capture, MAE, bias ratio, quantile calibration, pad-bootstrap standard errors, TreeSHAP importances.
- FORCE 2020 public logs: download with MD5 and SHA-256 checks, per-well audit rows, a committed 60 KB derived sample with CC BY 4.0 notice, a gamma-ray sand flag check against interpreted lithology.
- Tests including a full rerun against the committed results, `constraints.txt` with exact library versions, CI on ubuntu (Python 3.11) with a secret scan.
