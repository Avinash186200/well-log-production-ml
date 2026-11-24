# Walkthrough

A tour of the pipeline in the order it runs, with the commands to try each part on its own.

## 1. Make a basin and look at it

```bash
pip install -c constraints.txt -e ".[dev]"
wellprod simulate --seed 7 --out data/synthetic
```

This writes `data/synthetic/wells.csv` (one row per well: ids, location, zone depths, completion, the post-completion columns and the target) and `data/synthetic/las/W0001.las` and so on. Open a LAS file: it has a `~Well` section with the well name and map position, a `~Curve` section, and the data table. Different pads use different vendors, so you will see `GR`, `GRC` or `SGR` for gamma ray, and `NPHI` in V/V in one file but `NEU` in PU in another.

In Python:

```python
from wellprod.synth import simulate_basin
from wellprod.las import read_las
from wellprod.curves import from_las

basin = simulate_basin(7)
las = read_las(basin.las["W0001"])
logs = from_las(las)              # canonical names, canonical units, spikes -> NaN
logs.fixes["RHOB"]                # which raw curve, its unit, any conversion, spike count
```

## 2. Audit before modelling

```python
from wellprod.audit import audit
from wellprod.synth import DATA_DICTIONARY

dev = basin.wells[basin.wells.split == "dev"]
logs = {w: from_las(read_las(basin.las[w])) for w in dev.well_id}
report = audit(dev, logs, DATA_DICTIONARY)
report["summary"]["logs"]["survival"]       # wells left as you require more curves
report["summary"]["leakage"]                # post-completion columns, by dictionary and by data
report["inventory"]                          # one row per well: coverage, alias, unit, spikes
```

The questions to answer here, before any model:

- How many wells survive the curve set the features need? (539 of 578 for GR + RHOB + RT; 174 if you also demand sonic.)
- Which columns could not exist for a planned well? (Anything measured on production.)
- How clustered are the wells? (Pads of 3 to 8 wells 300 m apart; outcomes correlated out to about 10 km.)

## 3. Features

```python
from wellprod.features import feature_table
X = feature_table(dev, logs)
```

One row per well. Curve statistics are taken only inside the target zone, and petrophysics only when GR, RHOB and RT cover at least 80% of it. Missing curves stay NaN.

## 4. Cross-validation four ways

```python
import numpy as np
from wellprod.cv import make_splits, nearest_train_distance
d = dev.reset_index(drop=True)
for scheme in ("random", "pad", "block", "field"):
    sp = make_splits(scheme, d, k=5, seed=0, block_km=10.0)
    print(scheme, np.median(nearest_train_distance(d, sp)))
```

The distance from each test well to its nearest training well is the simplest way to see why the schemes disagree: 0.3 km for random folds (a pad-mate), 1.7 km by pad, 4.7 km by 10 km block, 17.9 km for a new field (medians, from `results/cv.json`).

## 5. The full run

```bash
make bench        # about 1.5 minutes on a laptop
```

`wellprod.bench.run` does everything above for all wells, scores the LightGBM grid and the baselines under all four schemes, chooses by spatial-block CV, refits on development wells, scores held-out wells once, computes pad-bootstrap standard errors, and writes `results/*.json` and `results/RESULTS.md`.

## 6. Real logs

```bash
make force        # needs network; ~170 MB; checks MD5 and SHA-256
```

This rebuilds `samples/force2020_wells.csv` from the FORCE 2020 LAS files and `results/force2020.json` from the sample. The committed sample lets CI rebuild the reported FORCE numbers without downloading anything. Things to look at in `results/force2020.json`: the survival chain at 80% and 50% coverage, the spike counts per curve, and `gr_sand_flag_vs_lithology`, which compares a gamma-ray sand flag with the geologists' lithology labels.

## 7. Reading the results

- Start with the CV table in the README: the same model, four estimates. The spread between rows is the cost of ignoring spatial structure.
- Then the held-out table: one honest number per model, with its standard error, and the oracle as the ceiling.
- Then the bias ratio and the quantile coverage: a ranking can be good while the volumes are off.
