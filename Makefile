.PHONY: install test bench report simulate force

install:
	pip install -c constraints.txt -e ".[dev]"

test:
	pytest -q

# Offline, deterministic: simulate, audit, CV four ways on development wells, choose by
# spatial-block CV, score held-out wells once. Writes results/*.json and RESULTS.md.
bench:
	python -m wellprod.cli bench --out results

report:
	python -m wellprod.cli report --out results

# The synthetic basin as wells.csv plus one LAS file per well.
simulate:
	python -m wellprod.cli simulate --seed 7 --out data/synthetic

# Real logs (needs network, ~170 MB): download, verify checksums, rebuild the committed
# per-well sample and results/force2020.json. Never run in CI.
force:
	python -m wellprod.cli force-download --dest data/force2020
	python -m wellprod.cli force-sample --zip data/force2020/LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip --out samples/force2020_wells.csv
	python -m wellprod.cli force-report --sample samples/force2020_wells.csv --out results
