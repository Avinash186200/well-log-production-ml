# Contributing

Thanks for taking a look. Changes are judged on whether they keep the repo offline, leak-free and honest about its numbers.

## Ground rules

1. **Everything in CI runs offline and on CPU.** Tests and `make bench` need no network. Only `make force` downloads anything, and CI never runs it.
2. **Numbers come from results.** Any number in a doc must be in a `results/*.json` file written by `make bench` (or `force-report`). If you change code that affects results, rerun `make bench` and commit the new JSON and `RESULTS.md` with the change. CI fails if the committed JSON does not match a fresh run.
3. **Never tune on the held-out wells.** Features, grids, kNN k, block size and the selection rule are chosen on development wells. If a change needs a new choice, make it on development wells and say in `docs/methodology.md` that the held-out wells have now been seen more than once.
4. **Features use the well's own data only.** No cross-well imputation, no statistics fitted on all wells, no post-completion columns. `test_features_exclude_post_completion_columns_and_are_row_independent` must keep passing.
5. **No timing assertions in tests.** CI runners are shared and slow; tests check correctness, never speed.
6. **Seeds stay fixed.** Every generator, split and bootstrap takes a seed; LightGBM runs single-threaded and deterministic.
7. **No large data.** Nothing over 1 MB is committed. Downloaded data goes under `data/` (git-ignored).
8. **No secrets.** Nothing here needs credentials; `.env` is git-ignored and CI runs a secret scan.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
pytest -q
```

## Adding a curve alias

Add the raw mnemonic to `ALIASES` in `curves.py` under its canonical name (order sets the tie-break), add a unit rule in `_convert` if the curve comes in more than one unit, and add a test in `tests/test_curves.py`.

## Adding a model

Add it to `models.py` with `fit` and `predict`, score it under every scheme in `bench.run`, add it to the held-out list, and add a row name in `NAMES`. Rerun `make bench`.
