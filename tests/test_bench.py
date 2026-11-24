import json
import math
from pathlib import Path

import pytest

from wellprod.bench import RESULT_FILES, run

from conftest import SMALL

ROOT = Path(__file__).resolve().parents[1]
TINY_GRID = [{"num_leaves": 4, "min_data_in_leaf": 10, "num_rounds": 50,
              "tweedie_variance_power": 1.5},
             {"num_leaves": 4, "min_data_in_leaf": 20, "num_rounds": 50,
              "tweedie_variance_power": 1.2}]


def test_small_run_end_to_end(tmp_path):
    res = run(tmp_path, seed=11, cfg=SMALL, log=lambda *_: None, grid_override=TINY_GRID,
              n_boot=20)
    for k in RESULT_FILES:
        assert (tmp_path / f"{k}.json").exists()
    assert (tmp_path / "RESULTS.md").read_text(encoding="utf-8").startswith("# Results")
    cv = res["cv"]
    assert cv["selected"]["tweedie"] in TINY_GRID
    # the selection is the best pooled OOF Spearman under the spatial-block scheme
    blk = cv["schemes"]["block"]["tweedie_grid"]
    assert cv["selected"]["tweedie"] == max(blk, key=lambda e: e["spearman"])["params"]
    ho = res["holdout"]
    assert ho["wells"] == res["data"]["holdout"]["wells"]
    assert all(m["n"] == ho["wells"] for m in ho["models"].values())
    # nothing about held-out wells reaches the audit
    assert res["audit"]["wells"] == res["data"]["dev"]["wells"]


def test_results_written_with_lf_only():
    for k in RESULT_FILES + ("RESULTS",):
        name = f"{k}.md" if k == "RESULTS" else f"{k}.json"
        assert b"\r\n" not in (ROOT / "results" / name).read_bytes()


def _diff(a, b, path="$"):
    """Differences between two JSON trees; floats match to 1e-6 relative, 1e-9 absolute."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
        return [] if ok else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    run(out, log=lambda *_: None)
    return out


@pytest.mark.parametrize("name", RESULT_FILES)
def test_committed_results_reproduce(full_run, name):
    fresh = json.loads((full_run / f"{name}.json").read_text(encoding="utf-8"))
    committed = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
    diffs = _diff(fresh, committed)
    assert not diffs, "\n".join(diffs[:20])


def test_committed_results_md_reproduces(full_run):
    fresh = (full_run / "RESULTS.md").read_text(encoding="utf-8")
    assert fresh == (ROOT / "results" / "RESULTS.md").read_text(encoding="utf-8")
