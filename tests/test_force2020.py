import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wellprod.force2020 import SANDSTONE, summary_from_sample, verify, well_row
from wellprod.las import write_las

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "force2020_wells.csv"


def test_committed_sample_is_small_lf_and_rebuilds_reported_numbers():
    raw = SAMPLE.read_bytes()
    assert len(raw) < 1_000_000 and b"\r\n" not in raw
    fresh = summary_from_sample(pd.read_csv(SAMPLE))
    committed = json.loads((ROOT / "results" / "force2020.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(fresh)) == committed


def test_attribution_notice_ships_with_the_sample():
    text = (ROOT / "samples" / "NOTICE.txt").read_text(encoding="utf-8")
    assert "CC BY 4.0" in text and "10.5281/zenodo.4351156" in text


def test_verify_rejects_a_wrong_file(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="checksum"):
        verify(p)


def _force_like_las():
    d = np.arange(1000.0, 1100.0, 0.152)
    n = len(d)
    gr = np.where(np.arange(n) % 2 == 0, 30.0, 120.0)
    lith = np.where(gr < 50, SANDSTONE, 65000.0)
    gr[:20] = np.nan                           # GR starts late: defines the interval
    cols = [("DEPT", "m", "DEPTH"), ("FORCE_2020_LITHOFACIES_LITHOLOGY", "_", "lith"),
            ("GR", "gAPI", "GR"), ("RHOB", "g/cm3", "RHOB"), ("x_loc", "_", "x"),
            ("y_loc", "_", "y"), ("CALI", "in", "CALI")]
    data = np.column_stack([d, lith, gr, np.full(n, 2.4), np.full(n, 450_000.0),
                            np.full(n, 6_500_000.0), np.full(n, np.nan)])
    return write_las({"WELL": ("", "T-1")}, cols, data), d


def test_well_row_on_a_force_like_file():
    text, d = _force_like_las()
    row = well_row("T-1", text)
    assert row["top_m"] == pytest.approx(d[20], abs=1e-4)
    assert row["x_km"] == 450.0 and row["y_km"] == 6500.0
    assert row["cov_GR"] == 1.0 and row["cov_RHOB"] == 1.0
    assert row["empty_curves"] == "CALI"
    # alternating 30 / 120 gAPI: the clean flag is exactly the sandstone samples
    assert row["n_sand_lith"] == row["n_clean_and_sand"] == row["n_clean_gr"] > 0
    assert "X_LOC" in row["unknown_curves"]
