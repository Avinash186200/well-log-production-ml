import json
import shutil
from pathlib import Path

from wellprod.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_simulate_writes_wells_and_las(tmp_path, capsys):
    assert main(["simulate", "--seed", "3", "--out", str(tmp_path)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["wells"] == stats["las_files"] > 100
    wells = (tmp_path / "wells.csv").read_bytes()
    assert wells.startswith(b"well_id,pad_id,field_id,") and b"\r\n" not in wells
    one = sorted((tmp_path / "las").glob("*.las"))[0].read_bytes()
    assert one.startswith(b"~VERSION") and b"\r\n" not in one


def test_report_rerenders_committed_results(tmp_path):
    for p in (ROOT / "results").glob("*.json"):
        shutil.copy(p, tmp_path / p.name)
    assert main(["report", "--out", str(tmp_path)]) == 0
    fresh = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert fresh == (ROOT / "results" / "RESULTS.md").read_text(encoding="utf-8")


def test_force_report_from_sample(tmp_path):
    assert main(["force-report", "--sample", str(ROOT / "samples" / "force2020_wells.csv"),
                 "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "force2020.json").read_text(encoding="utf-8")) == \
        json.loads((ROOT / "results" / "force2020.json").read_text(encoding="utf-8"))
