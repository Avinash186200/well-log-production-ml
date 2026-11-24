"""wellprod simulate | bench | report | force-download | force-sample | force-report"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _simulate(args) -> int:
    from .synth import simulate_basin
    basin = simulate_basin(args.seed)
    out = Path(args.out)
    (out / "las").mkdir(parents=True, exist_ok=True)
    basin.wells.to_csv(out / "wells.csv", index=False, lineterminator="\n")
    for w, text in basin.las.items():
        (out / "las" / f"{w}.las").write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"wells": len(basin.wells), "las_files": len(basin.las),
                      "out": str(out)}))
    return 0


def _bench(args) -> int:
    from .bench import run
    run(Path(args.out))
    return 0


def _report(args) -> int:
    from .bench import RESULT_FILES, render
    out = Path(args.out)
    res = {}
    for k in RESULT_FILES:
        p = out / f"{k}.json"
        res[k] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    (out / "RESULTS.md").write_text(render(res), encoding="utf-8", newline="\n")
    return 0


def _force_download(args) -> int:
    from .force2020 import download
    path = download(Path(args.dest))
    print(f"verified {path}")
    return 0


def _force_sample(args) -> int:
    from .force2020 import summarise_zip, verify
    zp = Path(args.zip)
    verify(zp)
    df, failed = summarise_zip(zp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    print(json.dumps({"wells": len(df), "failed": failed, "out": str(out)}))
    return 0 if not failed else 1


def _force_report(args) -> int:
    import pandas as pd

    from .bench import _write_json
    from .force2020 import summary_from_sample
    res = summary_from_sample(pd.read_csv(args.sample))
    _write_json(Path(args.out) / "force2020.json", res)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wellprod")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="write the synthetic basin as wells.csv + LAS files")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="data/synthetic")
    s.set_defaults(fn=_simulate)
    b = sub.add_parser("bench", help="the full offline run; writes results/*.json")
    b.add_argument("--out", default="results")
    b.set_defaults(fn=_bench)
    r = sub.add_parser("report", help="re-render RESULTS.md from results/*.json")
    r.add_argument("--out", default="results")
    r.set_defaults(fn=_report)
    d = sub.add_parser("force-download", help="download FORCE 2020 LAS zip (~170 MB), verify")
    d.add_argument("--dest", default="data/force2020")
    d.set_defaults(fn=_force_download)
    fs = sub.add_parser("force-sample", help="per-well audit rows from the verified zip")
    fs.add_argument("--zip", required=True)
    fs.add_argument("--out", default="samples/force2020_wells.csv")
    fs.set_defaults(fn=_force_sample)
    fr = sub.add_parser("force-report", help="results/force2020.json from the sample")
    fr.add_argument("--sample", default="samples/force2020_wells.csv")
    fr.add_argument("--out", default="results")
    fr.set_defaults(fn=_force_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
