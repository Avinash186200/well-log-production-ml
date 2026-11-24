"""A small LAS 2.0 reader and writer (unwrapped files only).

LAS is the plain-text well-log format: a few `~` sections of `MNEM.UNIT  VALUE : DESCRIPTION`
lines, then an ASCII table of depth-indexed samples (`~A`). The reader keeps every header
line, returns the curve table as a float array, and turns the file's NULL value into NaN.
It raises `LasError` (never returns a half-read file) on anything it does not understand:
wrapped files, a column count that does not match the curve list, or no `~A` section.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np

_LINE = re.compile(r"^\s*([^.]*?)\s*\.([^\s:]*)\s*(.*)\s*:([^:]*)$")


class LasError(ValueError):
    """The file is not a LAS 2.0 file this reader can read."""


@dataclass
class HeaderItem:
    mnemonic: str
    unit: str
    value: str
    descr: str


@dataclass
class LasFile:
    version: dict[str, HeaderItem] = field(default_factory=dict)
    well: dict[str, HeaderItem] = field(default_factory=dict)
    params: dict[str, HeaderItem] = field(default_factory=dict)
    curves: list[HeaderItem] = field(default_factory=list)
    data: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def mnemonics(self) -> list[str]:
        return [c.mnemonic for c in self.curves]

    def curve(self, mnemonic: str) -> np.ndarray:
        return self.data[:, self.mnemonics.index(mnemonic)]

    def well_value(self, key: str, default: str = "") -> str:
        item = self.well.get(key.upper())
        return item.value if item else default


def _parse_header_line(line: str) -> HeaderItem | None:
    # The unit is whatever follows the first dot up to the first space, and the value runs up
    # to the LAST colon (values such as times may contain colons; descriptions may not).
    m = _LINE.match(line)
    if not m:
        return None
    mnem = (m.group(1) or "").strip()
    return HeaderItem(mnem.upper(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip())


def read_las(text: str) -> LasFile:
    """Parse LAS 2.0 text. Raises LasError on anything unsupported."""
    las = LasFile()
    section = None
    data_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("~"):
            section = line.lstrip()[1:2].upper()
            continue
        if section == "A":
            data_lines.append(line)
            continue
        item = _parse_header_line(line)
        if item is None:
            continue
        if section == "V":
            las.version[item.mnemonic] = item
        elif section == "W":
            las.well[item.mnemonic] = item
        elif section == "P":
            las.params[item.mnemonic] = item
        elif section == "C":
            las.curves.append(item)
    if "WRAP" in las.version and las.version["WRAP"].value.upper().startswith("Y"):
        raise LasError("wrapped LAS files are not supported")
    if not las.curves:
        raise LasError("no ~C (curve) section")
    if section is None or not data_lines:
        raise LasError("no ~A (data) section")
    try:
        arr = np.loadtxt(io.StringIO("\n".join(data_lines)), dtype=float, ndmin=2)
    except ValueError as exc:
        raise LasError(f"bad data table: {exc}") from exc
    if arr.shape[1] != len(las.curves):
        raise LasError(f"{arr.shape[1]} data columns but {len(las.curves)} curves")
    null = las.well.get("NULL")
    if null is not None:
        try:
            nv = float(null.value)
        except ValueError:
            nv = None
        if nv is not None:
            arr[np.isclose(arr, nv)] = np.nan
    las.data = arr
    return las


def write_las(well: dict[str, tuple[str, str]], curves: list[tuple[str, str, str]],
              data: np.ndarray, null: float = -999.25) -> str:
    """LAS 2.0 text. `well` maps mnemonic -> (unit, value); `curves` is (mnemonic, unit, descr).

    The first curve is the depth index. NaN is written as the NULL value. LF line endings.
    """
    out = ["~VERSION INFORMATION",
           " VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0",
           " WRAP.   NO  : ONE LINE PER DEPTH STEP",
           "~WELL INFORMATION"]
    depth = data[:, 0]
    std = {"STRT": ("M", f"{depth[0]:.4f}"), "STOP": ("M", f"{depth[-1]:.4f}"),
           "STEP": ("M", f"{depth[1] - depth[0]:.4f}" if len(depth) > 1 else "0"),
           "NULL": ("", f"{null}")}
    for k, (u, v) in {**std, **well}.items():
        out.append(f" {k}.{u} {v} : ")
    out.append("~CURVE INFORMATION")
    for mnem, unit, descr in curves:
        out.append(f" {mnem}.{unit}  : {descr}")
    out.append("~ASCII")
    filled = np.where(np.isnan(data), null, data)
    for row in filled:
        out.append(" ".join(f"{v:.4f}" for v in row))
    return "\n".join(out) + "\n"
