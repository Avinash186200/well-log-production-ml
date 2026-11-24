import numpy as np
import pytest

from wellprod.las import LasError, read_las, write_las


def _sample():
    data = np.array([[100.0, 50.0, 2.4], [100.5, np.nan, 2.5], [101.0, 70.0, 2.6]])
    return write_las({"WELL": ("", "W1"), "XKM": ("KM", "12.5")},
                     [("DEPT", "M", "depth"), ("GR", "GAPI", "gamma"), ("RHOB", "G/C3", "den")],
                     data)


def test_round_trip_keeps_values_and_turns_null_into_nan():
    text = _sample()
    assert "\r\n" not in text
    las = read_las(text)
    assert las.mnemonics == ["DEPT", "GR", "RHOB"]
    assert las.curves[1].unit == "GAPI"
    assert las.well_value("WELL") == "W1"
    assert las.well["XKM"].unit == "KM" and las.well_value("XKM") == "12.5"
    assert np.isnan(las.curve("GR")[1])
    np.testing.assert_allclose(las.curve("RHOB"), [2.4, 2.5, 2.6])


def test_header_value_may_contain_colons_and_mnemonic_spaces():
    text = ("~V\n VERS. 2.0 : v\n WRAP. NO : w\n~W\n NULL . -999.25 :\n"
            " DATE.  2020-08-09 20:01:10   : Log Export Date\n"
            "~C\nDEPT .m  : DEPTH\nGR .gAPI : GR\n~A\n 1.0 -999.25\n 2.0 30.0\n")
    las = read_las(text)
    assert las.well_value("DATE") == "2020-08-09 20:01:10"
    assert las.mnemonics == ["DEPT", "GR"] and las.curves[0].unit == "m"
    assert np.isnan(las.curve("GR")[0]) and las.curve("GR")[1] == 30.0


@pytest.mark.parametrize("text,msg", [
    ("~V\n WRAP. YES : w\n~C\nDEPT.M : d\n~A\n1.0\n", "wrapped"),
    ("~V\n WRAP. NO : w\n~C\nDEPT.M : d\nGR.API : g\n~A\n1.0\n", "data columns"),
    ("~V\n WRAP. NO : w\n~C\nDEPT.M : d\n", "no ~A"),
    ("~V\n WRAP. NO : w\n~A\n1.0\n", "no ~C"),
    ("~V\n WRAP. NO : w\n~C\nDEPT.M : d\n~A\n1.0\nabc\n", "bad data"),
])
def test_unreadable_files_raise_instead_of_half_reading(text, msg):
    with pytest.raises(LasError, match=msg):
        read_las(text)
