"""Test tide accessor."""
from pathlib import Path
import pytest
import datetime
import xarray as xr

from oceantide import read_oceantide


FILES_DIR = Path(__file__).parent / "test_files"


@pytest.fixture(scope="module")
def dset():
    filename = FILES_DIR / "oceantide.zarr"
    _dset = read_oceantide(filename)
    yield _dset


def test_accessor_created(dset):
    assert hasattr(dset, "tide")
    assert hasattr(dset.tide, "predict")


def test_predict_elevation_only(dset):
    times = [datetime.datetime(2001, 1, 1, H) for H in range(6)]
    eta = dset.tide.predict(times, components=["h"])
    assert "u" not in eta and "v" not in eta and "h" in eta


def test_predict_current_only(dset):
    times = [datetime.datetime(2001, 1, 1, H) for H in range(6)]
    eta = dset.tide.predict(times, components=["u", "v"])
    assert "h" not in eta and "u" in eta and "v" in eta


def test_always_predict_something(dset):
    times = [datetime.datetime(2001, 1, 1, H) for H in range(6)]
    with pytest.raises(ValueError):
        dset.tide.predict(times, components=[])
