"""Test tide accessor."""
import os
import pytest
import datetime
import xarray as xr

from oceantide import read_otis_merged


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


@pytest.fixture(scope="module")
def dset():
    filename = os.path.join(FILES_DIR, "otis_merged.nc")
    _dset = read_otis_merged(filename)
    yield _dset

@pytest.fixture(scope="function")
def computed_dset():
    filename = os.path.join(
        FILES_DIR, "otis_merged_tide_timeseries_hourly_2001-01-01T00z_2001-01-01T06z.nc"
    )
    _dset = xr.open_dataset(filename)
    return _dset


def test_accessor_created(dset):
    assert hasattr(dset, "tide")
    assert hasattr(dset.tide, "predict")


def test_predict(dset, computed_dset):
    times = [datetime.datetime(2001, 1, 1, H) for H in range(6)]
    eta = dset.tide.predict(times)
    assert eta.equals(computed_dset)

def test_predict_elevation_only(dset):
    times = [datetime.datetime(2001, 1, 1, H) for H in range(6)]
    eta = dset.tide.predict(times, tide_vars=["et"])
    assert "ut" not in eta and "vt" not in eta and "et" in eta

def test_predict_current_only(dset):
    times = [datetime.datetime(2001, 1, 1, H) for H in range(6)]
    eta = dset.tide.predict(times, tide_vars=["ut", "vt"])
    assert "et" not in eta and "ut" in eta and "vt" in eta

def test_always_predict_something(dset):
    times = [datetime.datetime(2001, 1, 1, H) for H in range(6)]
    with pytest.raises(ValueError):
        dset.tide.predict(times, tide_vars=[])

def test_dimension_alignment(dset,computed_dset):
    lons=xr.DataArray(dset['lon'][:5].values,dims="s")
    lats=xr.DataArray(dset['lat'][:5].values,dims="s")
    times = xr.DataArray([datetime.datetime(2001, 1, 1, H) for H in range(5)],dims="s")
    eta=dset.sel(lon=lons,lat=lats).tide.predict(times)
    assert (eta['et'].values==computed_dset.sel(lon=lons,lat=lats,time=times)['et'].values).all()
