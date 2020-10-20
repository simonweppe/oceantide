"""Test tide accessor."""
import os
import pytest
import datetime
import xarray as xr

from oceantide import Tide
from oceantide import read_otis_merged, read_otis_netcdf, read_otis_binary


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


def test_read_otis_merged():
    ncfile = os.path.join(FILES_DIR, "otis_merged.nc")
    zarrfile = os.path.join(FILES_DIR, "otis_merged.zarr")
    dset1 = read_otis_merged(ncfile)
    dset2 = read_otis_merged(zarrfile, file_format="zarr")
    assert dset1.equals(dset2)


def test_read_otis_binary():
    read_otis_binary(os.path.join(FILES_DIR, "otis_binary/Model_rag"))


def test_read_otis_binary():
    read_otis_binary(os.path.join(FILES_DIR, "otis_binary/Model_rag"))


def test_read_otis_netcdf():
    read_otis_netcdf(os.path.join(FILES_DIR, "otis_netcdf/Model_test"))
