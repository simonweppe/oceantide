"""Test tide accessor."""
import os
import pytest
import datetime
import xarray as xr

from oceantide.tide import Tide
from oceantide import (
    read_otis_merged,
    read_otis_netcdf,
    read_otis_binary,
    read_oceantide,
    read_dataset,
)


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")


def test_read_otis_merged():
    ncfile = os.path.join(FILES_DIR, "otis_merged.nc")
    zarrfile = os.path.join(FILES_DIR, "otis_merged.zarr")
    dset1 = read_otis_merged(ncfile)
    dset2 = read_otis_merged(zarrfile, file_format="zarr")
    assert dset1.equals(dset2)


def test_read_otis_binary_with_filename():
    read_otis_binary(os.path.join(FILES_DIR, "otis_binary/Model_rag"))


def test_read_otis_binary_without_filename():
    read_otis_binary(
        gfile=os.path.join(FILES_DIR, "otis_binary/grid_rag"),
        hfile=os.path.join(FILES_DIR, "otis_binary/h_rag"),
        ufile=os.path.join(FILES_DIR, "otis_binary/u_rag"),
    )


def test_otis_binary_correct_args():
    with pytest.raises(ValueError):
        read_otis_netcdf(
            hfile=os.path.join(FILES_DIR, "otis_binary/h_rag"),
            ufile=os.path.join(FILES_DIR, "otis_binary/u_rag"),
        )


def test_read_otis_netcdf_with_filename():
    read_otis_netcdf(os.path.join(FILES_DIR, "otis_netcdf/Model_test"))


def test_read_otis_netcdf_without_filename():
    read_otis_netcdf(
        gfile=os.path.join(FILES_DIR, "otis_netcdf/grid.test.nc"),
        hfile=os.path.join(FILES_DIR, "otis_netcdf/hf.test.nc"),
        ufile=os.path.join(FILES_DIR, "otis_netcdf/uv.test.nc"),
    )


def test_otis_netcdf_correct_args():
    with pytest.raises(ValueError):
        read_otis_netcdf(
            hfile=os.path.join(FILES_DIR, "otis_netcdf/hf.test.nc"),
            ufile=os.path.join(FILES_DIR, "otis_netcdf/uv.test.nc"),
        )


def test_read_oceantide():
    read_oceantide(os.path.join(FILES_DIR, "oceantide.zarr"))


def test_read_dataset():
    dsets = [
        xr.open_dataset(os.path.join(FILES_DIR, "otis_merged.nc")),
        xr.open_zarr(os.path.join(FILES_DIR, "oceantide.zarr"), consolidated=True),
    ]
    for dset in dsets:
        read_dataset(dset)
