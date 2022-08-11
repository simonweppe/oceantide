"""Test tide accessor."""
from pathlib import Path
import pytest
import datetime
import xarray as xr

from oceantide import read_otis_netcdf, read_otis_binary, read_oceantide


FILES_DIR = Path(__file__).parent / "test_files"


def test_read_otis_binary_with_filename():
    read_otis_binary(FILES_DIR / "otis_binary/Model_rag")


def test_read_otis_binary_without_filename():
    read_otis_binary(
        gfile=FILES_DIR / "otis_binary/grid_rag",
        hfile=FILES_DIR / "otis_binary/h_rag",
        ufile=FILES_DIR / "otis_binary/u_rag",
    )


def test_otis_binary_correct_args():
    with pytest.raises(ValueError):
        read_otis_netcdf(
            hfile=FILES_DIR / "otis_binary/h_rag",
            ufile=FILES_DIR / "otis_binary/u_rag",
        )


def test_read_otis_netcdf_with_filename():
    read_otis_netcdf(FILES_DIR / "otis_netcdf/Model_test")


def test_read_otis_netcdf_without_filename():
    read_otis_netcdf(
        gfile=FILES_DIR / "otis_netcdf/grid.test.nc",
        hfile=FILES_DIR / "otis_netcdf/hf.test.nc",
        ufile=FILES_DIR / "otis_netcdf/uv.test.nc",
    )


def test_otis_netcdf_correct_args():
    with pytest.raises(ValueError):
        read_otis_netcdf(
            hfile=FILES_DIR / "otis_netcdf/hf.test.nc",
            ufile=FILES_DIR / "otis_netcdf/uv.test.nc",
        )


def test_read_write_oceantide_netcdf(tmpdir):
    dset = read_oceantide(FILES_DIR / "oceantide.nc")
    dset.tide.to_oceantide(tmpdir / "newoceantide.nc")


def test_read_write_oceantide_zarr(tmpdir):
    dset = read_oceantide(FILES_DIR / "oceantide.zarr")
    dset.tide.to_oceantide(str(tmpdir / "newoceantide.zarr"))


def test_supported_oceantide_formats(tmpdir):
    dset = read_oceantide(FILES_DIR / "oceantide.zarr")
    with pytest.raises(ValueError):
        dset.tide.to_oceantide(tmpdir / "newoceantide.zarr", file_format="txt")
    with pytest.raises(ValueError):
        dset.tide.to_oceantide(tmpdir / "newoceantide.txt")
    dset.tide.to_oceantide(tmpdir / "newoceantide.txt", file_format="netcdf")


def test_write_otis_netcdf(tmpdir):
    dset = read_otis_netcdf(FILES_DIR / "otis_netcdf/Model_test")
    dset.tide.to_otis_netcdf(dirname=tmpdir, suffix="")
    dset2 = read_otis_netcdf(tmpdir / "model")


def test_write_otis_binary(tmpdir):
    dset = read_otis_binary(FILES_DIR / "otis_binary/Model_rag")
    dset.tide.to_otis_binary(dirname=tmpdir, suffix="")
    dset2 = read_otis_binary(tmpdir / "model")
