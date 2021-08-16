"""Test Otis functions."""
import filecmp
from pathlib import Path

from oceantide import read_otis_netcdf
from oceantide.core.otis import (
    read_otis_bin_h,
    read_otis_bin_u,
    read_otis_bin_grid,
    write_otis_bin_h,
    write_otis_bin_u,
    write_otis_bin_grid,
)


FILES_DIR = Path(__file__).parent / "test_files"


def test_read_otis_netcdf():
    hfile = FILES_DIR / "otis_netcdf/hf.test.nc"
    ufile = FILES_DIR / "otis_netcdf/uv.test.nc"
    gfile = FILES_DIR / "otis_netcdf/grid.test.nc"
    dset = read_otis_netcdf(gfile=gfile, hfile=hfile, ufile=ufile)
    assert hasattr(dset, "tide")


def test_binary_otis_io(tmpdir):
    """Read existing binaries, write new binaries, compare existing and new files."""

    hfile0 = FILES_DIR / "otis_binary/h_rag"
    ufile0 = FILES_DIR / "otis_binary/u_rag"
    gfile0 = FILES_DIR / "otis_binary/grid_rag"

    hfile1 = tmpdir / "h"
    ufile1 = tmpdir / "u"
    gfile1 = tmpdir / "g"

    # Reading
    dsh = read_otis_bin_h(hfile0)
    dsu = read_otis_bin_u(ufile0)
    dsg = read_otis_bin_grid(gfile0)

    # Writing
    write_otis_bin_h(hfile1, dsh.hRe, dsh.hIm, dsh.con, dsh.lon_z, dsh.lat_z)
    write_otis_bin_u(
        ufile1, dsu.URe, dsu.UIm, dsu.VRe, dsu.VIm, dsu.con, dsh.lon_z, dsh.lat_z
    )
    write_otis_bin_grid(gfile1, dsg.hz, dsg.mz, dsh.lon_z, dsh.lat_z, dt=12)

    assert filecmp.cmp(hfile0, hfile1, shallow=False)
    assert filecmp.cmp(ufile0, ufile1, shallow=False)

    # Grid files are different due to wrong iob in original files but values are set when reading
    dsg1 = read_otis_bin_grid(gfile1)
    assert dsg.identical(dsg1)


def test_otis_binary_writer_plugin(tmpdir):
    "Test output otis binary plugin."
    dset = read_otis_netcdf(FILES_DIR / "otis_netcdf/Model_test")
    filenames = dset.tide.to_otis_binary(tmpdir, hfile=True, ufile=True, gfile=True)
    print(filenames)
