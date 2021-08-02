"""Otis binary output."""
import sys
import numpy as np

from oceantide.core.otis import write_otis_bin_h


def to_otis(dset, dirname, hfile=None, ufile=None, gfile=None):
    """Write dataset as Otis binary format."""
    if hfile is not None:
        hfile = Path(dirname) / hfile
        write_otis_bin_h(
            hfile, dset.et.real, dset.et.imag, dset.con, dset.lon, dset.lat
        )


if __name__ == "__main__":
    import filecmp
    import xarray as xr
    from oceantide.core.otis import read_otis_bin_h, write_otis_bin_h

    ds = xr.open_dataset("/data/tide/tpxo9v4a/netcdf/DATA/h_tpxo9.v4a.nc")
    nc, nx, ny = ds.hRe.shape

    dx = ds.lon_z[:, 0][1] - ds.lon_z[:, 0][0]
    dy = ds.lat_z[0, :][1] - ds.lat_z[0, :][0]
    x0 = float(ds.lon_z[:, 0][0] - dx / 2)
    x1 = float(ds.lon_z[:, 0][-1] + dx / 2)
    y0 = float(ds.lat_z[0, :][0] - dy / 2)
    y1 = float(ds.lat_z[0, :][-1] + dy / 2)
    theta_lim = np.hstack([y0, y1, x0, x1]).astype(">f4")

    filename = "file1"

    h = (ds.hRe + 1j * ds.hIm).transpose("ny", "nx", "nc")

    # Write otis tpxo h to test
    with open(filename, 'wb') as fid:
        # Header
        np.array(4 * (nc + 7), dtype=">i4").tofile(fid)
        np.array(nx, dtype=">i4").tofile(fid)
        np.array(ny, dtype=">i4").tofile(fid)
        np.array(nc, dtype=">i4").tofile(fid)
        theta_lim.tofile(fid)
        ds.con.values.astype("S4").tofile(fid)
        np.array(4 * (nc + 7), dtype=">i4").tofile(fid)
        # Data
        constituent_header = np.array(8 * nx * ny, dtype=">i4")
        for ic in range(nc):
            constituent_header.tofile(fid)
            data = np.zeros((ny, nx * 2))
            data[:, 0 : 2 * nx - 1 : 2] = h[:, :, ic].real
            data[:, 1 : 2 * nx : 2] = h[:, :, ic].imag
            data.astype(">f4").tofile(fid)
            constituent_header.tofile(fid)

    # Check values in file written
    with open(filename, "rb") as f:
        ll_1, nx_1, ny_1, nc_1 = np.fromfile(f, dtype=np.int32, count=4).byteswap(True)
        y0_1, y1_1, x0_1, x1_1 = np.fromfile(f, dtype=np.float32, count=4).byteswap(True)

    print(f"ll: {ll_1}")
    print(f"nx: {nx_1}")
    print(f"ny: {ny_1}")
    print(f"nc: {nc_1}")
    print(f"x0: {x0_1}")
    print(f"x1: {x1_1}")
    print(f"x0: {y0_1}")
    print(f"x0: {y1_1}")

    # Compare with new function
    ds2 = ds.transpose("nc", "nx", "ny")
    write_otis_bin_h("file2", ds2.hRe, ds2.hIm, ds2.con, ds2.lon_z, ds2.lat_z)

    print(f'Same file: {filecmp.cmp("file1", "file2", shallow=False)}')

    # # Read original and new files to compare
    # filename0 = "/data/tide/tpxo9v4a/bin/DATA/h_tpxo9.v4a"
    # filename1 = filename
    # ds0 = read_otis_bin_h(filename0)
    # ds1 = read_otis_bin_h(filename1)
