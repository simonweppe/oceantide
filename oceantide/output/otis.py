"""Otis binary output."""
import sys
import numpy as np


def to_otis(dset, filename):
    """Write dataset as Otis binary format."""
    nc, ny, nx = dset.et.shape
    header1 = np.array(4 * (nc + 7), dtype=">i4")
    header2 = np.array(8 * nx * ny, dtype=">i4")
    with open(filename, 'wb') as fid:
        # Header
        header1.tofile(fid)
        np.array(nx, dtype=">i4").tofile(fid)
        np.array(ny, dtype=">i4").tofile(fid)
        np.array(nc, dtype=">i4").tofile(fid)
        theta_lim.tofile(fid)
        dset.con.values.astype("S4").tofile(fid)
        header1.tofile(fid)
        # Data
        for icon in range(nc):
            header2.tofile(fid)
            data = np.zeros((ny, nx * 2))
            data[:, 0 : 2 * nx - 1 : 2] = dset.et[icon].real
            data[:, 1 : 2 * nx : 2] = dset.et[icon].imag
            data.astype(">f4").tofile(fid)
            header2.tofile(fid)
