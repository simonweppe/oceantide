"""Otis object.

Grid conventions used in OTIS
-----------------------------
An Arakawa C grid is used for all dynamical calculations. Volume transports U and V are
specified on grid cell edges, and are interpreted to be the average volume transport
over the cell edge. Elevations are interpreted as the average over the cell, and are
given at the center. Boundary conditions at the coast are specified on the U and V
nodes. Open boundary conditions are given by specifying the elevation for open boundary
edge cells, or transports on edge U or V nodes

"""
import os
import re
import numpy as np

from oceantide import Tide


def from_otis(dset):
    """Format Otis-like dataset to implement the oceantide accessor."""
    otis = Otis(dset)
    return otis.ds


def otis_filenames(filename):
    """Otis data file names from `Model_*` metadata file.

    Args:
        filename (str): Name of `Model_*` metadata file specifying other files to read.

    Returns:
        gfile (str): Name of grid file to read, by default defined from `filename`.
        hfile (str): Name of elevation file to read, by default defined from `filename`.
        ufile (str): Name of currents file to read, by default defined from `filename`.

    """
    with open(filename) as stream:
        files = stream.read().split()
    dirname = os.path.dirname(filename)
    gfile = ufile = hfile = None
    for f in files:
        if os.path.split(f)[-1].startswith("g"):
            gfile = os.path.join(dirname, os.path.basename(f))
        elif os.path.split(f)[-1].startswith("u"):
            ufile = os.path.join(dirname, os.path.basename(f))
        elif os.path.split(f)[-1].startswith("h"):
            hfile = os.path.join(dirname, os.path.basename(f))
    assert gfile, f"Cannot identify gfile from {filename} ({files})"
    assert hfile, f"Cannot identify hfile from {filename} ({files})"
    assert ufile, f"Cannot identify ufile from {filename} ({files})"
    return gfile, hfile, ufile


def read_otis_bin_uv(uvfile):
    """Read transport constituents data from otis binary file.

    Args:
        uvfile (str): Name of transport constituents binary file to read.

    Returns:
        URe (array 3d): Real U component URe(con,lat,lon).
        UIm (array 3d): Imag U component UIm(con,lat,lon).
        VRe (array 3d): Real V component VRe(con,lat,lon).
        VIm (array 3d): Imag V component VIm(con,lat,lon).

    """
    with open(uvfile, "rb") as f:
        ll, nx, ny, nc = np.fromfile(f, dtype=np.int32, count=4).byteswap(True)

    URe = np.zeros((nc, ny, nx))
    UIm = np.zeros((nc, ny, nx))
    VRe = np.zeros((nc, ny, nx))
    VIm = np.zeros((nc, ny, nx))

    for ic in range(nc):
        with open(uvfile, "rb") as f:
            np.fromfile(f, dtype=np.int32, count=4).byteswap(True)
            np.fromfile(f, dtype=np.float32, count=4).byteswap(True)

            nskip = int((ic) * (nx * ny * 16 + 8) + 8 + ll - 28)
            f.seek(nskip, 1)
            data = np.fromfile(f, dtype=np.float32, count=4 * nx * ny).byteswap(True)
            data = data.reshape((ny, 4 * nx))

        URe[ic] = data[:, 0 : 4 * nx - 3 : 4]
        UIm[ic] = data[:, 1 : 4 * nx - 2 : 4]
        VRe[ic] = data[:, 2 : 4 * nx - 1 : 4]
        VIm[ic] = data[:, 3 : 4 * nx : 4]

    return URe, UIm, VRe, VIm


def read_otis_bin_h(hfile):
    """Read elevation constituents data from otis binary file.

    Args:
        hfile (str): Name of elevation constituents binary file to read.

    Returns:
        hRe (array 3d): Real elevation component hRe(con,lat,lon).
        hIm (array 3d): Imag elevation component hIm(con,lat,lon).

    """
    with open(hfile, "rb") as f:
        ll, nx, ny, nc = np.fromfile(f, dtype=np.int32, count=4).byteswap(True)

    hRe = np.zeros((nc, ny, nx))
    hIm = np.zeros((nc, ny, nx))

    for ic in range(nc):
        with open(hfile, "rb") as f:
            np.fromfile(f, dtype=np.int32, count=4)
            np.fromfile(f, dtype=np.float32, count=4)

            nskip = int((ic) * (nx * ny * 8 + 8) + 8 + ll - 28)
            f.seek(nskip, 1)

            data = np.fromfile(f, dtype=np.float32, count=2 * nx * ny).byteswap(True)
            data = data.reshape((ny, 2 * nx))
            hRe[ic] = data[:, 0 : 2 * nx - 1 : 2]
            hIm[ic] = data[:, 1 : 2 * nx : 2]

    return hRe, hIm


def read_otis_bin_cons(hfile):
    """Read constituents from otis binary file.

    Args:
        hfile (str): Name of elevation constituents binary file to read.

    Returns:
        cons (array 1d): Constituents with '|S4' dtype.

    """
    CHAR = np.dtype(">c")
    with open(hfile, "rb") as f:
        __, __, __, nc = np.fromfile(f, dtype=np.int32, count=4).byteswap(True)
        np.fromfile(f, dtype=np.int32, count=4)[0]
        cons = [np.fromfile(f, CHAR, 4).tobytes().upper() for i in range(nc)]
        cons = np.array([c.ljust(4).lower() for c in cons])
    return cons


def read_otis_bin_grid(gfile):
    """Reads grid data from otis binary file.

    Args:
        gfile (str): Name of grid binary file to read.

    Returns:
        lon_z, lat_z, lon_u, lat_u, lon_v, lat_v (1darray): Arakawa C-grid coordinates.
        hz (2darray): Depth at Z nodes.
        mz (2darray): Mask at Z nodes.

    """
    with open(gfile, "rb") as f:

        f.seek(4, 0)
        n, m = np.fromfile(f, dtype=np.int32, count=2).byteswap(True)
        y0, y1, x0, x1 = np.fromfile(f, dtype=np.float32, count=4).byteswap(True)
        np.fromfile(f, dtype=np.float32, count=1).byteswap(True)

        nob = np.fromfile(f, dtype=np.int32, count=1).byteswap(True)
        if nob == 0:
            f.seek(20, 1)
            iob = []
        else:
            f.seek(8, 1)
            iob = np.fromfile(f, dtype=np.int32, count=int(2 * nob)).byteswap(True)
            iob = iob.reshape((2, int(nob)))
            f.seek(8, 1)

        hz = np.fromfile(f, dtype=np.float32, count=int(n * m)).byteswap(True)
        f.seek(8, 1)
        mz = np.fromfile(f, dtype=np.int32, count=int(n * m)).byteswap(True)

        hz = hz.reshape((m, n))
        mz = mz.reshape((m, n))

    dx = (x1 - x0) / n
    dy = (y1 - y0) / m

    xcorner = np.clip(np.arange(x0, x1, dx), x0, x1)[0 : n]
    ycorner = np.clip(np.arange(y0, y1, dy), y0, y1)[0 : m]

    lon_z = xcorner + dx / 2
    lon_u = xcorner
    lon_v = xcorner + dx / 2

    lat_z = ycorner + dy / 2
    lat_u = ycorner + dy / 2
    lat_v = ycorner

    return lon_z, lat_z, lon_u, lat_u, lon_v, lat_v, hz, mz


class Otis:
    """Otis object formatter."""

    def __init__(self, dset_otis):
        self.ds = dset_otis
        self.validate()
        self.construct()

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self.ds))

    def _fix_topo(self):
        """Make topography values above zero as they are inconveniently zero."""
        self.ds["hz"] = self.ds["hz"].where(self.ds["hz"] != 0).fillna(0.001)
        self.ds["hu"] = self.ds["hu"].where(self.ds["hu"] != 0).fillna(0.001)
        self.ds["hv"] = self.ds["hv"].where(self.ds["hv"] != 0).fillna(0.001)

    def _mask_vars(self):
        """Mask land in constituents to avoid zero values."""
        for data_var in self.ds.data_vars.values():
            if len(data_var.dims) > 2:
                data_var = data_var.where(data_var != 0)

    def _to_complex(self):
        """Merge real and imaginary components into a complex variable."""
        for v in ["h", "u", "v"]:
            self.ds[f"{v}"] = self.ds[f"{v}Re"] + 1j * self.ds[f"{v}Im"]
            self.ds = self.ds.drop_vars([f"{v}Re", f"{v}Im"])
        self.ds = self.ds.drop_vars(["URe", "UIm", "VRe", "VIm"])
        self.ds = self.ds.rename({"h": "et", "u": "ut", "v": "vt"})

    def _to_single_grid(self):
        """Convert Arakawa into a common grid at the cell centre."""
        lat = self.ds.lat_z
        lon = self.ds.lon_z

        self.ds = self.ds.interp(
            coords={"lon_u": lon, "lon_v": lon, "lat_u": lat, "lat_v": lat},
            kwargs={"fill_value": "extrapolate"},
        ).reset_coords()

        mz = self.ds.hRe.isel(con=0).notnull()
        mu = self.ds.uRe.isel(con=0).notnull()
        mv = self.ds.vRe.isel(con=0).notnull()
        self.ds = self.ds.where(mz).where(mu).where(mv)

        self.ds = self.ds.rename({"lat_z": "lat", "lon_z": "lon", "hz": "depth"})

        self.ds = self.ds.drop_vars(
            ["lat_u", "lat_v", "lon_u", "lon_v", "hu", "hv", "mz", "mu", "mv"],
            errors="ignore",
        )

    def _format_cons(self):
        """Format constituents coordinates."""
        self.ds = self.ds.assign_coords(
            {"con": [c.upper() for c in self.ds.con.values.tobytes().decode().split()]}
        )

    def _set_attributes(self):
        """Define attributes for formatted dataset."""
        self.ds.attrs = {"description": "Tide constituents"}
        self.ds.depth.attrs = {
            "standard_name": "sea_floor_depth_below_mean_sea_level",
            "units": "m",
        }
        self.ds.et.attrs = {
            "standard_name": "tidal_elevation_complex_amplitude",
            "units": "m",
        }
        self.ds.ut.attrs = {
            "standard_name": "tidal_we_velocity_complex_amplitude",
            "units": "m",
        }
        self.ds.vt.attrs = {
            "standard_name": "tidal_ns_velocity_complex_amplitude",
            "units": "m s-1",
        }
        self.ds.con.attrs = {"standard_name": "tidal_constituent", "units": ""}
        self.ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
        self.ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}

    def validate(self):
        """Check that input dataset has all requirements."""
        complexes = ["hRe", "hIm", "uRe", "uIm", "vRe", "vIm"]
        for v in complexes:
            if v not in self.ds.data_vars:
                raise ValueError(f"Variable {v} is required in Otis dataset.")
        reais = ["hz", "hu", "hv"]
        for v in reais:
            if v not in self.ds.data_vars:
                raise ValueError(f"Variable {v} is required in Otis dataset.")
        if self.ds.con.dtype != np.dtype("S4"):
            raise ValueError(f"Constituents variables dtype must be 'S4'.")

    def construct(self):
        """Define constituents dataset."""
        self._fix_topo()
        self._mask_vars()
        self._to_single_grid()
        self._format_cons()
        self._to_complex()
        self._set_attributes()
