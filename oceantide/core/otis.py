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

from oceantide.input import read_dataset
from oceantide.tide import Tide


# def read_otis(filename, file_format="netcdf"):
#     """Read tide constituents from Otis format.

#     Args:
#         filename (str):

#     """
#     dset = read_dataset(filename, file_format=file_format)
#     return from_otis(dset)


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
    for f in files:
        if "grid" in f:
            gfile = os.path.basename(f)
        elif "uv." in f or "uv_" in f:
            ufile = os.path.basename(f)
        elif "h." in f or "hf." in f or "h_" in f or "hf_" in f:
            hfile = os.path.basename(f)
    return gfile, hfile, ufile


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
        # y0, y1, x0, x1 = np.fromfile(f, dtype=np.float32, count=4).byteswap(True)

    hRe = np.zeros((nc, ny, nx))
    hIm = np.zeros((nc, ny, nx))

    for ic in range(nc):
        with open(hfile, "rb") as f:
            np.fromfile(f, dtype=np.int32, count=4)
            np.fromfile(f, dtype=np.float32, count=4)

            nskip = int((ic)*(nx * ny * 8 + 8) + 8 + ll - 28)
            f.seek(nskip, 1)

            data = np.fromfile(f, dtype=np.float32, count=2*nx*ny).byteswap(True).reshape((ny, 2*nx))
            hRe[ic] = data[:, 0 : 2*nx-1 : 2]
            hIm[ic] = data[:, 1 : 2*nx : 2]

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
            ["lat_u", "lat_v", "lon_u", "lon_v", "hu", "hv"]
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
            "units": "m",
        }
        self.ds.con.attrs = {
            "standard_name": "tidal_constituent",
            "units": ""
        }
        self.ds.lat.attrs = {
            "standard_name": "latitude",
            "units": "degrees_north"
        }
        self.ds.lon.attrs = {
            "standard_name": "longitude",
            "units": "degrees_east"
        }

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

