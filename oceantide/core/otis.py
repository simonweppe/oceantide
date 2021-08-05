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
import dask.array as da
import xarray as xr

from oceantide.tide import Tide
from oceantide.core.utils import arakawa_grid, set_attributes


CHAR = np.dtype(">c")
INT = np.dtype(">i4")
FLOAT = np.dtype(">f4")


def from_otis(dset):
    """Format Otis-like dataset to implement the oceantide accessor."""
    otis = Otis(dset)
    return otis.ds


def theta_lim(lon, lat):
    """Grid limits (y0, y1, x0, x1).

    Args:
        - lon (DataArray, 1darray): Longitude coordinates at the cell centre (Z-nodes).
        - lat (DataArray, 1darray): Latitude coordinates at the cell centre (Z-nodes).

    Return:
        - theta_lim (1darray): Grid limits with dtype for writing to binary file.

    """
    if lon.ndim > 1:
        if np.sum(np.diff(lon[:, 0])) != 0:
            lon = lon[:, 0]
        else:
            lon = lon[0, :]
    if lat.ndim > 1:
        if np.sum(np.diff(lat[0, :])) != 0:
            lat = lat[0, :]
        else:
            lat = lat[:, 0]
    dx = lon[1] - lon[0]
    dy = lat[1] - lat[0]
    if dx == 0 or dy == 0:
        raise ValueError(f"Longitude or latitude values are all equal")
    x0 = float(lon[0] - dx / 2)
    x1 = float(lon[-1] + dx / 2)
    y0 = float(lat[0] - dy / 2)
    y1 = float(lat[-1] + dy / 2)
    return np.hstack([y0, y1, x0, x1]).astype(FLOAT)


def indices_open_boundary(mz):
    """Indices of open boundary iob.

    Args:
        - mz (DataArray): Land mask in Otis format :math:`m_z(nx,ny)`.

    Return:
        - iob (2darray): Indices of open boundary :math:`iob(2,nob)`. Longitude and
          latitude indices are defined in the first and second rows respectively.

    """
    nx, ny = mz.shape

    # Define indices coordinates
    mz = mz.assign_coords({"nx": range(1, nx + 1), "ny": range(1, ny + 1)}).fillna(1.)

    # Slice four boundaries, sort so they are continuous, stack along new coord
    dsl = mz.isel(ny=[0]).stack({"n": ["nx", "ny"]})
    dst = mz.isel(nx=[-1]).stack({"n": ["nx", "ny"]})
    dsr = mz.isel(ny=[-1]).sortby("nx", ascending=False).stack({"n": ["nx", "ny"]})
    dsb = mz.isel(nx=[0]).sortby("ny", ascending=False).stack({"n": ["nx", "ny"]})

    # Concatenate boundaries and keep water points
    iob = xr.concat([dsl, dst, dsr, dsb], dim="n")
    iob = iob.where(iob == 1, drop=True)

    return np.array([iob.nx.values, iob.ny.values], dtype=INT)


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
        if os.path.split(f)[-1].lower().startswith("g"):
            gfile = os.path.join(dirname, os.path.basename(f))
        elif os.path.split(f)[-1].lower().startswith("u"):
            ufile = os.path.join(dirname, os.path.basename(f))
        elif os.path.split(f)[-1].lower().startswith("h"):
            hfile = os.path.join(dirname, os.path.basename(f))
    assert gfile, f"Cannot identify gfile from {filename} ({files})"
    assert hfile, f"Cannot identify hfile from {filename} ({files})"
    assert ufile, f"Cannot identify ufile from {filename} ({files})"
    return gfile, hfile, ufile


def read_otis_bin_u(ufile):
    """Read transport constituents data from otis binary file.

    Args:
        - ufile (str): Name of transport constituents binary file to read.

    Returns:
        - dset (Dataset): Transport constituents grid with variables:
            - URe: Real component of eastward transport :math:`U_{Re}(nc,nx,ny)`.
            - UIm: Imag component of eastward transport :math:`U_{Im}(nc,nx,ny)`.
            - VRe: Real component of northward transport :math:`V_{Re}(nc,nx,ny)`.
            - VIm: Imag component of northward transport :math:`V_{Im}(nc,nx,ny)`.

    TODO: Avoid transposing.

    """
    with open(ufile, "rb") as f:
        ll, nx, ny, nc = np.fromfile(f, dtype=np.int32, count=4).byteswap(True)
        y0, y1, x0, x1 = np.fromfile(f, dtype=np.float32, count=4).byteswap(True)
        cons = [np.fromfile(f, CHAR, 4).tobytes().upper() for i in range(nc)]

    URe = np.zeros((nc, ny, nx))
    UIm = np.zeros((nc, ny, nx))
    VRe = np.zeros((nc, ny, nx))
    VIm = np.zeros((nc, ny, nx))

    for ic in range(nc):
        with open(ufile, "rb") as f:
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

    URe = URe.transpose((0, 2, 1))
    UIm = UIm.transpose((0, 2, 1))
    VRe = VRe.transpose((0, 2, 1))
    VIm = VIm.transpose((0, 2, 1))

    # Dataset Otis style
    cons = np.array([c.ljust(4).lower() for c in cons])
    lon_u, lat_u = arakawa_grid(nx, ny, x0, x1, y0, y1, "u")
    lon_v, lat_v = arakawa_grid(nx, ny, x0, x1, y0, y1, "v")
    lat_u, lon_u = np.meshgrid(lat_u, lon_u)
    lat_v, lon_v = np.meshgrid(lat_v, lon_v)

    dset = xr.Dataset()
    dset["con"] = xr.DataArray(da.from_array(cons), dims=("nc",))
    dset["lon_u"] = xr.DataArray(da.from_array(lon_u), dims=("nx", "ny"))
    dset["lat_u"] = xr.DataArray(da.from_array(lat_u), dims=("nx", "ny"))
    dset["lon_v"] = xr.DataArray(da.from_array(lon_v), dims=("nx", "ny"))
    dset["lat_v"] = xr.DataArray(da.from_array(lat_v), dims=("nx", "ny"))
    dset["URe"] = xr.DataArray(da.from_array(URe), dims=("nc", "nx", "ny"))
    dset["UIm"] = xr.DataArray(da.from_array(UIm), dims=("nc", "nx", "ny"))
    dset["VRe"] = xr.DataArray(da.from_array(VRe), dims=("nc", "nx", "ny"))
    dset["VIm"] = xr.DataArray(da.from_array(VIm), dims=("nc", "nx", "ny"))

    # Attributes
    set_attributes(dset, "otis")
    dset.attrs = {
        "type": "OTIS tidal transport file",
        "title": "Oceantide tidal transport from binary file"
    }

    return dset


def read_otis_bin_h(hfile):
    """Read elevation constituents data from otis binary file.

    Args:
        - hfile (str): Name of elevation constituents binary file to read.

    Returns:
        - dset (Dataset): Elevation constituents grid with variables:
            - hRe: Real component of tidal elevation :math:`h_{Re}(nc,nx,ny)`.
            - hIm: Imag component of tidal elevation :math:`h_{Im}(nc,nx,ny)`.

    TODO: Avoid transposing.

    """
    with open(hfile, "rb") as f:
        ll, nx, ny, nc = np.fromfile(f, dtype=np.int32, count=4).byteswap(True)
        y0, y1, x0, x1 = np.fromfile(f, dtype=np.float32, count=4).byteswap(True)
        cons = [np.fromfile(f, CHAR, 4).tobytes().upper() for i in range(nc)]

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

    hRe = hRe.transpose((0, 2, 1))
    hIm = hIm.transpose((0, 2, 1))

    # Dataset Otis style
    lon_z, lat_z = arakawa_grid(nx, ny, x0, x1, y0, y1, "h")
    lat_z, lon_z = np.meshgrid(lat_z, lon_z)
    cons = np.array([c.ljust(4).lower() for c in cons])

    dset = xr.Dataset()
    dset["con"] = xr.DataArray(da.from_array(cons), dims=("nc",))
    dset["lon_z"] = xr.DataArray(da.from_array(lon_z), dims=("nx", "ny"))
    dset["lat_z"] = xr.DataArray(da.from_array(lat_z), dims=("nx", "ny"))
    dset["hRe"] = xr.DataArray(da.from_array(hRe), dims=("nc", "nx", "ny"))
    dset["hIm"] = xr.DataArray(da.from_array(hIm), dims=("nc", "nx", "ny"))

    # Attributes
    set_attributes(dset, "otis")
    dset.attrs = {
        "type": "OTIS tidal elevation file",
        "title": "Oceantide tidal elevation from binary file"
    }

    return dset


def read_otis_bin_cons(hfile):
    """Read constituents from otis binary file.

    Args:
        - hfile (str): Name of elevation constituents binary file to read.

    Returns:
        - cons (array 1d): Constituents with '|S4' dtype.

    TODO: Deprecate.

    """
    with open(hfile, "rb") as f:
        __, __, __, nc = np.fromfile(f, dtype=np.int32, count=4).byteswap(True)
        np.fromfile(f, dtype=np.int32, count=4)[0]
        cons = [np.fromfile(f, CHAR, 4).tobytes().upper() for i in range(nc)]
        cons = np.array([c.ljust(4).lower() for c in cons])
    return cons


def read_otis_bin_grid(gfile):
    """Read grid data from otis binary file.

    Args:
        - gfile (str): Name of grid binary file to read.

    Returns:
        - dset (Dataset): Grid with variables:
            - hz: Depth :math:`hz(lat_z,lon_z)`.
            - mz: Mask :math:`mz(lat_z,lon_z)`.

    TODO: Avoid transposing.

    """
    with open(gfile, "rb") as f:

        f.seek(4, 0)
        nx, ny = np.fromfile(f, dtype=INT, count=2)
        y0, y1, x0, x1 = np.fromfile(f, dtype=FLOAT, count=4)
        dt = np.fromfile(f, dtype=FLOAT, count=1) # lat-lon if > 0
        nob = np.fromfile(f, dtype=INT, count=1)
        if nob == 0:
            f.seek(20, 1)
            iob = []
        else:
            f.seek(8, 1)
            iob = np.fromfile(f, INT, count=int(2 * nob))
            iob = iob.reshape((2, int(nob)))
            f.seek(8, 1)

        hz = np.fromfile(f, dtype=FLOAT, count=int(nx * ny))
        hz = hz.reshape((ny, nx))
        f.seek(8, 1)
        mz = np.fromfile(f, dtype=INT, count=int(nx * ny))
        mz = mz.reshape((ny, nx))

    hz = hz.transpose()
    mz = mz.transpose()

    # Dataset Otis style
    lon_z, lat_z = arakawa_grid(nx, ny, x0, x1, y0, y1, "h")
    lat_z, lon_z = np.meshgrid(lat_z, lon_z)
    dset = xr.Dataset()

    dset["lon_z"] = xr.DataArray(da.from_array(lon_z), dims=("nx", "ny"))
    dset["lat_z"] = xr.DataArray(da.from_array(lat_z), dims=("nx", "ny"))
    dset["hz"] = xr.DataArray(da.from_array(hz), dims=("nx", "ny"))
    dset["mz"] = xr.DataArray(da.from_array(mz), dims=("nx", "ny"))

    # Attributes
    set_attributes(dset, "otis")
    dset.attrs = {
        "type": "OTIS Arakawa C-grid file",
        "title": "Oceantide bathymetry from binary file"
    }

    return dset


def write_otis_bin_u(ufile, URe, UIm, VRe, VIm, con, lon, lat):
    """Write elevation constituents data in the otis binary file.

    Args:
        - ufile (str): Name of transports binary constituents file to write.
        - URe (DataArray, 3darray): Real eastward transport :math:`\\Re{U}(nc,nx,ny)`.
        - UIm (DataArray, 3darray): Imag eastward transport :math:`\\Im{U}(nc,nx,ny)`.
        - VRe (DataArray, 3darray): Real northward transport :math:`\\Re{V}(nc,nx,ny)`.
        - VIm (DataArray, 3darray): Imag northward transport :math:`\\Im{V}(nc,nx,ny)`.
        - con (1darray): Constituents names (lowercase).
        - lon (DataArray, 1darray): Longitude coordinates at the cell centre (Z-nodes).
        - lat (DataArray, 1darray): Latitude coordinates at the cell centre (Z-nodes).

    Note:
        - Arrays must have shape consistent with Otis convention :math:`(nc,nx,ny)`.
        - Coordinates lon and lat can be 1d or 2d arrays.

    """
    nc, nx, ny = URe.shape

    with open(ufile, "wb") as fid:

        # Header
        delim = np.array(4 * (nc + 7), dtype=INT)
        delim.tofile(fid)
        np.array(nx, dtype=INT).tofile(fid)
        np.array(ny, dtype=INT).tofile(fid)
        np.array(nc, dtype=INT).tofile(fid)
        theta_lim(lon, lat).tofile(fid)
        con.values.astype("S4").tofile(fid)
        delim.tofile(fid)

        # Records
        delim = np.array(2 * 8 * nx * ny, dtype=INT)
        for ic in range(nc):
            delim.tofile(fid)
            data = np.zeros((ny, nx * 4))
            data[:, 0 : 4 * nx - 3 : 4] = URe[ic].T
            data[:, 1 : 4 * nx - 2 : 4] = UIm[ic].T
            data[:, 2 : 4 * nx - 1 : 4] = VRe[ic].T
            data[:, 3 : 4 * nx - 0 : 4] = VIm[ic].T
            data.astype(FLOAT).tofile(fid)
            delim.tofile(fid)


def write_otis_bin_h(hfile, hRe, hIm, con, lon, lat):
    """Write elevation constituents data in the otis binary file.

    Args:
        - hfile (str): Name of elevation binary constituents file to write.
        - hRe (DataArray, 3darray): Real elevation :math:`\\Re{h}(nc,nx,ny)`.
        - hIm (DataArray, 3darray): Imag elevation :math:`\\Im{h}(nc,nx,ny)`.
        - con (1darray): Constituents names (lowercase).
        - lon (DataArray, 1darray): Longitude coordinates at the cell centre (Z-nodes).
        - lat (DataArray, 1darray): Latitude coordinates at the cell centre (Z-nodes).

    Note:
        - Arrays must have shape consistent with Otis convention :math:`(nc,nx,ny)`.
        - Coordinates lon and lat can be 1d or 2d arrays.

    """
    nc, nx, ny = hRe.shape

    with open(hfile, "wb") as fid:

        # Header
        delim = np.array(4 * (nc + 7), dtype=INT)
        delim.tofile(fid)
        np.array(nx, dtype=INT).tofile(fid)
        np.array(ny, dtype=INT).tofile(fid)
        np.array(nc, dtype=INT).tofile(fid)
        theta_lim(lon, lat).tofile(fid)
        con.values.astype("S4").tofile(fid)
        delim.tofile(fid)

        # Records
        delim = np.array(8 * nx * ny, dtype=INT)
        for ic in range(nc):
            delim.tofile(fid)
            data = np.zeros((ny, nx * 2))
            data[:, 0 : 2 * nx - 1 : 2] = hRe[ic].T
            data[:, 1 : 2 * nx - 0 : 2] = hIm[ic].T
            data.astype(FLOAT).tofile(fid)
            delim.tofile(fid)


def write_otis_bin_grid(gfile, hz, mz, lon, lat, dt=12):
    """Write grid data in the otis binary file.

    Args:
        - gfile (str): Name of grid binary file to write.
        - hz (DataArray, 2darray): Water depth at Z-nodes :math:`h(nx,ny)`.
        - mz (DataArray, 2darray): Land mask at Z-nodes :math:`m(nx,ny)`.
        - lon (DataArray, 1darray): Longitude coordinates at the cell centre (Z-nodes).
        - lat (DataArray, 1darray): Latitude coordinates at the cell centre (Z-nodes).

    Note:
        - Arrays must have shape consistent with Otis convention :math:`(nc,nx,ny)`.
        - Coordinates lon and lat can be 1d or 2d arrays.

    """
    nx, ny = hz.shape
    iob = indices_open_boundary(mz)
    nob = iob.shape[1]

    with open(gfile, "wb") as fid:

        # Header
        delim = np.array(32, dtype=INT)
        delim.tofile(fid)
        np.array([nx, ny], dtype=INT).tofile(fid)
        theta_lim(lon, lat).tofile(fid)
        np.array(dt, dtype=FLOAT).tofile(fid)
        np.array(nob, dtype=INT).tofile(fid)
        delim.tofile(fid)

        # Indices open boundaries
        delim = np.array(8 * nob, dtype=INT)
        if nob == 0:
            np.array([4, 0, 4], dtype=INT).tofile(fid)
        else:
            delim.tofile(fid)
            iob.tofile(fid)
            delim.tofile(fid)

        # Water depth
        delim = np.array(4 * nx * ny, dtype=INT)
        delim.tofile(fid)
        hz.fillna(0.).values.T.astype(FLOAT).tofile(fid)
        delim.tofile(fid)

        # Land mask
        delim.tofile(fid)
        mz.fillna(1.).values.T.astype(INT).tofile(fid)
        delim.tofile(fid)


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
        decoded = [c.upper() for c in self.ds.con.values.tobytes().decode().split()]
        self.ds = self.ds.assign_coords({"con": np.array(decoded).astype("U4")})

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


if __name__ == "__main__":

    hfile = "/data/tide/tpxo9v4a/bin/DATA/h_tpxo9.v4a"
    ufile = "/data/tide/tpxo9v4a/bin/DATA/u_tpxo9.v4a"
    gfile = "/data/tide/tpxo9v4a/bin/DATA/grid_tpxo9.v4a"

    # Original netcdf
    dsg0 = xr.open_dataset("/data/tide/tpxo9v4a/netcdf/DATA/grid_tpxo9.v4a.nc")

    # Reading
    dsh = read_otis_bin_h(hfile)
    # dsu = read_otis_bin_u(ufile)
    dsg = read_otis_bin_grid(gfile)

    # Writing
    # write_otis_bin_h("./hfile", dsh.hRe, dsh.hIm, dsh.con, dsh.lon_z, dsh.lat_z)
    # write_otis_bin_u("./ufile", dsu.URe, dsu.UIm, dsu.VRe, dsu.VIm, dsu.con, dsh.lon_z, dsh.lat_z)
    write_otis_bin_grid("./gfile", dsg.hz, dsg.mz, dsh.lon_z, dsh.lat_z, dt=12)

    # Reading written files
    # dsh1 = read_otis_bin_h("./hfile")
    # dsu1 = read_otis_bin_u("./ufile")
    dsg1 = read_otis_bin_grid("./gfile")

