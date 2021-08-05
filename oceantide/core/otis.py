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

    """
    with open(ufile, "rb") as f:
        ll, nx, ny, nc = np.fromfile(f, dtype=INT, count=4)
        y0, y1, x0, x1 = np.fromfile(f, dtype=FLOAT, count=4)
        cons = [np.fromfile(f, CHAR, 4).tobytes().upper() for i in range(nc)]

    URe = np.zeros((nc, ny, nx))
    UIm = np.zeros((nc, ny, nx))
    VRe = np.zeros((nc, ny, nx))
    VIm = np.zeros((nc, ny, nx))

    for ic in range(nc):
        with open(ufile, "rb") as f:
            np.fromfile(f, dtype=INT, count=4)
            np.fromfile(f, dtype=FLOAT, count=4)

            nskip = int((ic) * (nx * ny * 16 + 8) + 8 + ll - 28)
            f.seek(nskip, 1)
            data = np.fromfile(f, dtype=FLOAT, count=4 * nx * ny).reshape((ny, 4 * nx))

        URe[ic] = data[:, 0 : 4 * nx - 3 : 4]
        UIm[ic] = data[:, 1 : 4 * nx - 2 : 4]
        VRe[ic] = data[:, 2 : 4 * nx - 1 : 4]
        VIm[ic] = data[:, 3 : 4 * nx - 0 : 4]

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

    """
    with open(hfile, "rb") as f:
        ll, nx, ny, nc = np.fromfile(f, dtype=INT, count=4)
        y0, y1, x0, x1 = np.fromfile(f, dtype=FLOAT, count=4)
        cons = [np.fromfile(f, CHAR, 4).tobytes().upper() for i in range(nc)]

    hRe = np.zeros((nc, ny, nx))
    hIm = np.zeros((nc, ny, nx))

    for ic in range(nc):
        with open(hfile, "rb") as f:
            np.fromfile(f, dtype=INT, count=4)
            np.fromfile(f, dtype=FLOAT, count=4)

            nskip = int((ic) * (nx * ny * 8 + 8) + 8 + ll - 28)
            f.seek(nskip, 1)

            data = np.fromfile(f, dtype=FLOAT, count=2 * nx * ny).reshape((ny, 2 * nx))
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


def read_otis_bin_grid(gfile):
    """Read grid data from otis binary file.

    Args:
        - gfile (str): Name of grid binary file to read.

    Returns:
        - dset (Dataset): Grid with variables:
            - lon_z: Longitude coordinates at Z-nodes :math:`lon_z(nx,ny)`.
            - lat_z: Latitude coordinates at Z-nodes :math:`lat_z(nx,ny)`.
            - lon_u: Longitude coordinates at U-nodes :math:`lon_u(nx,ny)`.
            - lat_u: Latitude coordinates at U-nodes :math:`lat_u(nx,ny)`.
            - lon_v: Longitude coordinates at V-nodes :math:`lon_v(nx,ny)`.
            - lat_v: Latitude coordinates at V-nodes :math:`lat_v(nx,ny)`.
            - mz: Mask at Z-nodes :math:`m_z(nx,ny)`.
            - mu: Mask at U-nodes :math:`m_u(nx,ny)`.
            - mv: Mask at V-nodes :math:`m_v(nx,ny)`.
            - hz: Depth at Z-nodes :math:`h_z(nx,ny)`.
            - hu: Depth at U-nodes :math:`h_u(nx,ny)`.
            - hv: Depth at V-nodes :math:`h_v(nx,ny)`.
            - iob_z: Indices of open boundary in hz :math:`iob_z(2,nob)`.
            - iob_u: Indices of open boundary in hu :math:`iob_u(2,nob)`.
            - iob_v: Indices of open boundary in hv :math:`iob_v(2,nob)`.

    Note:
        - Indices of open boundary are defined from mz, mu, mv, the values of iob
          read from file are ignored (there are inconsistencies in tpxo files).

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
            iob = np.fromfile(f, INT, count=int(2 * nob)).reshape((2, int(nob)))
            f.seek(8, 1)

        hz = np.fromfile(f, dtype=FLOAT, count=int(nx * ny)).reshape((ny, nx))
        f.seek(8, 1)
        mz = np.fromfile(f, dtype=INT, count=int(nx * ny)).reshape((ny, nx))

    hz = hz.transpose()
    mz = mz.transpose()

    lon_z, lat_z = arakawa_grid(nx, ny, x0, x1, y0, y1, "h")
    lon_u, lat_u = arakawa_grid(nx, ny, x0, x1, y0, y1, "u")
    lon_v, lat_v = arakawa_grid(nx, ny, x0, x1, y0, y1, "v")
    lat_z, lon_z = np.meshgrid(lat_z, lon_z)
    lat_u, lon_u = np.meshgrid(lat_u, lon_u)
    lat_v, lon_v = np.meshgrid(lat_v, lon_v)

    # Dataset Otis style
    dset = xr.Dataset()

    # Coords
    dset["lon_z"] = xr.DataArray(da.from_array(lon_z), dims=("nx", "ny"))
    dset["lat_z"] = xr.DataArray(da.from_array(lat_z), dims=("nx", "ny"))
    dset["lon_u"] = xr.DataArray(da.from_array(lon_u), dims=("nx", "ny"))
    dset["lat_u"] = xr.DataArray(da.from_array(lat_u), dims=("nx", "ny"))
    dset["lon_v"] = xr.DataArray(da.from_array(lon_v), dims=("nx", "ny"))
    dset["lat_v"] = xr.DataArray(da.from_array(lat_v), dims=("nx", "ny"))

    # Mask
    dset["mz"] = xr.DataArray(da.from_array(mz), dims=("nx", "ny")).astype("int32")
    dset["mu"] = dset.mz * dset.mz.roll(nx=1, roll_coords=False)
    dset["mv"] = dset.mz * dset.mz.roll(ny=1, roll_coords=False)

    # Depth
    dset["hz"] = xr.DataArray(da.from_array(hz), dims=("nx", "ny")).astype("float64")
    iku = (dset.mz + dset.mz.roll(nx=1, roll_coords=False)) == 1
    ikv = (dset.mz + dset.mz.roll(ny=1, roll_coords=False)) == 1
    hu = dset.mu * (dset.hz + dset.hz.roll(nx=1, roll_coords=False)) / 2
    hv = dset.mv * (dset.hz + dset.hz.roll(ny=1, roll_coords=False)) / 2
    dset["hu"] = xr.where(iku, dset.hz, hu)
    dset["hv"] = xr.where(ikv, dset.hz, hv)

    # Indices open boundaries
    # dset["iob_z"] = xr.DataArray(da.from_array(iob), dims=("iiob", "nob_z"))
    dset["iob_z"] = xr.DataArray(indices_open_boundary(dset.mz), dims=("iiob", "nob_z"))
    dset["iob_u"] = xr.DataArray(indices_open_boundary(dset.mu), dims=("iiob", "nob_u"))
    dset["iob_v"] = xr.DataArray(indices_open_boundary(dset.mv), dims=("iiob", "nob_v"))

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


def otis_to_oceantide(dsg, dsh, dsu):
    """Convert otis datasets into oceantide format."""
    otis = Otis(dsg, dsh, dsu)
    ds = otis()
    return ds


class Otis:
    """Otis object formatter.

    Args:
        dsg (Dataset): Otis grid dataset.
        dsh (Dataset): Otis elevation dataset.
        dsu (Dataset): Otis transports dataset.

    """

    def __init__(self, dsg, dsh, dsu):
        self.dsg = dsg
        self.dsh = dsh
        self.dsu = dsu

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self.ds))

    def __call__(self):
        self.validate()
        self.construct()
        return self.ds

    def _merge_otis_datasets(self):
        """Combine h, u and grid datasets into single dataset."""
        dsg = self.dsg.transpose("ny", "nx", ...)
        dsh = self.dsh.transpose("nc", "ny", "nx", ...)
        dsu = self.dsu.transpose("nc", "ny", "nx", ...)

        mz = dsg.mz.rename({"nx": "lon_z", "ny": "lat_z"})
        mu = dsg.mu.rename({"nx": "lon_u", "ny": "lat_u"})
        mv = dsg.mv.rename({"nx": "lon_v", "ny": "lat_v"})

        URe = dsu.URe.rename({"nc": "con", "nx": "lon_u", "ny": "lat_u"}).where(mu)
        UIm = dsu.UIm.rename({"nc": "con", "nx": "lon_u", "ny": "lat_u"}).where(mu)
        VRe = dsu.VRe.rename({"nc": "con", "nx": "lon_v", "ny": "lat_v"}).where(mv)
        VIm = dsu.VIm.rename({"nc": "con", "nx": "lon_v", "ny": "lat_v"}).where(mv)

        self.ds = xr.Dataset(
            coords={
                "con": dsh.con.rename({"nc": "con"}),
                "lon_z": dsh.lon_z.isel(ny=0).rename({"nx": "lon_z"}),
                "lat_z": dsh.lat_z.isel(nx=0).rename({"ny": "lat_z"}),
                "lon_u": dsu.lon_u.isel(ny=0).rename({"nx": "lon_u"}),
                "lat_u": dsu.lat_u.isel(nx=0).rename({"ny": "lat_u"}),
                "lon_v": dsu.lon_v.isel(ny=0).rename({"nx": "lon_v"}),
                "lat_v": dsu.lat_v.isel(nx=0).rename({"ny": "lat_v"}),
            },
        )
        self.ds["hz"] = dsg.hz.rename({"nx": "lon_z", "ny": "lat_z"}).where(mz)
        self.ds["hu"] = dsg.hu.rename({"nx": "lon_u", "ny": "lat_u"}).where(mu)
        self.ds["hv"] = dsg.hv.rename({"nx": "lon_v", "ny": "lat_v"}).where(mv)
        self.ds["hRe"] = dsh.hRe.rename({"nc": "con", "nx": "lon_z", "ny": "lat_z"}).where(mz)
        self.ds["hIm"] = dsh.hIm.rename({"nc": "con", "nx": "lon_z", "ny": "lat_z"}).where(mz)
        self.ds["uRe"] = URe / self.ds["hu"]
        self.ds["uIm"] = UIm / self.ds["hu"]
        self.ds["vRe"] = VRe / self.ds["hv"]
        self.ds["vIm"] = VIm / self.ds["hv"]
        self.ds["con"] = self.ds.con.astype("S4")

        self.ds = self.ds.where(self.ds < 1e10)

    def _to_complex(self):
        """Merge real and imaginary components into a complex variable."""
        for v in ["h", "u", "v"]:
            self.ds[f"{v}"] = self.ds[f"{v}Re"] + 1j * self.ds[f"{v}Im"]
            self.ds = self.ds.drop_vars([f"{v}Re", f"{v}Im"])
        self.ds = self.ds.rename({"h": "et", "u": "ut", "v": "vt"})

    def _to_single_grid(self):
        """Convert Arakawa into single grid at Z-nodes."""
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
        for v in ["hRe", "hIm"]:
            if v not in self.dsh.data_vars:
                raise ValueError(f"Variable {v} required in elevation dataset dsh.")
        for v in ["URe", "UIm", "VRe", "VIm"]:
            if v not in self.dsu.data_vars:
                raise ValueError(f"Variable {v} required in transports dataset dsu.")
        for v in ["hz", "hu", "hv", "mz", "mu", "mv"]:
            if v not in self.dsg.data_vars:
                raise ValueError(f"Variable {v} required in grid dataset dsg.")
        if self.dsh.con.dtype != np.dtype("S4"):
            raise ValueError(f"Constituents variables dtype must be 'S4'.")

    def construct(self):
        """Define constituents dataset."""
        self._merge_otis_datasets()
        self._to_single_grid()
        self._format_cons()
        self._to_complex()
        self._set_attributes()


if __name__ == "__main__":

    hfile = "/data/tide/tpxo9v4a/bin/DATA/h_tpxo9.v4a"
    ufile = "/data/tide/tpxo9v4a/bin/DATA/u_tpxo9.v4a"
    gfile = "/data/tide/tpxo9v4a/bin/DATA/grid_tpxo9.v4a"

    # Original netcdf
    # dsh0 = xr.open_dataset("/data/tide/tpxo9v4a/netcdf/DATA/h_tpxo9.v4a.nc")
    # dsu0 = xr.open_dataset("/data/tide/tpxo9v4a/netcdf/DATA/u_tpxo9.v4a.nc")
    dsg0 = xr.open_dataset("/data/tide/tpxo9v4a/netcdf/DATA/grid_tpxo9.v4a.nc")

    # Reading
    # dsh = read_otis_bin_h(hfile)
    # dsu = read_otis_bin_u(ufile)
    dsg = read_otis_bin_grid(gfile)

    # Writing
    # write_otis_bin_h("./hfile", dsh.hRe, dsh.hIm, dsh.con, dsh.lon_z, dsh.lat_z)
    # write_otis_bin_u("./ufile", dsu.URe, dsu.UIm, dsu.VRe, dsu.VIm, dsu.con, dsh.lon_z, dsh.lat_z)
    # write_otis_bin_grid("./gfile", dsg.hz, dsg.mz, dsh.lon_z, dsh.lat_z, dt=12)

    # Reading written files
    # dsh1 = read_otis_bin_h("./hfile")
    # dsu1 = read_otis_bin_u("./ufile")
    # dsg1 = read_otis_bin_grid("./gfile")
