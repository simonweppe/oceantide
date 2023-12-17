"""Read Otis Atlas netcdf file format."""
import os
import glob
from pathlib import Path
import dask
import numpy as np
import dask.array as da
import xarray as xr

from oceantide.core.otis import otis_to_oceantide
from oceantide.tide import Tide


dask.config.set({"array.slicing.split_large_chunks": False})

CONS = [
    "M2",
    "S2",
    "N2",
    "K2",
    "K1",
    "O1",
    "P1",
    "Q1",
    "MM",
    "MF",
    "M4",
    "MN4",
    "MS4",
    "2N2",
    "S1",
]


def read_otis_atlas_netcdf(
    dirname: str,
    x0: float = None,
    x1: float = None,
    y0: float = None,
    y1: float = None,
    nxchunk: int = 500,
    nychunk: int = 500,
) -> xr.Dataset:
    """Read Otis Netcdf file format.

    Parameters
    ----------
    dirname (str)
        Path name with all netcdf Atlas files. Atlas files are organised as one grid
        file and one file per constituent for uv and h.
    x0 (float)
        Longitude left corner to read.
    x1 (float)
        Longitude right corner to read.
    y0 (float)
        Latitude bottom corner to read.
    y1 (float)
        Latitude top corner to read.
    nxchunk (int)
        Chunk size along the x-axis.
    nychunk (int)
        Chunk size along the y-axis.

    Returns
    -------
    dset (xr.Dataset)
        Formatted dataset with the tide accessor.

    Notes
    -----

    - The atlas dataset is very large and requires large amount of RAM to be
      processed, we recommend setting the bounds x0, x1, y0, y1 when reading.
    - It is a good idea setting nxchunk, nychunk close to the slicing sizes.

    """
    gfile = list(Path(dirname).glob("g*.nc"))[0]
    hfile = Path(dirname).glob("h*.nc")
    ufile = Path(dirname).glob("u*.nc")

    if not hfile:
        raise ValueError(f"Cannot find elevation constituents files from {dirname}")
    if not ufile:
        raise ValueError(f"Cannot find transport constituents files from {dirname}")

    bounds = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
    dsg = read_grid(gfile, chunks={"nx": nxchunk, "ny": nychunk}, **bounds)
    dsh = read_elevations(
        hfile, chunks={"nc": None, "nx": nxchunk, "ny": nychunk}, **bounds
    )
    dsu = read_transports(
        ufile, chunks={"nc": None, "nx": nxchunk, "ny": nychunk}, **bounds
    )

    # Make coordinates 2d so they are consistent with non-atlas datasets
    dsh["lon_z"] = dsg.lon_z
    dsh["lat_z"] = dsg.lat_z
    dsu["lon_u"] = dsg.lon_u
    dsu["lat_u"] = dsg.lat_u
    dsu["lon_v"] = dsg.lon_v
    dsu["lat_v"] = dsg.lat_v

    dset = otis_to_oceantide(dsg, dsh, dsu)
    cons = [c for c in CONS if c in dset.con]
    return dset.sel(con=cons, lon=slice(x0, x1), lat=slice(y0, y1))


def indices(
    lon: np.ndarray, lat: np.ndarray, x0: float, x1: float, y0: float, y1: float
) -> tuple[int]:
    """Indices of coordinates in lon and lat arrays.

    Parameters
    ----------
    lon (np.ndarray)
        Longitudes.
    lat (np.ndarray)
        Latitudes.
    x0 (float)
        Left longitude corner to read.
    x1 (float)
        Right longitude corner to read.
    y0 (float)
        Bottom latitude corner to read.
    y1 (float)
        Top latitude corner to read.

    Returns
    -------
    ix0 (int)
        Left longitude index.
    ix1 (int)
        Right longitude index.
    iy0 (int)
        Bottom latitude index.
    iy1 (int)
        Top latitude index.

    """
    ix0 = 0
    ix1 = lon.size
    iy0 = 0
    iy1 = lat.size
    if x0 is not None:
        ix0 = np.maximum(ix0, int(np.abs(lon - x0).argmin()) - 2)
    if x1 is not None:
        ix1 = np.minimum(ix1, int(np.abs(lon - x1).argmin()) + 2)
    if y0 is not None:
        iy0 = np.maximum(iy0, int(np.abs(lat - y0).argmin()) - 2)
    if y1 is not None:
        iy1 = np.minimum(iy1, int(np.abs(lat - y1).argmin()) + 2)
    return ix0, ix1, iy0, iy1


def read_grid(
    filename: str, chunks: dict, x0: float, x1: float, y0: float, y1: float
) -> xr.Dataset:
    """Read grid data.

    Parameters
    ----------
    filename (str)
        Name of grid file to read.
    chunks (dict)
        Mapping dimension names and chunking sizes.
    x0 (float)
        Left longitude corner to read.
    x1 (float)
        Right longitude corner to read.
    y0 (float)
        Bottom latitude corner to read.
    y1 (float)
        Top latitude corner to read.

    Returns
    -------
    dset (xr.Dataset)
        Merged grid dataset.

    """
    dset = xr.open_dataset(filename, chunks=chunks)

    # Slicing
    ix0, ix1, iy0, iy1 = indices(dset.lon_z, dset.lat_z, x0, x1, y0, y1)
    dset = dset.isel(nx=slice(ix0, ix1), ny=slice(iy0, iy1)).chunk(chunks)

    # Define masks
    dset["mz"] = dset.hz > 0
    dset["mu"] = dset.hu > 0
    dset["mv"] = dset.hv > 0

    # Set 2d coordinates
    lat_z, lon_z = da.meshgrid(dset.lat_z, dset.lon_z)
    lat_u, lon_u = da.meshgrid(dset.lat_u, dset.lon_u)
    lat_v, lon_v = da.meshgrid(dset.lat_v, dset.lon_v)
    dset["lon_z"] = xr.DataArray(lon_z, dims=("nx", "ny"))
    dset["lat_z"] = xr.DataArray(lat_z, dims=("nx", "ny"))
    dset["lon_u"] = xr.DataArray(lon_u, dims=("nx", "ny"))
    dset["lat_u"] = xr.DataArray(lat_u, dims=("nx", "ny"))
    dset["lon_v"] = xr.DataArray(lon_v, dims=("nx", "ny"))
    dset["lat_v"] = xr.DataArray(lat_v, dims=("nx", "ny"))
    return dset


def read_elevations(
    filenames: str, chunks: dict, x0: float, x1: float, y0: float, y1: float
) -> xr.Dataset:
    """Read and concatenate individual elevations constituents netcdf files.

    Parameters
    ----------
    filenames (list)
        Name of elevation files to read.
    chunks (dict)
        Mapping dimension names and chunking sizes.
    x0 (float)
        Left longitude corner to read.
    x1 (float)
        Right longitude corner to read.
    y0 (float)
        Bottom latitude corner to read.
    y1 (float)
        Top latitude corner to read.

    Returns
    -------
    dset (xr.Dataset)
        Merged elevation dataset.

    """
    dset = xr.open_mfdataset(
        filenames, concat_dim="nc", combine="nested", chunks=chunks
    )
    ix0, ix1, iy0, iy1 = indices(dset.lon_z, dset.lat_z, x0, x1, y0, y1)
    dset = dset.isel(nx=slice(ix0, ix1), ny=slice(iy0, iy1)).chunk(chunks)
    dset["hRe"] = dset.hRe * 1e-3
    dset["hIm"] = dset.hIm * 1e-3
    return dset


def read_transports(
    filenames: list, chunks: dict, x0: float, x1: float, y0: float, y1: float
) -> xr.Dataset:
    """Read and concatenate individual transports constituents netcdf files.

    Parameters
    ----------
    filenames (list)
        Name of transport files to read.
    chunks (dict)
        Mapping dimension names and chunking sizes.
    x0 (float)
        Left longitude corner to read.
    x1 (float)
        Right longitude corner to read.
    y0 (float)
        Bottom latitude corner to read.
    y1 (float)
        Top latitude corner to read.

    Returns
    -------
    dset (xr.Dataset)
        Merged transport dataset.

    """
    dset = xr.open_mfdataset(
        filenames, concat_dim="nc", combine="nested", chunks=chunks
    )
    ix0, ix1, iy0, iy1 = indices(dset.lon_v, dset.lat_u, x0, x1, y0, y1)
    dset = dset.isel(nx=slice(ix0, ix1), ny=slice(iy0, iy1)).chunk(chunks)
    dset["uRe"] = dset.uRe * 1e-4
    dset["uIm"] = dset.uIm * 1e-4
    dset["vRe"] = dset.vRe * 1e-4
    dset["vIm"] = dset.vIm * 1e-4
    return dset.rename({"uRe": "URe", "uIm": "UIm", "vRe": "VRe", "vIm": "VIm"})
