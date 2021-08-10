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


# dask.config.set({"array.slicing.split_large_chunks": False})
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
    "S1"
]


def read_otis_atlas_netcdf(filename, x0=None, x1=None, y0=None, y1=None, nxchunk=500, nychunk=500):
    """Read Otis Netcdf file format.

    Args:
        - filename (str): Path name with all netcdf Atlas files. Atlas files are
          organised as one single file per constituent for u and h and one grid file.
        - x0 (float): Longitude left corner to read.
        - x1 (float): Longitude right corner to read.
        - y0 (float): Latitude left corner to read.
        - y1 (float): Latitude right corner to read.
        - nxchunk (int): Chunk size along the x-axis.
        - nychunk (int): Chunk size along the y-axis.

    Returns:
        - Formatted dataset with the tide accessor.

    Note:
        - The atlas dataset is very large and requires large amount of RAM to be
          processed, we recommend setting the bounds x0, x1, y0, y1 when reading.
        - It is a good idea setting nxchunk, nychunk close to the slicing sizes.

    """
    gfile = list(Path(filename).glob("g*.nc"))[0]
    hfile = Path(filename).glob("h*.nc")
    ufile = Path(filename).glob("u*.nc")

    if not hfile:
        raise ValueError(f"Cannot find elevation constituents files from {filename}")
    if not ufile:
        raise ValueError(f"Cannot find transport constituents files from {filename}")

    bounds = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
    dsg = read_grid(gfile, chunks={"nx": nxchunk, "ny": nychunk}, **bounds)
    dsh = read_elevations(hfile, chunks={"nc": None, "nx": nxchunk, "ny": nychunk}, **bounds)
    dsu = read_transports(ufile, chunks={"nc": None, "nx": nxchunk, "ny": nychunk}, **bounds)

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


def indices(lon, lat, x0, x1, y0, y1):
    """Indices of coordinates in lon and lat arrays."""
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


def read_grid(filename, chunks, x0, x1, y0, y1):
    """Read grid netcdf files.

    Args:
        filename (str):
        nx (int): Chunking size along the nx dimension.
        ny (int): Chunking size along the ny dimension.

    Return:
        dset (Dataset): Merged elevations dataset.

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


def read_elevations(filenames, chunks, x0, x1, y0, y1):
    """Read and concatenate individual elevations constituents netcdf files.

    Args:
        filenames (list):
        nx (int): Chunking size along the nx dimension.
        ny (int): Chunking size along the ny dimension.

    Return:
        dset (Dataset): Merged elevations dataset.

    """
    dset = xr.open_mfdataset(
        filenames, concat_dim="nc", combine="nested", chunks=chunks
    )
    ix0, ix1, iy0, iy1 = indices(dset.lon_z, dset.lat_z, x0, x1, y0, y1)
    dset = dset.isel(nx=slice(ix0, ix1), ny=slice(iy0, iy1)).chunk(chunks)
    dset["hRe"] = dset.hRe * 1e-3
    dset["hIm"] = dset.hIm * 1e-3
    return dset


def read_transports(filenames, chunks, x0, x1, y0, y1):
    """Read and concatenate individual transports constituents netcdf files.

    Args:
        filenames (list):
        nx (int): Chunking size along the nx dimension.
        ny (int): Chunking size along the ny dimension.

    Return:
        dset (Dataset): Merged transports dataset.

    """
    dset = xr.open_mfdataset(
        filenames, concat_dim="nc", combine="nested", chunks=chunks
    )
    ix0, ix1, iy0, iy1 = indices(dset.lon_v, dset.lat_u, x0, x1, y0, y1)
    dset = dset.isel(nx=slice(ix0, ix1), ny=slice(iy0, iy1)).chunk(chunks)
    dset["uRe"] = dset.uRe * 1e-2
    dset["uIm"] = dset.uIm * 1e-2
    dset["vRe"] = dset.vRe * 1e-2
    dset["vIm"] = dset.vIm * 1e-2
    return dset.rename({"uRe": "URe", "uIm": "UIm", "vRe": "VRe", "vIm": "VIm"})


if __name__ == "__main__":

    import itertools
    import datetime
    from dask.diagnostics.progress import ProgressBar

    def chunked_iterable(iterable, size):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, size))
            if not chunk:
                break
            yield chunk

    filename = "/data/tide/tpxo9v4_atlas/TPXO9_atlas_nc"
    dset = read_otis_atlas_netcdf(
        filename,
        nxchunk=500,
        nychunk=500,
        x0=165,
        x1=180,
        y0=-48,
        y1=-34,
        )

    # import ipdb; ipdb.set_trace()

    # size = 50
    # for ind in chunked_iterable(range(dset.lon.size), size=size):
    #     i0 = ind[0]
    #     print(f"Writing lon chunk {i0}")
    #     filename = "atlas.zarr"
    #     with ProgressBar():
    #         ds = dset.isel(lon=list(ind)).load()
    #     if i0 == 0:
    #         append_dim = None
    #     else:
    #         append_dim = "lon"
    #     ds.to_zarr(filename, mode="w", consolidated=True, append_dim=append_dim)

    # # eta = dset.tide.predict(times=[datetime.datetime(2012, 1, 1, 0)])
    # # eta = dset.sel(lon=289.52, lat=41.05, method="nearest").tide.predict(
    # #     times=[datetime.datetime(2012, 1, 1, H) for H in range(24)],
    # #     time_chunk=1,
    # # )
    # # with ProgressBar():
    # #     eta.to_netcdf("atlas.nc")

