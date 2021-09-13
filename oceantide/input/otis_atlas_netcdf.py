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
    dirname, x0=None, x1=None, y0=None, y1=None, nxchunk=500, nychunk=500
):
    """Read Otis Netcdf file format.

    Args:
        - dirname (str): Path name with all netcdf Atlas files. Atlas files are
          organised as one grid file and one file per constituent for uv and h.
        - x0 (float): Longitude left corner to read.
        - x1 (float): Longitude right corner to read.
        - y0 (float): Latitude bottom corner to read.
        - y1 (float): Latitude top corner to read.
        - nxchunk (int): Chunk size along the x-axis.
        - nychunk (int): Chunk size along the y-axis.

    Returns:
        - Formatted dataset with the tide accessor.

    Note:
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
        - filename (str):
        - nx (int): Chunking size along the nx dimension.
        - ny (int): Chunking size along the ny dimension.

    Return:
        - dset (Dataset): Merged elevations dataset.

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
        - filenames (list):
        - nx (int): Chunking size along the nx dimension.
        - ny (int): Chunking size along the ny dimension.

    Return:
        - dset (Dataset): Merged elevations dataset.

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
        - filenames (list):
        - nx (int): Chunking size along the nx dimension.
        - ny (int): Chunking size along the ny dimension.

    Return:
        - dset (Dataset): Merged transports dataset.

    """
    dset = xr.open_mfdataset(
        filenames, concat_dim="nc", combine="nested", chunks=chunks
    )
    ix0, ix1, iy0, iy1 = indices(dset.lon_v, dset.lat_u, x0, x1, y0, y1)
    dset = dset.isel(nx=slice(ix0, ix1), ny=slice(iy0, iy1)).chunk(chunks)
    dset["uRe"] = dset.uRe * 1e-1
    dset["uIm"] = dset.uIm * 1e-1
    dset["vRe"] = dset.vRe * 1e-1
    dset["vIm"] = dset.vIm * 1e-1
    return dset.rename({"uRe": "URe", "uIm": "UIm", "vRe": "VRe", "vIm": "VIm"})


if __name__ == "__main__":
    import itertools
    import warnings
    from pathlib import Path
    import datetime
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt

    # from oceantide import read_otis_atlas_netcdf, read_oceantide

    warnings.filterwarnings("ignore", category=RuntimeWarning)


    atlas_path = Path("/data/tide/tpxo9v4_atlas/TPXO9_atlas_nc")

    dset = read_otis_atlas_netcdf(
        dirname=atlas_path,
        x0=356,
        x1=358,
        y0=55.5,
        y1=57,
        # nxchunk=500,
        # nychunk=500,
    ).load()

    dset.tide.amplitude("u").isel(con=0).plot()
    plt.show()

    def chunked_iterable(iterable, size):
        """Iterate through array over chunks with specific size."""
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, size))
            if not chunk:
                break
            yield chunk

    # lons = xr.open_dataset(atlas_path / "grid_tpxo9_atlas_v4.nc").lon_z.values

    # chunksizes = 100
    # itersize = 200

    # for ind in chunked_iterable(range(lons.size), size=itersize):
    #     x0 = lons[ind[0]]
    #     x1 = lons[ind[-1]]
    #     print(f"Writing lon chunk {ind[0]}")
    #     dset = read_otis_atlas_netcdf(atlas_path, x0=x0, x1=x1, nxchunk=itersize, nychunk=None)
    #     dset = dset.chunk({"con": None, "lon": chunksizes, "lat": chunksizes})
    #     if ind[0] == 0:
    #         mode = "w"
    #         append_dim = None
    #     else:
    #         mode = "a"
    #         append_dim = "lon"
    #     dset.to_zarr("./atlas.zarr", mode=mode, consolidated=True, append_dim=append_dim)

