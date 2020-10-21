"""Read Otis binary file format."""
import os
import dask.array as da
import xarray as xr

from oceantide.core.otis import (
    from_otis,
    otis_filenames,
    read_otis_bin_cons,
    read_otis_bin_grid,
    read_otis_bin_h,
    read_otis_bin_uv,
)
from oceantide import Tide


def read_otis_binary(filename=None, gfile=None, hfile=None, ufile=None):
    """Read Otis binary file format.

    Args:
        filename (str): Name of `Model_*` metadata file specifying other files to read.
        gfile (str): Name of grid file to read, by default defined from `filename`.
        hfile (str): Name of elevation file to read, by default defined from `filename`.
        ufile (str): Name of currents file to read, by default defined from `filename`.

    Returns:
        Formatted dataset with the tide accessor.

    Note:
        Otis data are provided in 4 separate files:
        * Meta file named as `Model_*` specifyig the 3 data files to read.
        * Grid file named as `grid*` with grid information and model depths.
        * Elevation file named as `h*.` with elevation constituents data.
        * Transport file named as `uv.` with transport constituents data.
        The path of the three data files can be prescribed either by specifying the
            meta file path (`filename`) or by explicitly providing their path
            (`gfile`, `hfile` and `ufile`).

    """
    if filename is not None:
        _gfile, _hfile, _ufile = otis_filenames(filename)
    else:
        _gfile = _hfile = _ufile = None
        if not all([gfile, hfile, ufile]):
            raise ValueError(
                "Either specify `filename` or all of `gfile`, `hfile`, `ufile`."
            )

    gfile = gfile or _gfile
    hfile = hfile or _hfile
    ufile = ufile or _ufile

    lon_z, lat_z, lon_u, lat_u, lon_v, lat_v, hz, mz = read_otis_bin_grid(gfile)
    cons = read_otis_bin_cons(hfile)
    hRe, hIm = read_otis_bin_h(hfile)
    URe, UIm, VRe, VIm = read_otis_bin_uv(ufile)

    dset = xr.Dataset(
        coords={
            "con": cons,
            "lat_z": lat_z,
            "lon_z": lon_z,
            "lat_u": lat_u,
            "lon_u": lon_u,
            "lat_v": lat_v,
            "lon_v": lon_v,
        },
    )

    # mu, mv, hu, hv are not currently read from binaries, assumed same as mz, hz
    mask = mz
    mz = data_array(mask, ("lat_z", "lon_z"), dset).astype(bool)
    mu = data_array(mask, ("lat_u", "lon_u"), dset).astype(bool)
    mv = data_array(mask, ("lat_v", "lon_v"), dset).astype(bool)

    dset["hz"] = data_array(hz, ("lat_z", "lon_z"), dset).where(mz)
    dset["hu"] = data_array(hz, ("lat_u", "lon_u"), dset).where(mu)
    dset["hv"] = data_array(hz, ("lat_v", "lon_v"), dset).where(mv)
    dset["hRe"] = data_array(hRe, ("con", "lat_z", "lon_z"), dset).where(mz)
    dset["hIm"] = data_array(hIm, ("con", "lat_z", "lon_z"), dset).where(mz)
    dset["URe"] = data_array(URe, ("con", "lat_u", "lon_u"), dset).where(mu)
    dset["UIm"] = data_array(UIm, ("con", "lat_u", "lon_u"), dset).where(mu)
    dset["VRe"] = data_array(VRe, ("con", "lat_v", "lon_v"), dset).where(mv)
    dset["VIm"] = data_array(VIm, ("con", "lat_v", "lon_v"), dset).where(mv)
    dset["uRe"] = dset["URe"] / dset["hu"]
    dset["uIm"] = dset["UIm"] / dset["hu"]
    dset["vRe"] = dset["VRe"] / dset["hv"]
    dset["vIm"] = dset["VIm"] / dset["hv"]

    dset = dset.where(dset < 1e10)
    dset = from_otis(dset)

    return dset


def data_array(data, dims, dset_template):
    """Create DataArray from template.

    Args:
        data (array): Data.
        dims (list): Dimension names.
        dset_template (Dataset): Dataset template containing dims.

    Returns:
        DataArray.

    """
    coords = {v: dset_template[v] for v in dims}
    return xr.DataArray(da.from_array(data), coords=coords, dims=dims)
