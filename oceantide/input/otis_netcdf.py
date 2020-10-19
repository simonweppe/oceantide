"""Read Otis netcdf file format."""
import os
import xarray as xr

from oceantide.core.otis import from_otis, otis_filenames
from oceantide.tide import Tide


def read_otis_netcdf(filename, gfile=None, hfile=None, ufile=None):
    """Read Otis Netcdf file format.

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
        * Grid file named as `grid*.nc` with grid information and model depths.
        * Elevation file named as `h*.*.nc` with elevation constituents data.
        * Transport file named as `uv.*.nc` with transport constituents data.
        The path of the three data files can be prescribed either by specifying the
            meta file path (`filename`) or by explicitly providing their path
            (`gfile`, `hfile` and `ufile`).

    """
    dirname = os.path.dirname(filename)
    _gfile, _hfile, _ufile = otis_filenames(filename)
    gfile = gfile or os.path.join(dirname, _gfile)
    hfile = hfile or os.path.join(dirname, _hfile)
    ufile = ufile or os.path.join(dirname, _ufile)

    dsg = xr.open_dataset(gfile, chunks={}).transpose("ny", "nx", ...)
    dsh = xr.open_dataset(hfile, chunks={}).transpose("nc", "ny", "nx", ...)
    dsu = xr.open_dataset(ufile, chunks={}).transpose("nc", "ny", "nx", ...)

    mz = dsg.mz.rename({"nx": "lon_z", "ny": "lat_z"})
    mu = dsg.mz.rename({"nx": "lon_u", "ny": "lat_u"})
    mv = dsg.mz.rename({"nx": "lon_v", "ny": "lat_v"})

    dset = xr.Dataset(
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
    dset["hz"] = dsg.hz.rename({"nx": "lon_z", "ny": "lat_z"}).where(mz)
    dset["hu"] = dsg.hz.rename({"nx": "lon_u", "ny": "lat_u"}).where(mu)
    dset["hv"] = dsg.hz.rename({"nx": "lon_v", "ny": "lat_v"}).where(mv)
    dset["hRe"] = dsh.hRe.rename({"nc": "con", "nx": "lon_z", "ny": "lat_z"}).where(mz)
    dset["hIm"] = dsh.hIm.rename({"nc": "con", "nx": "lon_z", "ny": "lat_z"}).where(mz)
    dset["URe"] = dsu.URe.rename({"nc": "con", "nx": "lon_u", "ny": "lat_u"}).where(mu)
    dset["UIm"] = dsu.UIm.rename({"nc": "con", "nx": "lon_u", "ny": "lat_u"}).where(mu)
    dset["VRe"] = dsu.VRe.rename({"nc": "con", "nx": "lon_v", "ny": "lat_v"}).where(mv)
    dset["VIm"] = dsu.VIm.rename({"nc": "con", "nx": "lon_v", "ny": "lat_v"}).where(mv)
    dset["uRe"] = dset["URe"] / dset["hu"]
    dset["uIm"] = dset["UIm"] / dset["hu"]
    dset["vRe"] = dset["VRe"] / dset["hv"]
    dset["vIm"] = dset["VIm"] / dset["hv"]
    dset["con"] = dset.con.astype("S4")

    dset = from_otis(dset)
    return dset


if __name__ == "__main__":

    filename = "/data/tide/otis_tpxo8_raw_files/netcdf/tmp/DATA/Model_ES2008"
    dset = read_otis_netcdf(filename)