"""Read Otis Atlas netcdf file format."""
import os
import glob
import dask
import xarray as xr

from oceantide.core.otis import from_otis
from oceantide.tide import Tide


dask.config.set({"array.slicing.split_large_chunks": False})


def read_otis_atlas_netcdf(filename=None, nxchunk=500, nychunk=500):
    """Read Otis Netcdf file format.

    Args:
        filename (str): Name of the folder with all netcdf Atlas files. Atlas files are
            organised as one single file per constituent for u and h and one grid file.
        nxchunk (int): Chunk size along the x-axis in output dataset.
        nychunk (int): Chunk size along the y-axis in output dataset.

    Returns:
        Formatted dataset with the tide accessor.

    """
    gfile = glob.glob(os.path.join(filename, "g*.nc"))
    hfile = glob.glob(os.path.join(filename, "h*.nc"))
    ufile = glob.glob(os.path.join(filename, "u*.nc"))

    if len(gfile) != 1:
        raise ValueError(f"A single grid file cannot be located in {filename}.")
    if not hfile:
        raise ValueError(f"Cannot find elevation constituents files from {filename}")
    if not ufile:
        raise ValueError(f"Cannot find transport constituents files from {filename}")

    dsg = xr.open_dataset(gfile[0], chunks={"nx": nxchunk, "ny": nychunk}).transpose("ny", "nx")
    dsh = read_individual_cons(hfile, chunks={"nx": nxchunk, "ny": nychunk})
    dsu = read_individual_cons(ufile, chunks={"nx": nxchunk, "ny": nychunk})

    mz = dsg.hz.notnull().rename({"nx": "lon_z", "ny": "lat_z"})
    mu = dsg.hu.notnull().rename({"nx": "lon_u", "ny": "lat_u"})
    mv = dsg.hv.notnull().rename({"nx": "lon_v", "ny": "lat_v"})

    dset = xr.Dataset(
        coords={
            "con": dsh.con,
            "lon_z": dsh.lon_z.isel(con=0, drop=True).rename({"nx": "lon_z"}),
            "lat_z": dsh.lat_z.isel(con=0, drop=True).rename({"ny": "lat_z"}),
            "lon_u": dsu.lon_u.isel(con=0, drop=True).rename({"nx": "lon_u"}),
            "lat_u": dsu.lat_u.isel(con=0, drop=True).rename({"ny": "lat_u"}),
            "lon_v": dsu.lon_v.isel(con=0, drop=True).rename({"nx": "lon_v"}),
            "lat_v": dsu.lat_v.isel(con=0, drop=True).rename({"ny": "lat_v"}),
        },
    )
    dset["hz"] = dsg.hz.rename({"nx": "lon_z", "ny": "lat_z"}).where(mz)
    dset["hu"] = dsg.hz.rename({"nx": "lon_u", "ny": "lat_u"}).where(mu)
    dset["hv"] = dsg.hz.rename({"nx": "lon_v", "ny": "lat_v"}).where(mv)
    dset["hRe"] = dsh.hRe.rename({"nx": "lon_z", "ny": "lat_z"}).where(mz)
    dset["hIm"] = dsh.hIm.rename({"nx": "lon_z", "ny": "lat_z"}).where(mz)
    dset["URe"] = dsu.uRe.rename({"nx": "lon_u", "ny": "lat_u"}).where(mu)
    dset["UIm"] = dsu.uIm.rename({"nx": "lon_u", "ny": "lat_u"}).where(mu)
    dset["VRe"] = dsu.vRe.rename({"nx": "lon_v", "ny": "lat_v"}).where(mv)
    dset["VIm"] = dsu.vIm.rename({"nx": "lon_v", "ny": "lat_v"}).where(mv)
    dset["uRe"] = dset["URe"] / dset["hu"]
    dset["uIm"] = dset["UIm"] / dset["hu"]
    dset["vRe"] = dset["VRe"] / dset["hv"]
    dset["vIm"] = dset["VIm"] / dset["hv"]
    dset["con"] = dset.con.astype("S4")

    dset = dset.where(dset < 1e10)
    dset = from_otis(dset)

    return dset


def read_individual_cons(filenames, chunks={"nx": 1000, "ny": 1000}):
    """Read and concatenate individual constituents netcdf files."""
    dset = xr.concat(
        [xr.open_dataset(f).set_coords("con").chunk(chunks) for f in filenames], dim="con"
    ).chunk({"con": None}).transpose("con", "ny", "nx")
    return dset


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

    filename = "/data/tide/tpxo9_atlas_nc"
    dset = read_otis_atlas_netcdf(filename)

    outdir = "./tpxo9-atlas_v9_zarr_slices"
    encoding = {v: {"dtype": "int32", "scale_factor": 1e-3, "_FillValue": -32767} for v in dset.data_vars}
    encoding.update({v: {"chunks": (dset[v].size,)} for v in dset.coords})

    # size = 50
    # for ind in chunked_iterable(range(dset.lon.size), size=size):
    #     i0 = ind[0]
    #     print(f"Writing lon chunk {i0}")
    #     filename = os.path.join(outdir, f"slice_{i0:05.0f}.zarr")
    #     ds = dset.isel(lon=list(ind))
    #     # import ipdb; ipdb.set_trace()
    #     with ProgressBar():
    #         if i0 == 0:
    #             append_dim = None
    #         else:
    #             append_dim = "lon"
    #         ds.to_zarr(filename, consolidated=True, append_dim=append_dim, encoding=encoding)

    # eta = dset.tide.predict(times=[datetime.datetime(2012, 1, 1, 0)])
    # eta = dset.sel(lon=289.52, lat=41.05, method="nearest").tide.predict(
    #     times=[datetime.datetime(2012, 1, 1, H) for H in range(24)],
    #     time_chunk=1,
    # )
    # with ProgressBar():
    #     eta.to_netcdf("atlas.nc")

