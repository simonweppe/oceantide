"""Otis binary output."""
from pathlib import Path
import numpy as np
import xarray as xr

from oceantide.core.otis import u_from_z, v_from_z, indices_open_boundary
from oceantide.core.utils import set_attributes


def to_otis_netcdf(
    self,
    dirname: str,
    hfile: bool = True,
    ufile: bool = True,
    gfile: bool = True,
    suffix: bool = None,
) -> dict:
    """Write dataset as Otis binary format.

    Parameters
    ----------
    self (oceantide.tide.Tide)
        Oceantide Tide instance.
    dirname (str)
        Directory to save binary files.
    hfile (bool)
        Save tidal elevation binary file.
    ufile (bool)
        Save tidal transports binary file.
    gfile (bool)
        Save grid file.
    suffix (str)
        Suffix to define file names, by default defined by cons names.

    Returns
    -------
    filename (dict)
        Name of files written.

    """
    ds = self._obj.transpose("con", "lon", "lat", ...)
    ds = ds.rename({"con": "nc", "lon": "nx", "lat": "ny"})
    ds = ds.fillna(0.0)

    if suffix is None:
        suffix = "".join(list(ds.nc.values)).lower()

    lat_z, lon_z = np.meshgrid(ds.ny, ds.nx)
    dx = float(ds.nx[1] - ds.nx[0])
    dy = float(ds.ny[1] - ds.ny[0])
    ds = ds.drop(["nc", "nx", "ny"])
    cons = np.array([f"{c:4s}".lower() for c in self._obj.con.values], dtype="S4")

    mz = xr.where(ds.dep > 0, 1.0, 0.0)
    if ufile or gfile:
        mu = mz * mz.roll(nx=1, roll_coords=False)
        mv = mz * mz.roll(ny=1, roll_coords=False)
        lon_u = lon_z - dx
        lat_u = lat_z
        lon_v = lon_z
        lat_v = lat_z - dy

    filenames = {}

    # Write elevations
    if hfile:
        fname = f"h_{suffix.lstrip('_')}.nc" if suffix else "h.nc"
        filenames["hfile"] = Path(dirname) / fname

        ha = self.amplitude("h").transpose("con", "lon", "lat", ...)
        ha = ha.rename({"con": "nc", "lon": "nx", "lat": "ny"}).drop(["nc", "nx", "ny"])
        hp = self.phase("h").transpose("con", "lon", "lat", ...)
        hp = hp.rename({"con": "nc", "lon": "nx", "lat": "ny"}).drop(["nc", "nx", "ny"])
        dsout = xr.Dataset()
        dsout["con"] = xr.DataArray(cons, dims=("nc"))
        dsout["lon_z"] = xr.DataArray(lon_z, dims=("nx", "ny"))
        dsout["lat_z"] = xr.DataArray(lat_z, dims=("nx", "ny"))
        dsout["ha"] = xr.DataArray(ha, dims=("nc", "nx", "ny"))
        dsout["hp"] = xr.DataArray(hp, dims=("nc", "nx", "ny"))
        dsout["hRe"] = xr.DataArray(ds.h.real, dims=("nc", "nx", "ny"))
        dsout["hIm"] = xr.DataArray(ds.h.imag, dims=("nc", "nx", "ny"))

        set_attributes(dsout, "otis")
        dsout.attrs = {"title": "OTIS tidal elevation file", "source": "Oceantide"}

        dsout.to_netcdf(filenames["hfile"])

    # Write transports
    if ufile:
        fname = f"u_{suffix.lstrip('_')}.nc" if suffix else "u.nc"
        filenames["ufile"] = Path(dirname) / fname

        ua = self.amplitude("u").transpose("con", "lon", "lat", ...)
        ua = ua.rename({"con": "nc", "lon": "nx", "lat": "ny"}).drop(["nc", "nx", "ny"])
        up = self.phase("u").transpose("con", "lon", "lat", ...)
        up = up.rename({"con": "nc", "lon": "nx", "lat": "ny"}).drop(["nc", "nx", "ny"])
        va = self.amplitude("v").transpose("con", "lon", "lat", ...)
        va = va.rename({"con": "nc", "lon": "nx", "lat": "ny"}).drop(["nc", "nx", "ny"])
        vp = self.phase("v").transpose("con", "lon", "lat", ...)
        vp = vp.rename({"con": "nc", "lon": "nx", "lat": "ny"}).drop(["nc", "nx", "ny"])
        dsout = xr.Dataset()
        dsout["con"] = dsout["con"] = xr.DataArray(cons, dims=("nc"))
        dsout["lon_u"] = xr.DataArray(lon_u, dims=("nx", "ny"))
        dsout["lat_u"] = xr.DataArray(lat_u, dims=("nx", "ny"))
        dsout["lon_v"] = xr.DataArray(lon_v, dims=("nx", "ny"))
        dsout["lat_v"] = xr.DataArray(lat_v, dims=("nx", "ny"))
        dsout["ua"] = u_from_z(xr.DataArray(ua, dims=("nc", "nx", "ny")), mz, mu)
        dsout["up"] = u_from_z(xr.DataArray(up, dims=("nc", "nx", "ny")), mz, mu)
        dsout["va"] = v_from_z(xr.DataArray(va, dims=("nc", "nx", "ny")), mz, mv)
        dsout["vp"] = v_from_z(xr.DataArray(vp, dims=("nc", "nx", "ny")), mz, mv)
        dsout["URe"] = u_from_z(
            xr.DataArray(ds.u.real * ds.dep, dims=("nc", "nx", "ny")), mz, mu
        )
        dsout["UIm"] = u_from_z(
            xr.DataArray(ds.u.imag * ds.dep, dims=("nc", "nx", "ny")), mz, mu
        )
        dsout["VRe"] = v_from_z(
            xr.DataArray(ds.v.real * ds.dep, dims=("nc", "nx", "ny")), mz, mv
        )
        dsout["VIm"] = v_from_z(
            xr.DataArray(ds.v.imag * ds.dep, dims=("nc", "nx", "ny")), mz, mv
        )

        set_attributes(dsout, "otis")
        dsout.attrs = {
            "title": "OTIS tidal transport/current file",
            "source": "Oceantide",
        }

        dsout.to_netcdf(filenames["ufile"])

    # Write grid
    if gfile:
        fname = f"grid_{suffix.lstrip('_')}.nc" if suffix else "grid.nc"
        filenames["gfile"] = Path(dirname) / fname

        dsout = xr.Dataset()
        dsout["lon_z"] = xr.DataArray(lon_z, dims=("nx", "ny"))
        dsout["lat_z"] = xr.DataArray(lat_z, dims=("nx", "ny"))
        dsout["lon_u"] = xr.DataArray(lon_u, dims=("nx", "ny"))
        dsout["lat_u"] = xr.DataArray(lat_u, dims=("nx", "ny"))
        dsout["lon_v"] = xr.DataArray(lon_v, dims=("nx", "ny"))
        dsout["lat_v"] = xr.DataArray(lat_v, dims=("nx", "ny"))
        dsout["mz"] = xr.DataArray(mz, dims=("nx", "ny"))
        dsout["mu"] = xr.DataArray(mu, dims=("nx", "ny"))
        dsout["mv"] = xr.DataArray(mv, dims=("nx", "ny"))
        dsout["hz"] = xr.DataArray(ds.dep, dims=("nx", "ny"))
        dsout["hu"] = u_from_z(xr.DataArray(ds.dep, dims=("nx", "ny")), mz, mu)
        dsout["hv"] = v_from_z(xr.DataArray(ds.dep, dims=("nx", "ny")), mz, mv)
        dsout["iob_z"] = xr.DataArray(
            indices_open_boundary(dsout.mz), dims=("iiob", "nob_z")
        )
        dsout["iob_u"] = xr.DataArray(
            indices_open_boundary(dsout.mu), dims=("iiob", "nob_u")
        )
        dsout["iob_v"] = xr.DataArray(
            indices_open_boundary(dsout.mv), dims=("iiob", "nob_v")
        )

        set_attributes(dsout, "otis")
        dsout.attrs = {"title": "OTIS Arakawa C-grid file", "source": "Oceantide"}

        dsout.to_netcdf(filenames["gfile"])

    fname = f"model_{suffix.lstrip('_')}" if suffix else "model"
    with open(Path(dirname) / fname, mode="w") as stream:
        for filename in filenames.values():
            stream.write(f"./{filename.name}\n")

    return filenames


if __name__ == "__main__":
    from oceantide import read_otis_netcdf

    dset = read_otis_netcdf("/data/tide/tpxo9v4a/netcdf/DATA/Model_tpxo9v4a")
    dset = dset.isel(lon=slice(None, None, 10), lat=slice(None, None, 10)).load()
    filenames = dset.tide.to_otis_netcdf("./", hfile=True, ufile=True, gfile=True)
    print(filenames)
