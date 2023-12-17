"""Otis binary output."""
from pathlib import Path
import numpy as np

from oceantide.core.otis import (
    write_otis_bin_h,
    write_otis_bin_u,
    write_otis_bin_grid,
    u_from_z,
    v_from_z,
)


def to_otis_binary(
    self,
    dirname: str,
    hfile: bool = True,
    ufile: bool = True,
    gfile: bool = True,
    suffix: str = None,
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

    filenames = {}

    # Write elevations
    if hfile:
        fname = f"h_{suffix.lstrip('_')}" if suffix else "h"
        filenames["hfile"] = Path(dirname) / fname
        write_otis_bin_h(
            hfile=filenames["hfile"],
            hRe=ds.h.real,
            hIm=ds.h.imag,
            con=ds.nc,
            lon=ds.nx,
            lat=ds.ny,
        )

    # Write transports
    if ufile:
        fname = f"u_{suffix.lstrip('_')}" if suffix else "u"
        filenames["ufile"] = Path(dirname) / fname
        mz = ds.dep > 0
        mu = mz * mz.roll(nx=1, roll_coords=False)
        mv = mz * mz.roll(ny=1, roll_coords=False)
        write_otis_bin_u(
            ufile=filenames["ufile"],
            URe=u_from_z(ds.u.real * ds.dep, mz, mu),
            UIm=u_from_z(ds.u.imag * ds.dep, mz, mu),
            VRe=v_from_z(ds.v.real * ds.dep, mz, mv),
            VIm=v_from_z(ds.v.imag * ds.dep, mz, mv),
            con=ds.nc,
            lon=ds.nx,
            lat=ds.ny,
        )

    # Write grid
    if gfile:
        fname = f"grid_{suffix.lstrip('_')}" if suffix else "grid"
        filenames["gfile"] = Path(dirname) / fname
        write_otis_bin_grid(
            gfile=filenames["gfile"],
            hz=ds.dep,
            mz=mz,
            lon=ds.nx,
            lat=ds.ny,
            dt=12.0,
        )

    fname = f"model_{suffix.lstrip('_')}" if suffix else "model"
    with open(Path(dirname) / fname, mode="w") as stream:
        for filename in filenames.values():
            stream.write(f"./{filename.name}\n")

    return filenames
