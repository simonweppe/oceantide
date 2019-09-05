# -*- coding: utf-8 -*-

"""OTIS tools."""

import numpy as np
import xarray as xr
import pyroms._remapping.flood as fflood


def load_otis(grdfile, version="v1", xlim=None, ylim=None, missing_value=-9999):
    """
    grd = load_otis(grdfile)

    Load Cgrid object for OTIS from netCDF file
    """
    consfile = ".".join([grdfile.split(".")[0].replace("grid", "h"), version, "nc"])

    ds = xr.open_dataset(grdfile)
    dsh = xr.open_dataset(consfile)

    lonh = ds.lon_z.values
    lath = ds.lat_z.values
    lonu = ds.lon_u.values
    latu = ds.lat_u.values
    lonv = ds.lon_v.values
    latv = ds.lat_v.values

    zh = ds.hz.values + 0.001  # avoid zero-division, otis land values are zero :/
    zu = ds.hu.values + 0.001
    zv = ds.hv.values + 0.001

    # land mask
    hmask = zh != 0
    umask = zu != 0
    vmask = zv != 0

    # longitude from -180 to 180
    lonh[lonh > 180] = lonh[lonh > 180] - 360
    lonu[lonu > 180] = lonu[lonu > 180] - 360
    lonv[lonv > 180] = lonv[lonv > 180] - 360

    cons = dsh.con.values.tostring().decode().split()
    cons = [c.upper() for c in cons]

    grid = CGrid_OTIS(
        cons,
        lonh,
        lath,
        lonu,
        latu,
        lonv,
        latv,
        hmask,
        umask,
        vmask,
        zh,
        zu,
        zv,
        missing_value,
        xlim,
        ylim,
    )

    return grid


def flood(otis, dmax=1):
    """
    Flood variables into land to avoid spurious values close to the coast

    Args:
        otis (xarray.Dataset):   OTIS dataset 
        dmax (int>0):              Maximum horizontal flooding distance

    """

    msk = np.isnan(dsu.URe.values[0, ...])

    for k in range(otis.dims["nc"] - 1, 0, -1):
        idxnan = np.where(msk == True)
        idx = np.where(msk == False)

        if list(idx[0]):
            wet = np.zeros((len(idx[0]), 2))
            dry = np.zeros((len(idxnan[0]), 2))
            wet[:, 0] = idx[0] + 1
            wet[:, 1] = idx[1] + 1
            dry[:, 0] = idxnan[0] + 1
            dry[:, 1] = idxnan[1] + 1

        for varname, var in otis.data_vars.items():
            if len(var.dims) > 2:  # leaving cons and coords out
                var.values[k, ...] = fflood(
                    var.values[k, ...], wet, dry, otis[lonv], otis.latv, dmax
                )

    return otis


class CGrid_OTIS(object):

    # CGrid object for OTIS

    def __init__(
        self,
        cons,
        lon_t,
        lat_t,
        lon_u,
        lat_u,
        lon_v,
        lat_v,
        mask_t,
        mask_u,
        mask_v,
        z_t,
        z_u,
        z_v,
        missing_value,
        xlim,
        ylim,
    ):
        self.cons = cons
        self.name = "otis"

        self.null = -9999.0

        f = np.where(
            (lon_t >= xlim[0])
            & (lon_t <= xlim[1])
            & (lat_t >= ylim[0])
            & (lat_t <= ylim[1])
        )
        x0, x1, y0, y1 = f[1].min(), f[1].max(), f[0].min(), f[0].max()

        self.z_t = z_t[y0 : y1 + 1, x0 : x1 + 1]

        self.z_u = z_u[y0 : y1 + 1, x0 : x1 + 1]
        self.z_v = z_v[y0 : y1 + 1, x0 : x1 + 1]

        self.lon_t = lon_t[y0 : y1 + 1, x0 : x1 + 1]
        self.lat_t = lat_t[y0 : y1 + 1, x0 : x1 + 1]

        self.lon_u = lon_u[y0 : y1 + 1, x0 : x1 + 1]
        self.lat_u = lat_u[y0 : y1 + 1, x0 : x1 + 1]
        self.lon_v = lon_v[y0 : y1 + 1, x0 : x1 + 1]
        self.lat_v = lat_v[y0 : y1 + 1, x0 : x1 + 1]

        self.lon_t_vert = 0.5 * (
            lon_t[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lon_t[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_t_vert = 0.5 * (
            lat_t[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lat_t[y0 : y1 + 2, x0 : x1 + 2]
        )

        self.lon_u_vert = 0.5 * (
            lon_u[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lon_u[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_u_vert = 0.5 * (
            lat_u[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lat_u[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lon_v_vert = 0.5 * (
            lon_v[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lon_v[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_v_vert = 0.5 * (
            lat_v[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lat_v[y0 : y1 + 2, x0 : x1 + 2]
        )

        self.mask_t = mask_t[y0 : y1 + 1, x0 : x1 + 1]
        self.mask_u = mask_u[y0 : y1 + 1, x0 : x1 + 1]
        self.mask_v = mask_v[y0 : y1 + 1, x0 : x1 + 1]

        ones = np.ones(self.z_t.shape)
        a1 = lat_u[y0 : y1 + 1, x0 + 1 : x1 + 2] - lat_u[y0 : y1 + 1, x0 : x1 + 1]
        a2 = lon_u[y0 : y1 + 1, x0 + 1 : x1 + 2] - lon_u[y0 : y1 + 1, x0 : x1 + 1]
        a3 = 0.5 * (
            lat_u[y0 : y1 + 1, x0 + 1 : x1 + 2] + lat_u[y0 : y1 + 1, x0 : x1 + 1]
        )
        a2 = np.where(a2 > 180 * ones, a2 - 360 * ones, a2)
        a2 = np.where(a2 < -180 * ones, a2 + 360 * ones, a2)
        a2 = a2 * np.cos(np.pi / 180.0 * a3)
        self.angle = np.arctan2(a1, a2)
