# -*- coding: utf-8 -*-

"""OTIS tools."""

import numpy as np
import xarray as xr
import pyroms


def load_otis(grdfile, version="v1", xlim=None, ylim=None, missing_value=-9999):
    """
    grd = load_otis(grdfile)

    Load Cgrid object for OTIS from netCDF file
    """
    consfile = '.'.join([grdfile.split('.')[0].replace('grid', 'h'), version, 'nc'])
    
    ds = xr.open_dataset(grdfile)
    dsh = xr.open_dataset(consfile)

    lonh = ds.lon_z.values
    lath = ds.lat_z.values
    lonu = ds.lon_u.values
    latu = ds.lat_u.values
    lonv = ds.lon_v.values
    latv = ds.lat_v.values

    zh = ds.hz.values
    zu = ds.hu.values
    zv = ds.hv.values

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

    grid = CGrid_OTIS(cons, lonh, lath, lonu, latu, lonv, latv, hmask, umask, vmask, zh, zu, zv, missing_value, xlim, ylim)

    return grid


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
        self.name = 'otis'

        self.null = -9999.0

        x0, x1 = xlim[0], xlim[1]
        y0, y1 = ylim[0], ylim[1]

        self.z_t = z_t[y0:y1 + 1, x0:x1 + 1]

        self.z_u = z_u[y0:y1 + 1, x0:x1 + 1]
        self.z_v = z_v[y0:y1 + 1, x0:x1 + 1]

        self.lon_t = lon_t[y0:y1 + 1, x0:x1 + 1]
        self.lat_t = lat_t[y0:y1 + 1, x0:x1 + 1]

        self.lon_u = lon_u[y0:y1 + 1, x0:x1 + 1]
        self.lat_u = lat_u[y0:y1 + 1, x0:x1 + 1]
        self.lon_v = lon_v[y0:y1 + 1, x0:x1 + 1]
        self.lat_v = lat_v[y0:y1 + 1, x0:x1 + 1]

        self.lon_t_vert = 0.5 * (
            lon_t[y0 - 1:y1 + 1, x0 - 1:x1 + 1] + lon_t[y0:y1 + 2, x0:x1 + 2]
        )
        self.lat_t_vert = 0.5 * (
            lat_t[y0 - 1:y1 + 1, x0 - 1:x1 + 1] + lat_t[y0:y1 + 2, x0:x1 + 2]
        )

        self.lon_u_vert = 0.5 * (
            lon_u[y0 - 1:y1 + 1, x0 - 1:x1 + 1] + lon_u[y0:y1 + 2, x0:x1 + 2]
        )
        self.lat_u_vert = 0.5 * (
            lat_u[y0 - 1:y1 + 1, x0 - 1:x1 + 1] + lat_u[y0:y1 + 2, x0:x1 + 2]
        )
        self.lon_v_vert = 0.5 * (
            lon_v[y0 - 1:y1 + 1, x0 - 1:x1 + 1] + lon_v[y0:y1 + 2, x0:x1 + 2]
        )
        self.lat_v_vert = 0.5 * (
            lat_v[y0 - 1:y1 + 1, x0 - 1:x1 + 1] + lat_v[y0:y1 + 2, x0:x1 + 2]
        )

        self.mask_t = mask_t[y0:y1 + 1, x0:x1 + 1]
        self.mask_u = mask_u[y0:y1 + 1, x0:x1 + 1]
        self.mask_v = mask_v[y0:y1 + 1, x0:x1 + 1]

        ones = np.ones(self.z_t.shape)
        a1 = lat_u[y0:y1 + 1, x0 + 1:x1 + 2] - lat_u[y0:y1 + 1, x0:x1 + 1]
        a2 = lon_u[y0:y1 + 1, x0 + 1:x1 + 2] - lon_u[y0:y1 + 1, x0:x1 + 1]
        a3 = 0.5 * (
            lat_u[y0:y1 + 1, x0 + 1:x1 + 2] + lat_u[y0:y1 + 1, x0:x1 + 1]
        )
        a2 = np.where(a2 > 180 * ones, a2 - 360 * ones, a2)
        a2 = np.where(a2 < -180 * ones, a2 + 360 * ones, a2)
        a2 = a2 * np.cos(np.pi / 180.0 * a3)
        self.angle = np.arctan2(a1, a2)