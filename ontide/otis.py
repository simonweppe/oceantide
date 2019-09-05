# -*- coding: utf-8 -*-

"""OTIS tools."""

import numpy as np
import xarray as xr
from fsspec import filesystem, get_mapper
import pyroms


OTIS_VERSION = "v1"


class NCOtis(object):
    """ Object to replace CGrid_OTIS and handle h, u and v at once

    Args: 
        otis_grid (str):   Filename of the regional OTIS grid in fspec format
                           Ex: gs://oceanum-static/otis/grid_tpxo9.nc

    Developer notes:
        - this should replace both .load_otis and .CGrid_OTIS
        - should receive the GRID and figure out path of U and V files
        - sort out dimensions and wrap it all in a single xr.Dataset
        - migrate flood here as a method

    """

    self.name = "otis"  # necessary for remap to work nicely

    def __init__(self, otis_grid="file:///data/otis/grid_tpxo9.nc"):
        gridfile = get_mapper(otis_grid).root
        elevfile = ".".join([grdfile.split(".")[0].replace("grid", "h"), version, "nc"])
        curfile = ".".join([grdfile.split(".")[0].replace("grid", "u"), version, "nc"])

        dsg = xr.open_dataset(grdfile)
        dsh = xr.open_dataset(elevfile)
        dsu = xr.open_dataset(curfile)

        dsu = dsu.drop("lon_u")
        dsu = dsu.drop("lat_u")
        dsu = dsu.drop("lon_v")
        dsu = dsu.drop("lat_v")
        dsh = dsh.drop("lat_z")
        dsh = dsh.drop("lon_z")
        self.ds = xr.merge([dsg, dsh, dsu])

        self._fix_topo()
        self._fix_east()
        self._mask_vars()

        # STOPPED HERE
        # NOTES:
        #   - lefts contents of load_otis and CGrid_OTIS below for guidance
        #   - idea is to try and modify xr.DataArrays in place
        #   - suggested methods drafted in the end of this __init__()

        lonh = dsg.lon_z.values
        lath = dsg.lat_z.values
        lonu = dsg.lon_u.values
        latu = dsg.lat_u.values
        lonv = dsg.lon_v.values
        latv = dsg.lat_v.values

        # land mask
        hmask = zh != 0
        umask = zu != 0
        vmask = zv != 0

        # longitude from -180 to 180
        lonh[lonh > 180] = lonh[lonh > 180] - 360
        lonu[lonu > 180] = lonu[lonu > 180] - 360
        lonv[lonv > 180] = lonv[lonv > 180] - 360
        

    def _fix_topo(self):
        """ Make topography values above zero as they are inconvenently = 0

        """
        self.ds.hz.values += 0.001
        self.ds.hu.values += 0.001
        self.ds.hv.values += 0.001

    def _fix_east(self):
        """ Convert 0 < lon < 360 to -180 < lon < 180

        """
        # could gran idea from seapy *east functions
        # watch as the data arrays need to be fliped and concatenated
        pass

    def _mask_vars(self):
        """ Apply mask to vars as land values are inconveniently = 0

        """
        pass

    @property
    def cons(self):
        """ Nicely formatted cons attribute

        """
        cons = self.ds.con.values.tostring().decode().split()
        cons = [c.upper() for c in cons]
        return cons

    def subset(x0, x1, y0, y1):
        """
        Developer notes:
            - Done this in a rush, should probably abstract it better
            - Idea here is that vertices and angle need derivatives, which 
              cannot be computed in the global or regional grid, otherwise
              we'd end up having to cut a couple of lines/cols at the boundaries
            - As we are subsetting, we end up with spare lines/cols, so it allows
              it to be done in a clean way. 
            - Before anything, it needs to be checked if these vertices and angles
              are being used for anything we need.  
        """
        f = np.where(
            (self.ds.lon_z.values >= x0)
            & (self.ds.lon_z.values <= x1)
            & (self.ds.lat_z.values >= y0)
            & (self.ds.lat_z.values <= y1)
        )
        x0, x1, y0, y1 = f[1].min(), f[1].max(), f[0].min(), f[0].max()

        # compute vertices and angle before we lose the indexes in isel
        self.lon_t_vert = 0.5 * (
            self.lon_z.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.lon_z.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_t_vert = 0.5 * (
            self.lat_z.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.lat_z.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lon_u_vert = 0.5 * (
            self.lon_u.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.lon_u.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_u_vert = 0.5 * (
            self.lat_u.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.lat_u.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lon_v_vert = 0.5 * (
            self.lon_v.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.lon_v.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_v_vert = 0.5 * (
            self.lat_v.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.lat_v.values[y0 : y1 + 2, x0 : x1 + 2]
        )

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

        # finally, subsetting
        self.ds = self.ds.isel(nx=slice(x0, x1), ny=slice(y0, y1))


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
        otis (xarray.Dataset):   OTIS dataset - must be masked beforehand
        dmax (int>0):            Maximum horizontal flooding distance

    Developer notes:
        - this should be a method of CGridOTIS object once it's refactored (NCOtis)
    """
    # getting mask from file now, but should be handled by future NCOtis in the future
    # figuring out a 3D var to get the mask from
    for varname, var in otis.data_vars.items():
        if len(var.dims) < 3:
            continue
        else:
            var3d = varname
            break

    msk = np.isnan(otis[var3d].values[0, ...])

    for k in range(otis.dims["nc"]):
        for varname, var in otis.data_vars.items():
            if len(var.dims) > 2:  # leaving cons and coords out
                # same as above, this should be better handled when dims and coords are tidied up into
                # a single OTIS xr.Dataset
                for varname2, var2 in otis.data_vars.items():
                    if varname2.startswith("lon"):
                        lonv = varname2
                    if varname2.startswith("lat"):
                        latv = varname2

                msk = np.isnan(var.values[k, ...])
                idxnan = np.where(msk == True)
                idx = np.where(msk == False)

                if list(idx[0]):
                    wet = np.zeros((len(idx[0]), 2))
                    dry = np.zeros((len(idxnan[0]), 2))
                    wet[:, 0] = idx[0] + 1
                    wet[:, 1] = idx[1] + 1
                    dry[:, 0] = idxnan[0] + 1
                    dry[:, 1] = idxnan[1] + 1

                var.values[k, ...] = pyroms._remapping.flood(
                    var.values[k, ...],
                    wet,
                    dry,
                    otis[lonv].values,
                    otis[latv].values,
                    dmax,
                )

    return otis
