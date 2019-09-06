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
        otis_grid (str):         Filename of the regional OTIS grid in fspec format
                                 Ex: gs://oceanum-static/otis/grid_tpxo9.nc
        drop_amp_params (bool):  Option to drop amplitude parameters (they are not used
                                 for ROMS tide file because complex params are more
                                 appropraite for remapping)

    Developer notes:
        - this should replace both .load_otis and .CGrid_OTIS
        - migrate flood here as a method

    """

    name = "otis"  # necessary for remap to play nicely

    def __init__(
        self,
        otis_grid="file:///data/otis/grid_tpxo9.nc",
        drop_amp_params=True,
        x0=None,
        x1=None,
        y0=None,
        y1=None,
    ):
        self.gridfile = get_mapper(otis_grid).root
        elevfile = ".".join(
            [self.gridfile.split(".")[0].replace("grid", "h"), OTIS_VERSION, "nc"]
        )
        curfile = ".".join(
            [self.gridfile.split(".")[0].replace("grid", "u"), OTIS_VERSION, "nc"]
        )

        dsg = xr.open_dataset(self.gridfile)
        dsh = xr.open_dataset(elevfile)
        dsu = xr.open_dataset(curfile)

        # drop unused vars, transpose and merge
        dsg = dsg.drop(
            ["x_z", "y_z", "x_u", "y_u", "x_v", "y_v", "iob_z", "iob_u", "iob_v"]
        ).transpose("ny", "nx")
        dsu = dsu.drop(["lon_u", "lat_u", "lon_v", "lat_v"]).transpose("nc", "ny", "nx")
        dsh = dsh.drop(["lat_z", "lon_z"]).transpose("nc", "ny", "nx")
        if drop_amp_params:
            dsh = dsh.drop(["ha", "hp"])
            dsu = dsu.drop(["Ua", "ua", "up", "Va", "va", "vp"])

        self.ds = xr.merge([dsg, dsh, dsu])

        # if subset is requested, better to run it before any operation
        if np.array([x0, x1, y0, y1]).any() is not None:
            assert (
                np.array([x0, x1, y0, y1]).all() is not None
            ), "If one of <x0,x1,y0,y1> is provided, all must be provided"
            self.subset(x0=x0, x1=x1, y0=y0, y1=y1)

        self._fix_topo()
        self._fix_east()
        self._mask_vars()
        self.was_subsetted = False

    def __repr__(self):
        _repr = "<OTIS {} nc={} x0={:0.2f} x1={:0.2f} y0={:0.2f} y1={:0.2f} subset={}>".format(
            self.gridfile,
            self.ds.dims["nc"],
            self.ds.lon_z.values.min(),
            self.ds.lon_z.values.max(),
            self.ds.lat_z.values.min(),
            self.ds.lat_z.values.max(),
            self.was_subsetted,
        )
        _repr += "\n{}".format(self.cons)
        return _repr

    def _fix_topo(self):
        """ Make topography values above zero as they are inconveniently = 0

        """
        self.ds.hz.values += 0.001
        self.ds.hu.values += 0.001
        self.ds.hv.values += 0.001

    def _fix_east(self):
        """ Convert 0 < lon < 360 to -180 < lon < 180 and shift all vars accordingly

        """
        lon = self.ds.lon_z.values
        lon[lon > 180] -= 360
        idx = np.argsort(lon)
        lon = np.take_along_axis(lon, idx, axis=-1)

        print("shifting along x-axis: ")
        for varname, var in self.ds.data_vars.items():
            if "ny" in var.dims and "nx" in var.dims and not varname.startswith("lat"):
                print(varname)
                vals = var.values
                if "lon" in varname:
                    vals[vals > 180] -= 360
                    self.ds[varname].values = vals

                if len(var.dims) > 2:
                    vals = np.take_along_axis(
                        vals, idx[None, ...].repeat(self.ds.dims["nc"], axis=0), axis=-1
                    )
                else:
                    vals = np.take_along_axis(vals, idx, axis=-1)

                self.ds[varname].values = vals

    def _mask_vars(self):
        """ Apply mask to vars as land values are inconveniently = 0

        """
        for varname, var in self.ds.data_vars.items():
            if len(var.dims) > 2:  # leaving cons and coords out
                var.values = np.ma.masked_where(var.values == 0, var.values)

    @property
    def cons(self):
        """ Nicely formatted cons attribute

        """
        cons = self.ds.con.values.tostring().decode().split()
        cons = [c.upper() for c in cons]
        return cons

    def subset(self, x0, x1, y0, y1):
        """ Subset the grid

        Args: 
            x0:    Minimum longitude                
            x1:    Maximum longitude
            y0:    Minimum latitude
            y1:    Maximum latitude

        Developer notes:
            - Implement constituents subset as well
            - Done this in a rush, should probably abstract it better
            - Idea here is that vertices and angle need derivatives, which 
              cannot be computed in the global or regional grid, otherwise
              we'd end up having to cut a couple of lines/cols at the boundaries
            - As we are subsetting, we end up with spare lines/cols, so it allows
              it to be done in a clean way. 
            - Before anything, it needs to be checked if these vertices and angles
              are being used for anything we need.  
        """
        self.was_subsetted = True
        f = np.where(
            (self.ds.lon_z.values >= x0)
            & (self.ds.lon_z.values <= x1)
            & (self.ds.lat_z.values >= y0)
            & (self.ds.lat_z.values <= y1)
        )
        x0, x1, y0, y1 = f[1].min(), f[1].max(), f[0].min(), f[0].max()

        # compute vertices and angle before we lose the indexes in isel
        # self.lon_t_vert = 0.5 * (
        #     self.ds.lon_z.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
        #     + self.ds.lon_z.values[y0 : y1 + 2, x0 : x1 + 2]
        # )
        # self.lat_t_vert = 0.5 * (
        #     self.ds.lat_z.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
        #     + self.ds.lat_z.values[y0 : y1 + 2, x0 : x1 + 2]
        # )
        # self.lon_u_vert = 0.5 * (
        #     self.ds.lon_u.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
        #     + self.ds.lon_u.values[y0 : y1 + 2, x0 : x1 + 2]
        # )
        # self.lat_u_vert = 0.5 * (
        #     self.ds.lat_u.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
        #     + self.ds.lat_u.values[y0 : y1 + 2, x0 : x1 + 2]
        # )
        # self.lon_v_vert = 0.5 * (
        #     self.ds.lon_v.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
        #     + self.ds.lon_v.values[y0 : y1 + 2, x0 : x1 + 2]
        # )
        # self.lat_v_vert = 0.5 * (
        #     self.ds.lat_v.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
        #     + self.ds.lat_v.values[y0 : y1 + 2, x0 : x1 + 2]
        # )

        # ones = np.ones(self.ds.hz.shape)
        # a1 = self.ds.lat_u[y0 : y1 + 1, x0 + 1 : x1 + 2] - self.ds.lat_u[y0 : y1 + 1, x0 : x1 + 1]
        # a2 = self.ds.lon_u[y0 : y1 + 1, x0 + 1 : x1 + 2] - self.ds.lon_u[y0 : y1 + 1, x0 : x1 + 1]
        # a3 = 0.5 * (
        #     self.ds.lat_u[y0 : y1 + 1, x0 + 1 : x1 + 2] + self.ds.lat_u[y0 : y1 + 1, x0 : x1 + 1]
        # )
        # a2 = np.where(a2 > 180 * ones, a2 - 360 * ones, a2)
        # a2 = np.where(a2 < -180 * ones, a2 + 360 * ones, a2)
        # a2 = a2 * np.cos(np.pi / 180.0 * a3)
        # self.angle = np.arctan2(a1, a2)

        # finally, subsetting
        self.ds = self.ds.isel(nx=slice(x0, x1), ny=slice(y0, y1))
        print(self.__repr__())

    def flood(self, dmax=1):
        """
        Flood variables into land to avoid spurious values close to the coast

        Args:
            dmax (int>0):            Maximum horizontal flooding distance

        """
        for varname, var in self.ds.data_vars.items():
            if len(var.dims) < 3:
                continue
            else:
                var3d = varname
                break

        msk = np.isnan(self.ds[var3d].values[0, ...])

        for k in range(self.ds.dims["nc"]):
            for varname, var in self.ds.data_vars.items():
                if len(var.dims) > 2:  # leaving cons and coords out
                    for varname2, var2 in self.ds.data_vars.items():
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
                        self.ds[lonv].values,
                        self.ds[latv].values,
                        dmax,
                    )



# class CGrid_OTIS(object):

#     # CGrid object for OTIS

#     def __init__(
#         self,
#         cons,
#         lon_t,
#         lat_t,
#         lon_u,
#         lat_u,
#         lon_v,
#         lat_v,
#         mask_t,
#         mask_u,
#         mask_v,
#         z_t,
#         z_u,
#         z_v,
#         missing_value,
#         xlim,
#         ylim,
#     ):
#         self.cons = cons
#         self.name = "otis"

#         self.null = -9999.0

#         f = np.where(
#             (lon_t >= xlim[0])
#             & (lon_t <= xlim[1])
#             & (lat_t >= ylim[0])
#             & (lat_t <= ylim[1])
#         )
#         x0, x1, y0, y1 = f[1].min(), f[1].max(), f[0].min(), f[0].max()

#         self.z_t = z_t[y0 : y1 + 1, x0 : x1 + 1]

#         self.z_u = z_u[y0 : y1 + 1, x0 : x1 + 1]
#         self.z_v = z_v[y0 : y1 + 1, x0 : x1 + 1]

#         self.lon_t = lon_t[y0 : y1 + 1, x0 : x1 + 1]
#         self.lat_t = lat_t[y0 : y1 + 1, x0 : x1 + 1]

#         self.lon_u = lon_u[y0 : y1 + 1, x0 : x1 + 1]
#         self.lat_u = lat_u[y0 : y1 + 1, x0 : x1 + 1]
#         self.lon_v = lon_v[y0 : y1 + 1, x0 : x1 + 1]
#         self.lat_v = lat_v[y0 : y1 + 1, x0 : x1 + 1]

#         self.lon_t_vert = 0.5 * (
#             lon_t[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lon_t[y0 : y1 + 2, x0 : x1 + 2]
#         )
#         self.lat_t_vert = 0.5 * (
#             lat_t[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lat_t[y0 : y1 + 2, x0 : x1 + 2]
#         )

#         self.lon_u_vert = 0.5 * (
#             lon_u[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lon_u[y0 : y1 + 2, x0 : x1 + 2]
#         )
#         self.lat_u_vert = 0.5 * (
#             lat_u[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lat_u[y0 : y1 + 2, x0 : x1 + 2]
#         )
#         self.lon_v_vert = 0.5 * (
#             lon_v[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lon_v[y0 : y1 + 2, x0 : x1 + 2]
#         )
#         self.lat_v_vert = 0.5 * (
#             lat_v[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1] + lat_v[y0 : y1 + 2, x0 : x1 + 2]
#         )

#         self.mask_t = mask_t[y0 : y1 + 1, x0 : x1 + 1]
#         self.mask_u = mask_u[y0 : y1 + 1, x0 : x1 + 1]
#         self.mask_v = mask_v[y0 : y1 + 1, x0 : x1 + 1]

#         ones = np.ones(self.z_t.shape)
#         a1 = lat_u[y0 : y1 + 1, x0 + 1 : x1 + 2] - lat_u[y0 : y1 + 1, x0 : x1 + 1]
#         a2 = lon_u[y0 : y1 + 1, x0 + 1 : x1 + 2] - lon_u[y0 : y1 + 1, x0 : x1 + 1]
#         a3 = 0.5 * (
#             lat_u[y0 : y1 + 1, x0 + 1 : x1 + 2] + lat_u[y0 : y1 + 1, x0 : x1 + 1]
#         )
#         a2 = np.where(a2 > 180 * ones, a2 - 360 * ones, a2)
#         a2 = np.where(a2 < -180 * ones, a2 + 360 * ones, a2)
#         a2 = a2 * np.cos(np.pi / 180.0 * a3)
#         self.angle = np.arctan2(a1, a2)


# def load_otis(grdfile, version="v1", xlim=None, ylim=None, missing_value=-9999):
#     """
#     grd = load_otis(grdfile)

#     Load Cgrid object for OTIS from netCDF file
#     """
#     consfile = ".".join([grdfile.split(".")[0].replace("grid", "h"), version, "nc"])

#     ds = xr.open_dataset(grdfile)
#     dsh = xr.open_dataset(consfile)

#     lonh = ds.lon_z.values
#     lath = ds.lat_z.values
#     lonu = ds.lon_u.values
#     latu = ds.lat_u.values
#     lonv = ds.lon_v.values
#     latv = ds.lat_v.values

#     zh = ds.hz.values + 0.001  # avoid zero-division, otis land values are zero :/
#     zu = ds.hu.values + 0.001
#     zv = ds.hv.values + 0.001

#     # land mask
#     hmask = zh != 0
#     umask = zu != 0
#     vmask = zv != 0

#     # longitude from -180 to 180
#     lonh[lonh > 180] = lonh[lonh > 180] - 360
#     lonu[lonu > 180] = lonu[lonu > 180] - 360
#     lonv[lonv > 180] = lonv[lonv > 180] - 360

#     cons = dsh.con.values.tostring().decode().split()
#     cons = [c.upper() for c in cons]

#     grid = CGrid_OTIS(
#         cons,
#         lonh,
#         lath,
#         lonu,
#         latu,
#         lonv,
#         latv,
#         hmask,
#         umask,
#         vmask,
#         zh,
#         zu,
#         zv,
#         missing_value,
#         xlim,
#         ylim,
#     )

#     return grid



