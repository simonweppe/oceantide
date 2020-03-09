# -*- coding: utf-8 -*-

"""OTIS tools."""

import numpy as np
import xarray as xr
from fsspec import filesystem, get_mapper
import pyroms


OTIS_VERSION = "v1"


class NCOtis(object):
    """ Object to replace CGrid_OTIS from pyroms and handle h, u and v at once

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
        otis_grid="file:///data/tide/otis_netcdf/grid_tpxo9.nc",
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
        self._transp2vel()
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
        # this is a convenience for the remap fortran routines using scrip
        # needs to be done before topo null is no longer = 0
        self.hmask = self.ds.hz.values != 0
        self.umask = self.ds.hu.values != 0
        self.vmask = self.ds.hv.values != 0

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

    def _transp2vel(self):
        """ Compute complex velocities based on complex transports and append 
            them to the xr.Dataset
        
        """
        longname = "Tidal WE transport complex ampl., {c} part, at {n}-nodes"
        variables = dict(uRe=None, uIm=None, vRe=None, vIm=None)

        for node in ["u", "v"]:
            for com in ["Re", "Im"]:
                variables["{}{}".format(node, com)] = xr.Variable(
                    self.ds["{}{}".format(node.upper(), com)].dims,
                    self.ds["{}{}".format(node.upper(), com)].values
                    / self.ds["h{}".format(node)].values,
                    attrs=dict(long_name=longname.format(c=com, n=node.upper()), units="meter/s"),
                )

        self.ds = self.ds.assign(variables=variables)

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
        self.lon_t_vert = 0.5 * (
            self.ds.lon_z.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.ds.lon_z.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_t_vert = 0.5 * (
            self.ds.lat_z.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.ds.lat_z.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lon_u_vert = 0.5 * (
            self.ds.lon_u.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.ds.lon_u.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_u_vert = 0.5 * (
            self.ds.lat_u.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.ds.lat_u.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lon_v_vert = 0.5 * (
            self.ds.lon_v.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.ds.lon_v.values[y0 : y1 + 2, x0 : x1 + 2]
        )
        self.lat_v_vert = 0.5 * (
            self.ds.lat_v.values[y0 - 1 : y1 + 1, x0 - 1 : x1 + 1]
            + self.ds.lat_v.values[y0 : y1 + 2, x0 : x1 + 2]
        )

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


def predict_tide_grid(lon, lat, time, modfile, conlist=None):
    """ Performs a tidal prediction at all points in [lon,lat] at times.

	Args:
	
	modfile (str):            (Relative) path of the constituents model 
                                file on your file system
                                files must be in OTIS netcdf format
                                TODO: make it a kwarg and discover best resolution if not provided
	lon, lat (numpy ndarray): Each is an n-length array of longitude 
                                and latitudes in degrees to perform predictions at.
                                Lat ranges from -90 to 90. Lon can range from -180 to 360.
  	time:                       m-length array of times.  Acceptable formats are 
                                   a list of `datetime` objects, a list or array of 
                                   `numpy.datetime64` objects, or pandas date_range
	conlist :                 List of strings (optional). If supplied, gives a list 
                                of tidal constituents to include in prediction. 
                                Available are 'M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1'

	Returns
	-------
	h : m-by-n numpy array of tidal heights
		height is in meters, times are along the rows, and positions along
		the columns
	u : m-by-n numpy array of east-west tidal velocity [m/s]
	v : m-by-n numpy array of north tidal velocity [m/s]

	Examples
	--------

	dates = np.arange(np.datetime64('2001-04-03'),
	                  np.datetime64('2001-05-03'), dtype='datetime64[h]' )

	lon = np.array([198, 199])
	lat = np.array([21, 19])

	h, u, v = predict_tide_grid(lon, lat, time, '/data/tide/otis_netcdf/Model_ES2008') TODO: make it a netcdf file

	"""