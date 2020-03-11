# -*- coding: utf-8 -*-

"""OTIS tools."""

import os
import numpy as np
from scipy import interpolate
import xarray as xr
from fsspec import filesystem, get_mapper
import pyroms
from .core import nodal, astrol
from .constituents import OMEGA


OTIS_VERSION = "v1"


class NCOtis(object):
    """ Object to replace CGrid_OTIS from pyroms and handle h, u and v at once

    Args: 
        model (str):             Filename of the regional OTIS model file in fspec format
                                 Ex: gs://oceanum-prod/tide/Model_ES2008
        drop_amp_params (bool):  Option to drop amplitude parameters (they are not used
                                 for ROMS tide file because complex params are more
                                 appropraite for remapping)

    Developer notes:
        - this should replace both .load_otis and .CGrid_OTIS from pyroms
        - migrate flood here as a method

    """

    name = "otis"  # necessary for remap to play nicely

    def __init__(
        self,
        model="file:///data/tide/otis_netcdf/Model_ES2008",
        drop_amp_params=True,
        x0=None,
        x1=None,
        y0=None,
        y1=None,
    ):
        # self.gridfile = get_mapper(model).root
        # elevfile = ".".join(
        #     [self.gridfile.split(".")[0].replace("grid", "h"), OTIS_VERSION, "nc"]
        # )
        # curfile = ".".join(
        #     [self.gridfile.split(".")[0].replace("grid", "u"), OTIS_VERSION, "nc"]
        # )

        with open(model) as f:
            lines = f.readlines()
            elevfile = os.path.join(os.path.dirname(model), lines[0].split('/')[-1]).strip()
            curfile = os.path.join(os.path.dirname(model), lines[1].split('/')[-1]).strip()
            self.gridfile = os.path.join(os.path.dirname(model), lines[2].split('/')[-1]).strip()

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
        else:
            self.was_subsetted = False

        self._fix_topo()
        self._fix_east()
        self._mask_vars()
        self._transp2vel()

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
                                file grid on your file system
                                files must be in OTIS netcdf grid format
                                TODO: make it a kwarg and discover best resolution if not provided
                                TODO: convert to zarr and use fsspec to generalize this
	lon, lat (numpy ndarray): Each is an n-length array of longitude 
                                and latitudes in degrees to perform predictions at.
                                Lat ranges from -90 to 90. Lon can range from -180 to 360.
  	time:                     m-length array of times.  Acceptable formats are 
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
    otis = NCOtis(modfile, x0=lon.min(), x1=lon.max(), y0=lat.min(), y1=lat.max())
    # print("Flooding land to avoid interpolation noise")
    # otis.flood()
    conlist = conlist or otis.cons
    omega = [OMEGA[c] for c in conlist]

    # Nodal correction: nodal needs days since 1992:
    days = (time[0] - np.datetime64('1992-01-01', 'D')).days
    pu, pf, v0u = nodal(days + 48622.0, conlist) # not sure where 48622.0 comes from ???

    # interpolating to requested grid
    print("Interpolating variables to requested grid")
    hRe, hIm, uRe, uIm, vRe, vIm = _regrid(otis, lon, lat)
    hRe, hIm, uRe, uIm, vRe, vIm = _remask(hRe, hIm, uRe, uIm, vRe, vIm, otis, lon, lat)

    # Calculate the time series
    tsec = np.array((time - np.datetime64('1992-01-01', 's')).total_seconds())
    nt = time.size 
    nc = len(conlist)
    nj, ni = lon.shape

    hRe = hRe.reshape((nc, nj * ni))
    hIm = hIm.reshape((nc, nj * ni))
    uRe = uRe.reshape((nc, nj * ni))
    uIm = uIm.reshape((nc, nj * ni))
    vRe = vRe.reshape((nc, nj * ni))
    vIm = vIm.reshape((nc, nj * ni))

    h, u, v = np.zeros((nt, nj * ni)), np.zeros((nt, nj * ni)), np.zeros((nt, nj * ni))

    for k, om in enumerate(omega):
        for idx in range(nj * ni):
            h[:, idx] += pf[k] * hRe[k, idx] * np.cos(om * tsec + v0u[k] + pu[k]) - pf[k] * hIm[k, idx] * np.sin(om * tsec + v0u[k] + pu[k])
            u[:, idx] += pf[k] * uRe[k, idx] * np.cos(om * tsec + v0u[k] + pu[k]) - pf[k] * uIm[k, idx] * np.sin(om * tsec + v0u[k] + pu[k])
            v[:, idx] += pf[k] * vRe[k, idx] * np.cos(om * tsec + v0u[k] + pu[k]) - pf[k] * vIm[k, idx] * np.sin(om * tsec + v0u[k] + pu[k])

    # TODO: write netcdf file and refactor using dicts to respect DRY
    h, u, v = h.reshape((nt, nj, ni)), u.reshape((nt, nj, ni)), v.reshape((nt, nj, ni))

    # TODO: write variable attributes
    ha = xr.DataArray(dims=('time', 'lat', 'lon'), coords={'time': time, 'lat': lat[:,0], 'lon': lon[0,...]}, name='et', data=h)
    ua = xr.DataArray(dims=('time', 'lat', 'lon'), coords={'time': time, 'lat': lat[:,0], 'lon': lon[0,...]}, name='ut', data=u)
    va = xr.DataArray(dims=('time', 'lat', 'lon'), coords={'time': time, 'lat': lat[:,0], 'lon': lon[0,...]}, name='vt', data=v)
    ds = xr.Dataset({'et': ha, 'ut': ua, 'vt': va})


    return ds


def _remask(hRe, hIm, uRe, uIm, vRe, vIm, otis, lon, lat):
    depth = _interp(otis.ds.hz, otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())
    depth = depth.reshape(lon.shape)[None, :].repeat(len(otis.cons), axis=0)
    hRe = np.ma.masked_where(depth < 1, hRe)
    hIm = np.ma.masked_where(depth < 1, hIm)
    uRe = np.ma.masked_where(depth < 1, uRe)
    uIm = np.ma.masked_where(depth < 1, uIm)
    vRe = np.ma.masked_where(depth < 1, vRe)
    vIm = np.ma.masked_where(depth < 1, vIm)
    return hRe, hIm, uRe, uIm, vRe, vIm


def _regrid(otis, lon, lat):
    nj, ni = lon.shape
    nc = len(otis.cons)
    hRe = np.zeros((nc, nj * ni))
    hIm = np.zeros((nc, nj * ni))
    uRe = np.zeros((nc, nj * ni))
    uIm = np.zeros((nc, nj * ni))
    vRe = np.zeros((nc, nj * ni))
    vIm = np.zeros((nc, nj * ni))

    for idx in range(nc):
        hRe[idx, :] = _interp(otis.ds.hRe[idx,...], otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())
        hIm[idx, :] = _interp(otis.ds.hIm[idx,...], otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())
        uRe[idx, :] = _interp(otis.ds.uRe[idx,...], otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())
        uIm[idx, :] = _interp(otis.ds.uIm[idx,...], otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())
        vRe[idx, :] = _interp(otis.ds.vRe[idx,...], otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())
        vIm[idx, :] = _interp(otis.ds.vIm[idx,...], otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())

    s = (nc,) + lon.shape
    
    return hRe.reshape(s), hIm.reshape(s), uRe.reshape(s), uIm.reshape(s), vRe.reshape(s), vIm.reshape(s)
 

def _interp(arr, x, y, x2, y2):
    arr, x, y = arr.values, x.values, y.values
    arr[np.isnan(arr) == 1] = 0
    spl = interpolate.RectBivariateSpline(x[0, :], y[:, 0], arr.T)
    return spl(x2, y2, grid=False)


if __name__ == '__main__':
    import pandas as pd
    import datetime as dt

    # English Channel
    xi = np.linspace(-1, 4, 50) 
    yi = np.linspace(48.5, 53.56, 53)

    # Hauraki Gulf
    xi = np.linspace(173.8282, 176.6906, 50) 
    yi = np.linspace(-38.3221, -35.1955, 53)

    lon, lat = np.meshgrid(xi, yi) 
    time = pd.date_range(dt.datetime(2001, 1, 1), dt.datetime(2001, 1, 7, 23), freq="H")
    modfile = '/data/tide/otis_netcdf/Model_tpxo7'
    h, u, v = predict_tide_grid(lon, lat, time, modfile)