# -*- coding: utf-8 -*-

"""Tidal prediction tools."""

import os
import logging
import numpy as np
import numpy.ma as ma
from scipy import interpolate
import netCDF4
import pandas as pd
import xarray as xr

from fsspec import filesystem, get_mapper

from ontake.ontake import Ontake

from .settings import *
from .core import nodal, astrol
from .constituents import OMEGA


os.environ.update(
    {"GOOGLE_APPLICATION_CREDENTIALS": "/source/ontide/secrets/ontide.json"}
)


class NCcons(object):
    """ Object to represent tidal constituents file. 
           OTIS is the format we adopt internally 
           for the sake of standardisation.

    Args: 
        model (str):             Intake dataset of the regional consituents. 
                                 TIP: use ontake to discover datasets:
                                      ot = Ontake(namespace='tide', 
                                                  master_url='gs://oceanum-catalog/oceanum.yml')
                                      ot.datasets

        drop_amp_params (bool):  Option to drop amplitude parameters (they are not used
                                 for ROMS tide file because complex params are more
                                 appropraite for remapping)
        x0, x1, y0, y1 (float):  Bounds for subsetting, default = None, which is no subsetting
        catalog (str):           Intake catalog that has the source constituents dataset
        namespace (str):         Intake namespace
    """

    name = "otis"  # necessary for remap to play nicely

    def __init__(
        self,
        model="tpxo9_tide_glob_cons",
        drop_amp_params=True,
        x0=None,
        x1=None,
        y0=None,
        y1=None,
        catalog=ONTAKE_CATALOG,
        namespace=ONTAKE_NAMESPACE,
    ):
        self.model = model
        print(f"Loading {model} from intake catalog")
        ot = Ontake(namespace=namespace, master_url=catalog)
        self.ds = ot.dataset(model)

        # if subset is requested, better to run it before any operation
        if np.array([x0, x1, y0, y1]).any() is not None:
            assert (
                np.array([x0, x1, y0, y1]).all() is not None
            ), "If one of <x0,x1,y0,y1> is provided, all must be provided"
            print("Subsetting")
            self.subset(x0=x0, x1=x1, y0=y0, y1=y1)
        else:
            self.was_subsetted = False

        self._fix_topo()
        self._mask_vars()

    def __repr__(self):
        _repr = "<Tidecons {} nc={} x0={:0.2f} x1={:0.2f} y0={:0.2f} y1={:0.2f} subset={}>".format(
            self.model,
            self.ds.dims["con"],
            self.ds.lon_z.values.min(),
            self.ds.lon_z.values.max(),
            self.ds.lat_z.values.min(),
            self.ds.lat_z.values.max(),
            self.was_subsetted,
        )
        _repr += f"\n{self.cons}"
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

        TODO: Implement constituents subset 
        """
        self.was_subsetted = True
        self.ds = self.ds.sel(
            lon_z=slice(x0, x1),
            lat_z=slice(y0, y1),
            lon_u=slice(x0, x1),
            lat_u=slice(y0, y1),
            lon_v=slice(x0, x1),
            lat_v=slice(y0, y1),
        )
        print(self.__repr__())


def predict_tide_point(
    lon,
    lat,
    time,
    model="tpxo9_tide_glob_cons",
    catalog=ONTAKE_CATALOG,
    namespace=ONTAKE_NAMESPACE,
    conlist=None,
    outfile=None,
):
    """ Performs a tidal prediction at a [lon,lat] grid for a [time] array.

	Args:
	
	lon, lat (float):  Lat ranges from -90 to 90. Lon can range from -180 to 180.
  	time: Array of datetime objects or equivalents such pandas.data_range, etc.
    model (str): Intake dataset of the regional constituents. TIP: use ontake to discover datasets:
                     ot = Ontake(namespace='tide', master_url='gs://oceanum-catalog/oceanum.yml')
                     ot.datasets
    catalog (str): Intake catalog that has the source constituents dataset
    namespace (str): Intake namespace
	conlist : List of strings (optional). If supplied, gives a list 
                  of tidal constituents to include in prediction. If not supplied, 
                  default from model source will be used.
                  Available are 'M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1'
    outfile: Writes pandas.DataFrame to disk as a NetCDF file

	Returns
	pandas.DataFrame containing:
	    et : tidal heights is in [m]
	    ut : eastward tidal velocity [m/s]
	    vt : northward tidal velocity [m/s]

	Examples
	--------

	time = pd.date_range('2001-1-1', '2001-1-7 23:00', freq="H")
	lon = 170
	lat = -30
	df = predict_tide_point(170, -30, time)
	"""
    cons = NCcons(
        model,
        x0=lon - 1,
        x1=lon + 1,
        y0=lat - 1,
        y1=lat + 1,
        catalog=catalog,
        namespace=namespace,
    )
    conlist = conlist or cons.cons
    omega = [OMEGA[c] for c in conlist]

    # Nodal correction: nodal needs days since 1992:
    days = (time[0] - np.datetime64("1992-01-01", "D")).days
    pu, pf, v0u = nodal(
        days + 48622.0, conlist
    )  # not sure where 48622.0 comes from ???

    # extracting nearest point
    df = cons.ds.sel(
        lon_z=lon,
        lat_z=lat,
        lon_u=lon,
        lat_u=lat,
        lon_v=lon,
        lat_v=lat,
        method="nearest",
        drop=True,
    ).to_dataframe()

    # Calculate the time series
    tsec = np.array((time - np.datetime64("1992-01-01", "s")).total_seconds())
    nt = time.size
    nc = len(conlist)

    rvars = dict()
    for var in ["h", "u", "v"]:
        rvars[var] = np.zeros((nt))

    print("Converting complex harmonics into timeseries")
    for k, om in enumerate(omega):
        for p in ["h", "u", "v"]:
            rvars[p] += pf[k] * df[f"{p}Re"][k] * np.cos(
                om * tsec + v0u[k] + pu[k]
            ) - pf[k] * df[f"{p}Im"][k] * np.sin(om * tsec + v0u[k] + pu[k])

    df = pd.DataFrame(
        {"et": rvars["h"], "ut": rvars["u"], "vt": rvars["v"]}, index=time
    )
    df.index.rename("time", inplace=True)

    if df.dropna().empty:
        raise Exception(
            "No valid data found, please check if coordinates are in landmask"
        )

    if outfile:
        print(f"Output written to {outfile}")
        ds = xr.Dataset.from_dataframe(df)
        ds.attrs = {"longitude": lon, "latitude": lat}
        ds.to_netcdf(outfile)
    else:
        print(f"Output written to memory, use <output> kwarg if a file output is desired")

    return df


def predict_tide_grid(
    lon,
    lat,
    time,
    model="tpxo9_tide_glob_cons",
    catalog=ONTAKE_CATALOG,
    namespace=ONTAKE_NAMESPACE,
    conlist=None,
    outfile=None,
):
    """ Performs a tidal prediction at all points in [lon,lat] at times.

	Args:
	
	lon, lat (numpy ndarray): Each is an n-length array of longitude 
                                and latitudes in degrees to perform predictions at.
                                Lat ranges from -90 to 90. Lon can range from -180 to 180.
  	time: Array of datetime objects or equivalents such pandas.data_range, etc.
    model (str): Intake dataset of the regional constituents. TIP: use ontake to discover datasets:
                    ot = Ontake(namespace='tide', master_url='gs://oceanum-catalog/oceanum.yml')
                    ot.datasets
    catalog (str): Intake catalog that has the source constituents dataset
    namespace (str): Intake namespace
	conlist : List of strings (optional). If supplied, gives a list 
                  of tidal constituents to include in prediction. If not supplied, 
                  default from model source will be used.
                  Available are 'M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1'
    outfile: Writes xarray.Dataset to disk as a NetCDF file

	Returns
	xarray.Dataset containing:
	    et : 3D numpy array of tidal heights [m]
	    ut : 3D numpy array of eastward tidal velocity [m/s]
	    vt : 3D numpy array of northward tidal velocity [m/s]

	Examples
	--------

	time = pd.date_range('2001-1-1', '2001-1-7 23:00', freq="H")
	lon = np.array([168, 179])
	lat = np.array([11, 19])
	ds = predict_tide_grid(lon, lat, dates) 
	"""

    cons = NCcons(
        model,
        x0=lon.min(),
        x1=lon.max(),
        y0=lat.min(),
        y1=lat.max(),
        catalog=catalog,
        namespace=namespace,
    )
    print("Flooding land to avoid interpolation noise")
    # cons.flood()
    conlist = conlist or cons.cons
    omega = [OMEGA[c] for c in conlist]

    # Nodal correction: nodal needs days since 1992:
    days = (time[0] - np.datetime64("1992-01-01", "D")).days
    pu, pf, v0u = nodal(
        days + 48622.0, conlist
    )  # not sure where 48622.0 comes from ???

    # interpolating to requested grid
    print("Interpolating variables to requested grid")
    cvars = _regrid(cons, lon, lat)
    cvars = _remask(cvars, cons, lon, lat)

    # Calculate the time series
    tsec = np.array((time - np.datetime64("1992-01-01", "s")).total_seconds())
    nt = time.size
    nc = len(conlist)
    nj, ni = lon.shape

    for varname, var in cvars.items():
        cvars[varname] = var.reshape((nc, nj * ni))

    rvars = dict()
    for var in ["h", "u", "v"]:
        rvars[var] = np.zeros((nt, nj * ni))

    print("Converting complex harmonics into timeseries")
    # TODO: this is very inneficient for medium to large grids, need to optimize somehow
    for k, om in enumerate(omega):
        print(f"    constituent {k+1} | {len(omega)}")
        for idx in range(nj * ni):
            for p in ["h", "u", "v"]:
                rvars[p][:, idx] += pf[k] * cvars[f"{p}Re"][k, idx] * np.cos(
                    om * tsec + v0u[k] + pu[k]
                ) - pf[k] * cvars[f"{p}Im"][k, idx] * np.sin(om * tsec + v0u[k] + pu[k])

    for varname, var in rvars.items():
        rvars[varname] = var.reshape((nt, nj, ni))

    rvars = _remask(rvars, cons, lon, lat)
    ds = make_timeseries_dataset(time, lon, lat, rvars["h"], rvars["u"], rvars["v"])

    if outfile:
        ds.to_netcdf(outfile)

    return ds


def _remask(_vars, cons, lon, lat):
    for val in _vars.values():
        a, b, c = val.shape
        break

    depth = _interp(cons.ds.hz, cons.ds.lon_z, cons.ds.lat_z, lon.ravel(), lat.ravel())
    depth = depth.reshape(lon.shape)[None, :].repeat(a, axis=0)
    newlist = []
    for varname, var in _vars.items():
        _vars[varname] = np.ma.masked_where(depth < 1, var)

    return _vars


def _regrid(cons, lon, lat):
    nj, ni = lon.shape
    nc = len(cons.cons)
    cvars = dict()
    for var in COMPLEX_VARS:
        cvars[var] = np.zeros((nc, nj * ni))

    for idx in range(nc):
        for varname, var in cvars.items():
            p = (
                "z" if "h" in varname else varname[0]
            )  # because the OTIS format doesn't follow conventions :/
            var[idx, :] = _interp(
                cons.ds.data_vars[varname][idx, ...],
                cons.ds.coords[f"lon_{p}"],
                cons.ds.coords[f"lat_{p}"],
                lon.ravel(),
                lat.ravel(),
            )

    s = (nc,) + lon.shape
    for varname in cvars.keys():
        cvars[varname] = cvars[varname].reshape(s)

    return cvars


def _interp(arr, x, y, x2, y2):
    # from IPython import embed; embed()
    arr, x, y = arr.values, x.values, y.values
    arr[np.isnan(arr) == 1] = 0
    spl = interpolate.RectBivariateSpline(x, y, arr.T)
    return spl(x2, y2, grid=False)


def make_timeseries_dataset(
    time,
    lon,
    lat,
    et,
    ut,
    vt,
    attrs={
        "description": "Tide elevation and currents prediction time series",
        "institution": "Oceanum Ltd",
    },
):
    """ Create xarray.Dataset with tidal prediction timeseries.

        It can be used to create netcdf or zarr files

    Args:
        time (list of datetime objects 1D): Time coordinate
        lon (numpy.ndarray 2D): Lon coordinates
        lat (numpy.ndarray 2D): Lat coordinates 
        et (numpy.ma.core.MaskedArray 3D): Tidal elevations
        ut (numpy.ma.core.MaskedArray 3D): Tidal U-current
        vt (numpy.ma.core.MaskedArray 3D): Tidal V-current
        attrs (dict): Dataset global attributes dictionary (optional)  

    Returns:
        ds (xarray.Dataset)

    TODO: perhaps use CDL to create a skeleton Dataset and fill it up?
    """
    ha = xr.DataArray(
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat[:, 0], "lon": lon[0, :]},
        name="et",
        data=et,
        attrs={
            "standard_name": "tidal_sea_surface_height_above_mean_sea_level",
            "units": "m",
            "_FillValue": et.fill_value,
        },
    )
    ua = xr.DataArray(
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat[:, 0], "lon": lon[0, :]},
        name="ut",
        data=ut,
        attrs={
            "standard_name": "eastward_sea_water_velocity_due_to_tides",
            "units": "m s^-1",
            "_FillValue": ut.fill_value,
        },
    )
    va = xr.DataArray(
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat[:, 0], "lon": lon[0, :]},
        name="vt",
        data=vt,
        attrs={
            "standard_name": "northward_sea_water_velocity_due_to_tides",
            "units": "m s^-1",
            "_FillValue": vt.fill_value,
        },
    )

    ds = xr.Dataset({"et": ha, "ut": ua, "vt": va}, attrs=attrs)

    return ds
