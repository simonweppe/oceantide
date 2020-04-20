# -*- coding: utf-8 -*-

"""OTIS tools."""

import os
import numpy as np
from scipy import interpolate
import netCDF4
import xarray as xr
import pyroms
from fsspec import filesystem, get_mapper

from ontake.ontake import Ontake

from .core import nodal, astrol
from .constituents import OMEGA


OTIS_VERSION = "9"
COMPLEX_VARS = ["hRe", "hIm", "uRe", "uIm", "vRe", "vIm"]
ONTAKE_CATALOG = "gs://oceanum-catalog/oceanum.yml"
ONTAKE_NAMESPACE = "tide"


class NCOtis(object):
    """ Object to represent OTIS tidal constituents file

    Args: 
        model (str):             Intake dataset of the regional OTIS model. 
                                 TIP: use ontake to discover datasets:
                                      ot = Ontake(namespace='tide', 
                                                  master_url='gs://oceanum-catalog/oceanum.yml')
                                      ot.datasets

        drop_amp_params (bool):  Option to drop amplitude parameters (they are not used
                                 for ROMS tide file because complex params are more
                                 appropraite for remapping)
    """

    name = "otis"  # necessary for remap to play nicely

    def __init__(
        self, model="tpxo9", drop_amp_params=True, x0=None, x1=None, y0=None, y1=None,
    ):
        self.model = model
        print(f"Loading {model} from intake catalog")
        ot = Ontake(namespace=ONTAKE_NAMESPACE, master_url=ONTAKE_CATALOG)
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
        self._transp2vel()

    def __repr__(self):
        _repr = "<OTIS {} nc={} x0={:0.2f} x1={:0.2f} y0={:0.2f} y1={:0.2f} subset={}>".format(
            self.model,
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
                    attrs=dict(
                        long_name=longname.format(c=com, n=node.upper()),
                        units="meter/s",
                    ),
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


def predict_tide_grid(
    lon, lat, time, model="tpxo9", conlist=None, outfile=None,
):
    """ Performs a tidal prediction at all points in [lon,lat] at times.

	Args:
	
	model (str):                Intake dataset of the regional OTIS model. 
                                TIP: use ontake to discover datasets:
                                    ot = Ontake(namespace='tide', 
                                                master_url='gs://oceanum-catalog/oceanum.yml')
                                    ot.datasets
                                TODO: convert to zarr and use fsspec to generalize this
	lon, lat (numpy ndarray): Each is an n-length array of longitude 
                                and latitudes in degrees to perform predictions at.
                                Lat ranges from -90 to 90. Lon can range from -180 to 360.
  	time:                     m-length array of times.  Acceptable formats are 
                                a list of `datetime` objects, a list or array of 
                                `np.datetime64` objects, or pandas date_range
	conlist :                 List of strings (optional). If supplied, gives a list 
                                of tidal constituents to include in prediction. 
                                Available are 'M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1'
    outfile:                  Writes xarray.Dataset to disk as a NetCDF file

	Returns
	xarray.Dataset containing:
	    et : m-by-n numpy array of tidal heights
		     height is in meters
	    ut : m-by-n numpy array of eastward tidal velocity [m/s]
	    vt : m-by-n numpy array of northward tidal velocity [m/s]

	Examples
	--------

	dates = np.arange(np.datetime64('2001-04-03'),
	                  np.datetime64('2001-05-03'), dtype='datetime64[h]' )

	lon = np.array([198, 199])
	lat = np.array([21, 19])

	ds = predict_tide_grid(lon, lat, time) 
	"""

    otis = NCOtis(model, x0=lon.min(), x1=lon.max(), y0=lat.min(), y1=lat.max())
    print("Flooding land to avoid interpolation noise")
    otis.flood()
    conlist = conlist or otis.cons
    omega = [OMEGA[c] for c in conlist]

    # Nodal correction: nodal needs days since 1992:
    days = (time[0] - np.datetime64("1992-01-01", "D")).days
    pu, pf, v0u = nodal(
        days + 48622.0, conlist
    )  # not sure where 48622.0 comes from ???

    # interpolating to requested grid
    print("Interpolating variables to requested grid")
    cvars = _regrid(otis, lon, lat)
    cvars = _remask(cvars, otis, lon, lat)

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

    for k, om in enumerate(omega):
        for idx in range(nj * ni):
            for p in ["h", "u", "v"]:
                rvars[p][:, idx] += pf[k] * cvars["{}Re".format(p)][k, idx] * np.cos(
                    om * tsec + v0u[k] + pu[k]
                ) - pf[k] * cvars["{}Im".format(p)][k, idx] * np.sin(
                    om * tsec + v0u[k] + pu[k]
                )

    for varname, var in rvars.items():
        rvars[varname] = var.reshape((nt, nj, ni))

    rvars = _remask(rvars, otis, lon, lat)
    fill_value = rvars["u"].fill_value

    for varname, var in rvars.items():
        rvars[varname] = var.filled(var.fill_value)

    ha = xr.DataArray(
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat[:, 0], "lon": lon[0, ...]},
        name="et",
        data=rvars["h"],
        attrs={
            "standard_name": "tidal_sea_surface_height_above_mean_sea_level",
            "units": "m",
            "_FillValue": fill_value,
        },
    )
    ua = xr.DataArray(
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat[:, 0], "lon": lon[0, ...]},
        name="ut",
        data=rvars["u"],
        attrs={
            "standard_name": "eastward_sea_water_velocity_due_to_tides",
            "units": "m s^-1",
            "_FillValue": fill_value,
        },
    )
    va = xr.DataArray(
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat[:, 0], "lon": lon[0, ...]},
        name="vt",
        data=rvars["v"],
        attrs={
            "standard_name": "northward_sea_water_velocity_due_to_tides",
            "units": "m s^-1",
            "_FillValue": fill_value,
        },
    )
    ds = xr.Dataset({"et": ha, "ut": ua, "vt": va})

    if outfile:
        ds.to_netcdf(outfile)

    return ds


def _remask(_vars, otis, lon, lat):
    for val in _vars.values():
        a, b, c = val.shape
        break

    depth = _interp(otis.ds.hz, otis.ds.lon_z, otis.ds.lat_z, lon.ravel(), lat.ravel())
    depth = depth.reshape(lon.shape)[None, :].repeat(a, axis=0)
    newlist = []
    for varname, var in _vars.items():
        _vars[varname] = np.ma.masked_where(depth < 1, var)

    return _vars


def _regrid(otis, lon, lat):
    nj, ni = lon.shape
    nc = len(otis.cons)
    cvars = dict()
    for var in COMPLEX_VARS:
        cvars[var] = np.zeros((nc, nj * ni))

    for idx in range(nc):
        for varname, var in cvars.items():
            p = (
                "z" if "h" in varname else varname[0]
            )  # because OTIS doesn't follow conventions :/
            var[idx, :] = _interp(
                otis.ds.data_vars[varname][idx, ...],
                otis.ds.data_vars["lon_{}".format(p)],
                otis.ds.data_vars["lat_{}".format(p)],
                lon.ravel(),
                lat.ravel(),
            )

    s = (nc,) + lon.shape
    for varname in cvars.keys():
        cvars[varname] = cvars[varname].reshape(s)

    return cvars


def _interp(arr, x, y, x2, y2):
    arr, x, y = arr.values, x.values, y.values
    arr[np.isnan(arr) == 1] = 0
    spl = interpolate.RectBivariateSpline(x[0, :], y[:, 0], arr.T)
    return spl(x2, y2, grid=False)


def _fix_east(ds):
    """ Convert 0 < lon < 360 to -180 < lon < 180 and shift all vars accordingly. 
            IMPORTANT: this is peculiar to OTIS netcdf file provided by their servers.
                       It is not meant to work generically with any xarray.Dataset
            Args:
                ds (xarray.Dataset) :: Input xarray dataset

            Returns:
                xarray.Dataset
        """
    lon = ds.lon_z.values
    lon[lon > 180] -= 360
    idx = np.argsort(lon)
    lon = np.take_along_axis(lon, idx, axis=-1)

    print("shifting along x-axis: ")
    for varname, var in ds.data_vars.items():
        if "ny" in var.dims and "nx" in var.dims and not varname.startswith("lat"):
            print(varname)
            vals = var.values
            if "lon" in varname:
                vals[vals > 180] -= 360
                ds[varname].values = vals

            if len(var.dims) > 2:
                vals = np.take_along_axis(
                    vals, idx[None, ...].repeat(ds.dims["nc"], axis=0), axis=-1
                )
            else:
                vals = np.take_along_axis(vals, idx, axis=-1)

            ds[varname].values = vals

    return ds


def otisnc2zarr(
    model="file:///data/tide/otis_netcdf/Model_tpxo9",
    outfile="gs://oceanum-tide/otis/DATA_zarr/tpxo9.zarr",
    drop_amp_params=False,
):
    # TODO: write retrieval from bucket
    """ Function to convert h, u and v OTIS cons netcdf files into zarr so it can be 
        inserted into intake catalog

    Args: 
        model (str):             Filename of the regional OTIS model file in fspec format
                                 Ex: gs://oceanum-prod/tide/Model_ES2008
        drop_amp_params (bool):  Option to drop amplitude parameters (they are not used
                                 for ROMS tide file because complex params are more
                                 appropriate for remapping)

    Developer notes:
        - this should replace both .load_otis and .CGrid_OTIS from pyroms
    """
    with open(model) as f:
        lines = f.readlines()
        elevfile = os.path.join(os.path.dirname(model), lines[0].split("/")[-1]).strip()
        curfile = os.path.join(os.path.dirname(model), lines[1].split("/")[-1]).strip()
        gfile = os.path.join(os.path.dirname(model), lines[2].split("/")[-1]).strip()

    dsg = xr.open_dataset(gfile)
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

    ds = xr.merge([dsg, dsh, dsu])
    ds = _fix_east(ds)
    ds.to_zarr("/tmp", os.path.basename(outfile), consolidated=True)
    # TODO write bucket copy


def read_otis_grd_bin(grdfile):
    """ Reads the grid data from an otis binary file
        
    Args:
        grdfile (str): OTIS grid binary file path
        
    Returns: 
        lon_z, lat_z, lon_u, lat_u, lon_v, lat_v  ::  Arakawa C-grid coordinates 
        hz  ::  depth at Z nodes
        mz  ::  mask at Z nodes

    TODO: return depth and mask at U and V nodes

    See this post on byte ordering
            http://stackoverflow.com/questions/1632673/python-file-slurp-w-endian-conversion
    """
    f = open(grdfile, "rb")
    #
    ## Try numpy
    f.seek(4, 0)
    n = np.fromfile(f, dtype=np.int32, count=1)
    m = np.fromfile(f, dtype=np.int32, count=1)
    lat_z = np.fromfile(f, dtype=np.float32, count=2)
    lon_z = np.fromfile(f, dtype=np.float32, count=2)
    dt = np.fromfile(f, dtype=np.float32, count=1)

    n.byteswap(True)
    m.byteswap(True)
    n = int(n)
    m = int(m)
    lat_z.byteswap(True)
    lon_z.byteswap(True)
    dt.byteswap(True)

    nob = np.fromfile(f, dtype=np.int32, count=1)
    nob.byteswap(True)
    if nob == 0:
        f.seek(20, 1)
        iob = []
    else:
        f.seek(8, 1)
        iob = np.fromfile(f, dtype=np.int32, count=int(2 * nob))
        iob.byteswap(True)
        iob = np.reshape(iob, (2, int(nob)))
        f.seek(8, 1)

    hz = np.fromfile(f, dtype=np.float32, count=int(n * m))
    f.seek(8, 1)
    mz = np.fromfile(f, dtype=np.int32, count=int(n * m))

    hz.byteswap(True)
    mz.byteswap(True)

    hz = np.reshape(hz, (m, n))
    mz = np.reshape(mz, (m, n))

    f.close()

    lon_z, lat_z = np.meshgrid(
        np.linspace(lon_z[0], lon_z[1], n), np.linspace(lat_z[0], lat_z[1], m)
    )

    if (lon_z[0] < 0) & (lon_z[1] < 0):
        lon_z = lon_z + 360

    # WARNING: assuming OTIS grids will always be regular, easier than deducting from the binaries
    d2 = (lon_z[1, 0] - lon_z[0, 0]) / 2.0
    lon_u, lat_u = lon_z - d2, lat_z.copy()
    lon_v, lat_v = lon_z.copy(), lat_z - d2

    return lon_z, lat_z, lon_u, lat_u, lon_v, lat_v, hz, mz


def read_otis_cons_bin(hfile):
    """
    Returns the list of constituents in the file
    """
    CHAR = np.dtype(">c")

    with open(hfile, "rb") as f:
        nm = np.fromfile(f, dtype=np.int32, count=4)
        nm.byteswap(True)

        ncons = nm[-1]
        dum = np.fromfile(f, dtype=np.int32, count=4)[0]
        cons = []
        for i in range(ncons):
            scons = np.fromfile(f, CHAR, 4).tostring().upper()
            cons.append(scons.rstrip())

        cons = np.array([c.ljust(4).lower() for c in cons])


def read_otis_h_bin(hfile):
    INT = np.dtype(">i4")
    FLOAT = np.dtype(">f4")
    CHAR = np.dtype(">c")

    with open(hfile, "rb") as f:
        ll = np.fromfile(f, INT, 1)[0]
        nlat = np.fromfile(f, INT, 1)[0]
        nlon = np.fromfile(f, INT, 1)[0]
        ncons = np.fromfile(f, INT, 1)[0]
        gridbound = np.fromfile(f, FLOAT, 4)
        cid = []
        for i in range(ncons):
            scons = np.fromfile(f, CHAR, 4).tostring().upper()
            cid.append(scons.rstrip())

        nn = nlon * nlat
        h = []
        for i in range(ncons):
            htemp = np.fromfile(f, FLOAT, 2 * nn)
            h.append(np.reshape(htemp[::2] + 1j * htemp[1::2], (nlat, nlon)))

    h = np.array(h)
    hRe = np.real(h)
    hIm = np.imag(h)

    return hRe, hIm


def read_otis_uv_bin(uvfile, ncons):
    URe, UIm, VRe, VIm = [], [], [], []

    for ic in range(ncons):
        with open(uvfile, "rb") as f:
            ll = np.fromfile(f, dtype=np.int32, count=1)
            nm = np.fromfile(f, dtype=np.int32, count=3)
            th_lim = np.fromfile(f, dtype=np.float32, count=2)
            ph_lim = np.fromfile(f, dtype=np.float32, count=2)

            # Need to go from little endian to big endian
            ll.byteswap(True)
            nm.byteswap(True)
            th_lim.byteswap(True)
            ph_lim.byteswap(True)

            n = nm[0]
            m = nm[1]
            nc = nm[2]

            # Read the actual data
            nskip = int((ic) * (nm[0] * nm[1] * 16 + 8) + 8 + ll - 28)
            f.seek(nskip, 1)
            tmp = np.fromfile(f, dtype=np.float32, count=4 * n * m)
            tmp.byteswap(True)

        tmp = np.reshape(tmp, (4 * n, m))
        URe.append(tmp[0 : 4 * n - 3 : 4, :])
        UIm.append(tmp[1 : 4 * n - 2 : 4, :])
        VRe.append(tmp[2 : 4 * n - 1 : 4, :])
        VIm.append(tmp[3 : 4 * n : 4, :])

    URe, UIm = np.array(URe), np.array(UIm)
    VRe, VIm = np.array(VRe), np.array(VIm)

    return URe, UIm, VRe, VIm


def otisbin2xr(gfile, hfile, uvfile, dmin=1.0, outfile=None):
    """ Converts OTIS binary files to xarray.Dataset. To be used when running inverse model
        internally, as it only supports OTIS binary format.
        TODO 
            - at the moment netcdf is in UDS conventions, should be on OTIS netcdf convention
               TIP: use the NCOtis object to help with writting the netcdf or zarr
            - using netCDF4 as legacy, so xarray object is being created after loading netcdf
              file from disk, which is very inneficient - convert the whole thing to xarray

	Args:
        gfile (str):     Path of the constituents model grid on your file system
                            files must be in OTIS binary grid format
        hfile (str):     Path of the elevations constituents file 
        uvifle (str):    Path of the currents constituents file
        outfile (str):   Path of the output file (must have '.nc' or '.zarr' extension)
                            Default = None just returns xarray.Dataset

	Returns
        xarray.Dataset
	    output file saved at outfile path
	    
	Examples
	--------

    otisbin2xr('/path/gridES', '/path/h0.es.out', '/path/u0.es.out', write_to='netcdf', outfile='/path/cons.nc')
	"""
    lon_z, lat_z, lon_u, lat_u, lon_v, lat_v, hz, mz = read_otis_grd_bin(gfile)
    con = read_otis_cons_bin(hfile)
    hRe, hIm = read_otis_h_bin(hfile)
    URe, UIm, VRe, VIm = read_otis_uv_bin(uvfile, len(con))

    uRe, uIm, vRe, vIm = [], [], [], []

    for ic in range(len(con)):
        uRe.append(URe[ic, ...] / hz)
        vRe.append(VRe[ic, ...] / hz)
        uIm.append(UIm[ic, ...] / hz)
        vIm.append(VIm[ic, ...] / hz)

    dims = ("nc", "ny", "nx")

    attrs = {
        "description": "Tidal constituents file in OTIS format",
        "institution": "Oceanum LTD",
    }

    UIm_a = xr.DataArray(
        UIm,
        dims=dims,
        name="UIm",
        attrs={
            "field": "Im(U), vector W->E",
            "long_name": "Tidal transport complex ampl., Imag part, at U-nodes",
            "units": "meter^2/s",
        },
    )
    URe_a = xr.DataArray(
        URe,
        dims=dims,
        name="URe",
        attrs={
            "field": "Re(U), vector W->E",
            "long_name": "Tidal transport complex ampl., Real part, at U-nodes",
            "units": "meter^2/s",
        },
    )
    VIm_a = xr.DataArray(
        VIm,
        dims=dims,
        name="VIm",
        attrs={
            "field": "Im(V), vector W->E",
            "long_name": "Tidal transport complex ampl., Imag part, at V-nodes",
            "units": "meter^2/s",
        },
    )
    VRe_a = xr.DataArray(
        VRe,
        dims=dims,
        name="VRe",
        attrs={
            "field": "Re(V), vector W->E",
            "long_name": "Tidal transport complex ampl., Real part, at V-nodes",
            "units": "meter^2/s",
        },
    )
    con_a = xr.DataArray(
        con, dims=("nc"), name="con", attrs={"long_name": "Tidal constituents"}
    )
    hIm_a = xr.DataArray(
        hIm,
        dims=dims,
        name="hIm",
        attrs={
            "field": "Im(h), scalar",
            "long_name": "Tidal elevation complex amplitude, Imag part",
            "units": "meter",
        },
    )
    hRe_a = xr.DataArray(
        hRe,
        dims=dims,
        name="hRe",
        attrs={
            "field": "Re(h), scalar",
            "long_name": "Tidal elevation complex amplitude, Real part",
            "units": "meter",
        },
    )
    hu_a = xr.DataArray(
        hz,
        dims=("ny", "nx"),
        name="hu",
        attrs={
            "field": "bathy, scalar",
            "long_name": "Bathymetry at U-nodes",
            "units": "meter",
        },
    )
    hv_a = xr.DataArray(
        hz,
        dims=("ny", "nx"),
        name="hv",
        attrs={
            "field": "bathy, scalar",
            "long_name": "Bathymetry at V-nodes",
            "units": "meter",
        },
    )
    hz_a = xr.DataArray(
        hz,
        dims=("ny", "nx"),
        name="hz",
        attrs={
            "field": "bathy, scalar",
            "long_name": "Bathymetry at Z-nodes",
            "units": "meter",
        },
    )
    lat_u_a = xr.DataArray(
        lat_u,
        dims=("ny", "nx"),
        name="lat_u",
        attrs={"long_name": "latitude of U nodes", "units": "degree_north"},
    )
    lat_v_a = xr.DataArray(
        lat_v,
        dims=("ny", "nx"),
        name="lat_v",
        attrs={"long_name": "latitude of V nodes", "units": "degree_north"},
    )
    lat_z_a = xr.DataArray(
        lat_z,
        dims=("ny", "nx"),
        name="lat_z",
        attrs={"long_name": "latitude of Z nodes", "units": "degree_north"},
    )
    lon_u_a = xr.DataArray(
        lon_u,
        dims=("ny", "nx"),
        name="lon_u",
        attrs={"long_name": "longitude of U nodes", "units": "degree_east"},
    )
    lat_v_a = xr.DataArray(
        lat_v,
        dims=("ny", "nx"),
        name="lat_v",
        attrs={"long_name": "longitude of V nodes", "units": "degree_east"},
    )
    lat_z_a = xr.DataArray(
        lat_z,
        dims=("ny", "nx"),
        name="lat_z",
        attrs={"long_name": "longitude of Z nodes", "units": "degree_east"},
    )
    mu_a = xr.DataArray(
        mz,
        dims=("ny", "nx"),
        name="mu",
        attrs={
            "long_name": "water land mask on U nodes",
            "option_0": "land",
            "option_1": "water",
        },
    )
    mv_a = xr.DataArray(
        mz,
        dims=("ny", "nx"),
        name="mv",
        attrs={
            "long_name": "water land mask on V nodes",
            "option_0": "land",
            "option_1": "water",
        },
    )
    mz_a = xr.DataArray(
        mz,
        dims=("ny", "nx"),
        name="mz",
        attrs={
            "long_name": "water land mask on Z nodes",
            "option_0": "land",
            "option_1": "water",
        },
    )
    uIm_a = xr.DataArray(
        uIm,
        dims=dims,
        name="uIm",
        attrs={
            "long_name": "Tidal WE velocities complex ampl., Imag part, at U-nodes",
            "units": "meter/s",
        },
    )
    uRe_a = xr.DataArray(
        uRe,
        dims=dims,
        name="uRe",
        attrs={
            "long_name": "Tidal WE velocities complex ampl., Real part, at U-nodes",
            "units": "meter/s",
        },
    )
    vIm_a = xr.DataArray(
        vIm,
        dims=dims,
        name="vIm",
        attrs={
            "long_name": "Tidal NS velocities complex ampl., Imag part, at V-nodes",
            "units": "meter/s",
        },
    )
    vRe_a = xr.DataArray(
        vRe,
        dims=dims,
        name="vRe",
        attrs={
            "long_name": "Tidal NS velocities complex ampl., Real part, at V-nodes",
            "units": "meter/s",
        },
    )

    ds = xr.Dataset(
        data_vars={
            "UIm": UIm_a,
            "URe": URe_a,
            "VIm": VIm_a,
            "VRe": VRe_a,
            "con": con_a,
            "hIm": hIm_a,
            "hRe": hRe_a,
            "hu": hu_a,
            "hv": hv_a,
            "hz": hz_a,
            "lat_u": lat_u_a,
            "lat_v": lat_v_a,
            "lat_z": lat_z_a,
            "lon_u": lon_u_a,
            "lon_v": lon_v_a,
            "lon_z": lon_z_a,
            "mu": mu_a,
            "mv": mv_a,
            "mz": mz_a,
            "uIm": uIm_a,
            "uRe": uRe_a,
            "vIm": vIm_a,
            "vRe": vRe_a,
        },
        attrs=attrs,
    )

    # TODO, need to write something like _fix_lon() here before saving zarr file

    return ds
