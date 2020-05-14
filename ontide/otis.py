# -*- coding: utf-8 -*-

"""OTIS tools."""

import os
import logging
import numpy as np
import numpy.ma as ma
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
        x0, x1, y0, y1 (float):  Bounds for subsetting, default = None, which is no subsetting
        catalog (str):           Intake catalog that has the source constituents dataset
        namespace (str):         Intake namespace
    """

    name = "otis"  # necessary for remap to play nicely

    def __init__(
        self,
        model="tpxo9",
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
        _repr = "<OTIS {} nc={} x0={:0.2f} x1={:0.2f} y0={:0.2f} y1={:0.2f} subset={}>".format(
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

        for k in range(self.ds.dims["con"]):
            for varname, var in self.ds.data_vars.items():
                if len(var.dims) > 2:  # leaving depths and masks out
                    for key, coord in var.coords.items():
                        if "lon" in key:
                            lon = coord.values
                        if "lat" in key:
                            lat = coord.values

                    lon, lat = np.meshgrid(lon, lat)
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
                        var.values[k, ...], wet, dry, lon, lat, dmax,
                    )


def predict_tide_grid(
    lon,
    lat,
    time,
    model="tpxo9",
    catalog=ONTAKE_CATALOG,
    namespace=ONTAKE_NAMESPACE,
    conlist=None,
    outfile=None,
):
    """ Performs a tidal prediction at all points in [lon,lat] at times.

	Args:
	
	lon, lat (numpy ndarray): Each is an n-length array of longitude 
                                and latitudes in degrees to perform predictions at.
                                Lat ranges from -90 to 90. Lon can range from -180 to 360.
  	time:                     m-length array of times.  Acceptable formats are 
                                a list of `datetime` objects, a list or array of 
                                `np.datetime64` objects, or pandas date_range
    model (str):              Intake dataset of the regional OTIS model. 
                                TIP: use ontake to discover datasets:
                                    ot = Ontake(namespace='tide', 
                                                master_url='gs://oceanum-catalog/oceanum.yml')
                                    ot.datasets
    catalog (str):            Intake catalog that has the source constituents dataset
    namespace (str):          Intake namespace
	conlist :                 List of strings (optional). If supplied, gives a list 
                                of tidal constituents to include in prediction. 
                                Available are 'M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1'
    outfile:                  Writes xarray.Dataset to disk as a NetCDF file

	Returns
	xarray.Dataset containing:
	    et : 3D numpy array of tidal heights
		     height is in meters
	    ut : 3D numpy array of eastward tidal velocity [m/s]
	    vt : 3D numpy array of northward tidal velocity [m/s]

	Examples
	--------

	dates = np.arange(np.datetime64('2001-04-03'),
	                  np.datetime64('2001-05-03'), dtype='datetime64[h]' )

	lon = np.array([198, 199])
	lat = np.array([21, 19])

	ds = predict_tide_grid(lon, lat, time) 
	"""

    otis = NCOtis(
        model,
        x0=lon.min(),
        x1=lon.max(),
        y0=lat.min(),
        y1=lat.max(),
        catalog=catalog,
        namespace=namespace,
    )
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

    print("Converting complex harmonics into timeseries")
    # TODO: this is very inneficient for medium to large grids, need to optimize somehow
    for k, om in enumerate(omega):
        print(f'    constituent {k+1} | {len(omega)}')
        for idx in range(nj * ni):
            for p in ["h", "u", "v"]:
                rvars[p][:, idx] += pf[k] * cvars[f"{p}Re"][k, idx] * np.cos(
                    om * tsec + v0u[k] + pu[k]
                ) - pf[k] * cvars[f"{p}Im"][k, idx] * np.sin(
                    om * tsec + v0u[k] + pu[k]
                )

    for varname, var in rvars.items():
        rvars[varname] = var.reshape((nt, nj, ni))

    rvars = _remask(rvars, otis, lon, lat)
    ds = make_timeseries_dataset(time, lon, lat, rvars["h"], rvars["u"], rvars["v"])

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
                otis.ds.coords[f"lon_{p}"],
                otis.ds.coords[f"lat_{p}"],
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


def _fix_east(ds):
    """ Convert 0 < lon < 360 to -180 < lon < 180 and shift all vars accordingly. 
            IMPORTANT: this is peculiar to OTIS netcdf file provided by their servers.
                       It is not meant to work generically with any xarray.Dataset
            Args:
                ds (xarray.Dataset) :: Input xarray dataset

            Returns:
                xarray.Dataset
    """
    idx = {}

    lonz = ds.lon_z.values.copy()
    lonz[lonz > 180] -= 360
    idx["z"] = np.argsort(lonz)

    lonu = ds.lon_u.values.copy()
    lonu[lonu > 180] -= 360
    idx["u"] = np.argsort(lonu)

    lonv = ds.lon_v.values.copy()
    lonv[lonv > 180] -= 360
    idx["v"] = np.argsort(lonv)

    print("shifting along x-axis: ")
    for varname, var in ds.data_vars.items():
        if "ny" in var.dims and "nx" in var.dims:
            print(varname)
            vals = var.values

            if len(var.dims) > 2:
                if "z" in varname or "h" in varname:
                    vals = np.take_along_axis(
                        vals, idx["z"][None, ...].repeat(ds.dims["nc"], axis=0), axis=-1
                    )
                elif "u" in varname or "U" in varname:
                    vals = np.take_along_axis(
                        vals, idx["u"][None, ...].repeat(ds.dims["nc"], axis=0), axis=-1
                    )
                elif "v" in varname or "V" in varname:
                    vals = np.take_along_axis(
                        vals, idx["v"][None, ...].repeat(ds.dims["nc"], axis=0), axis=-1
                    )
            else:
                if "z" in varname or "h" in varname:
                    vals = np.take_along_axis(vals, idx["z"], axis=-1)
                elif "u" in varname or "U" in varname:
                    vals = np.take_along_axis(vals, idx["u"], axis=-1)
                elif "v" in varname or "V" in varname:
                    vals = np.take_along_axis(vals, idx["v"], axis=-1)

            if "lon" in varname:
                vals[vals > 180] -= 360

            ds[varname].values = vals

    return ds


def _transp2vel(ds):
    """ Compute complex velocities based on complex transports and append 
            them to the xr.Dataset
        
        """
    print("Computing complex velocities based on complex transports")
    longname = "Tidal WE transport complex ampl., {c} part, at {n}-nodes"
    variables = dict(uRe=None, uIm=None, vRe=None, vIm=None)

    for node in ["u", "v"]:
        for com in ["Re", "Im"]:
            variables["{node}{con}"] = xr.Variable(
                ds["{node.upper()}{con}"].dims,
                ds["{node.upper()}{con}"].values
                / ds["h{node}"].values,
                attrs=dict(
                    long_name=longname.format(c=com, n=node.upper()), units="meter/s",
                ),
            )

    ds = ds.assign(variables=variables)
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
                                 for most cases because complex params are more
                                 appropriate for remapping)

    Developer notes:
        - this should replace both .load_otis and .CGrid_OTIS from pyroms
    """
    modelfile = get_mapper(model).root

    with open(modelfile) as f:
        lines = f.readlines()
        elevfile = os.path.join(
            os.path.dirname(modelfile), lines[0].split("/")[-1]
        ).strip()
        curfile = os.path.join(
            os.path.dirname(modelfile), lines[1].split("/")[-1]
        ).strip()
        gfile = os.path.join(
            os.path.dirname(modelfile), lines[2].split("/")[-1]
        ).strip()

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
    ds = _transp2vel(ds)

    # TODO: write case where amplitude parameters need to be written and replace tpxo9 zarr at least
    ds = make_cons_dataset(
        ds.con.values,
        ds.lon_z.values,
        ds.lat_z.values,
        ds.lon_u.values,
        ds.lat_u.values,
        ds.lon_v.values,
        ds.lat_v.values,
        ds.hRe.values,
        ds.hIm.values,
        ds.uRe.values,
        ds.uIm.values,
        ds.vRe.values,
        ds.vIm.values,
        ds.URe.values,
        ds.UIm.values,
        ds.VRe.values,
        ds.VIm.values,
        ds.hz.values,
        ds.hu.values,
        ds.hv.values,
        ds.mz.values,
        ds.mu.values,
        ds.mv.values,
    )
    ds.to_zarr(
        os.path.join("/tmp", os.path.basename(outfile)), consolidated=True, mode="w"
    )
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

    # if (lon_z[0, 0] < 0) & (lon_z[0, 1] < 0): TODO need to write a proper fix_lon here
    #     lon_z = lon_z + 360

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

    return cons


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
            h.append(np.reshape(htemp[::2] + 1j * htemp[1::2], (nlon, nlat)))

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

        tmp = np.reshape(tmp, (m, 4 * n))
        URe.append(tmp[:, 0 : 4 * n - 3 : 4])
        UIm.append(tmp[:, 1 : 4 * n - 2 : 4])
        VRe.append(tmp[:, 2 : 4 * n - 1 : 4])
        VIm.append(tmp[:, 3 : 4 * n : 4])

    URe, UIm = np.array(URe), np.array(UIm)
    VRe, VIm = np.array(VRe), np.array(VIm)

    return URe, UIm, VRe, VIm


def otisbin2xr(gfile, hfile, uvfile, dmin=1.0, outfile=None):
    """ Converts OTIS binary files to xarray.Dataset. To be used when running inverse model
        internally, as it only supports OTIS binary format.

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

    uIm, uRe = np.array(uIm), np.array(uRe)
    vIm, vRe = np.array(vIm), np.array(vRe)

    # applying landmask
    landmask = mz[None, ...].repeat(con.size, axis=0)
    hRe = ma.masked_where(landmask == 0, hRe)
    hIm = ma.masked_where(landmask == 0, hIm)
    uRe = ma.masked_where(landmask == 0, uRe)
    uIm = ma.masked_where(landmask == 0, uIm)
    URe = ma.masked_where(landmask == 0, URe)
    UIm = ma.masked_where(landmask == 0, UIm)
    vRe = ma.masked_where(landmask == 0, vRe)
    vIm = ma.masked_where(landmask == 0, vIm)
    VRe = ma.masked_where(landmask == 0, vRe)
    VIm = ma.masked_where(landmask == 0, VIm)

    # creating xarray.Dataset ----------------------------------------------------
    ds = make_cons_dataset(
        con,
        lon_z,
        lat_z,
        lon_u,
        lat_u,
        lon_v,
        lat_v,
        hRe,
        hIm,
        uRe,
        uIm,
        vRe,
        vIm,
        URe,
        UIm,
        VRe,
        VIm,
        hz,
        hz,
        hz,
        mz,
        mz,
        mz,
    )

    if outfile.endswith(".zarr"):
        logging.info(f"Writting {outfile}")
        ds.to_zarr(get_mapper(outfile), consolidated=True)
    elif outfile.endswith(".nc"):
        ds.to_netcdf(outfile, format="NETCDF4")
    else:
        raise Exception(
            "outfile must have either .nc or .zarr extension and be comptible with fsspec notation"
        )

    return ds


def make_cons_dataset(
    con,
    lon_z,
    lat_z,
    lon_u,
    lat_u,
    lon_v,
    lat_v,
    hRe,
    hIm,
    uRe,
    uIm,
    vRe,
    vIm,
    URe,
    UIm,
    VRe,
    VIm,
    hz,
    hu,
    hv,
    mz,
    mu,
    mv,
    ha=None,
    hp=None,
    ua=None,
    up=None,
    va=None,
    vp=None,
    Ua=None,
    Up=None,
    Va=None,
    Vp=None,
    attrs={
        "description": "Tidal constituents in OTIS format",
        "institution": "Oceanum Ltd",
    },
):
    """ Create xarray.Dataset with tidal constituents consisting of merged OTIS format
            based on raw numpy arrays.

        It can be used to create netcdf or zarr files
        
        REQUIREMENTS: 
            - All variables must be masked
            - Elevations and currents arrays must have (nc, ny, nx) dimensions, where
                    nc = number of constituents
            - Depths and landmasks must have (ny, nx) dimensions
            - Constituents arrays must be numpy.ndarray with dtype='|S4' 

    Args:
        ----------- Coordinates (required) ---------------------------------------------
        con (numpy.ndarray, dtype='|S4'): Tidal constituents
        lon_z (numpy.ndarray 2D): Lon coord array at Z-points 
        lat_z (numpy.ndarray 2D): Lat coord array at Z-points 
        lon_u (numpy.ndarray 2D): Lon coord array at U-points 
        lat_u (numpy.ndarray 2D): Lat coord array at U-points 
        lon_v (numpy.ndarray 2D): Lon coord array at V-points 
        lat_v (numpy.ndarray 2D): Lat coord array at V-points 
        ----------- Complex parameters (required) --------------------------------------
        hRe, hIm (numpy.ma.core.MaskedArray 3D): Complex tidal elevation amplitude
        uRe, uIm (numpy.ma.core.MaskedArray 3D): Complex tidal U-current amplitude
        vRe, vIm (numpy.ma.core.MaskedArray 3D): Complex tidal V-current amplitude
        URe, UIm (numpy.ma.core.MaskedArray 3D): Complex tidal U-transport amplitude
        VRe, VIm (numpy.ma.core.MaskedArray 3D): Complex tidal V-transport amplitude
        ----------- Miscelania ---------------------------------------------------------
        hz, hu, hv (numpy.ndarray 2D): Depths at U, V, Z points
        mz, mu, mv (numpy.ndarray 2D): Landmask at U, V, Z points
        attrs (dict): Dataset global attributes dictionary (optional)  
        ----------- Amplitude / Phase parameters (optional) ----------------------------
        ha, hp   (numpy.ma.core.MaskedArray 3D): Elevation amplitude and phase (optional)
        ua, up   (numpy.ma.core.MaskedArray 3D): U-current amplitude and phase (optional)
        va, vp   (numpy.ma.core.MaskedArray 3D): V-current amplitude and phase (optional)
        Ua, Up   (numpy.ma.core.MaskedArray 3D): U-transport amplitude and phase (optional)
        Va, Vp   (numpy.ma.core.MaskedArray 3D): V-transport amplitude and phase (optional)
        --------------------------------------------------------------------------------

    Returns:
        ds (xarray.Dataset)

    TODO: perhaps use CDL to create a skeleton Dataset and fill it up?
    TODO: write the amplitude parameters metadata and include them
    """
    assert not any([ha, hp, ua, up, va, vp, Ua, Up, Va, Vp]) or all(
        [ha, hp, ua, up, va, vp, Ua, Up, Va, Vp]
    ), "All Amplitude / Phase parameters must be provided"

    # coordinates
    cc = xr.IndexVariable(["con"], con, attrs={"long_name": "Tidal constituents"})

    yu = xr.IndexVariable(
        ["lat_u"],
        lat_u[:, 0],
        attrs={"long_name": "latitude of U nodes", "units": "degree_north"},
    )
    xu = xr.IndexVariable(
        ["lon_u"],
        lon_u[0, :],
        attrs={"long_name": "longitude of U nodes", "units": "degree_east"},
    )
    yv = xr.IndexVariable(
        ["lat_v"],
        lat_v[:, 0],
        attrs={"long_name": "latitude of V nodes", "units": "degree_north"},
    )
    xv = xr.IndexVariable(
        ["lon_v"],
        lon_v[0, :],
        attrs={"long_name": "longitude of V nodes", "units": "degree_east"},
    )
    yz = xr.IndexVariable(
        ["lat_z"],
        lat_z[:, 0],
        attrs={"long_name": "latitude of Z nodes", "units": "degree_north"},
    )
    xz = xr.IndexVariable(
        ["lon_z"],
        lon_z[0, :],
        attrs={"long_name": "longitude of Z nodes", "units": "degree_east"},
    )

    # data variables
    UIm_a = xr.DataArray(
        UIm,
        coords=[cc, yu, xu],
        name="UIm",
        attrs={
            "field": "Im(U), vector W->E",
            "long_name": "Tidal transport complex ampl., Imag part, at U-nodes",
            "units": "meter^2/s",
        },
    )
    URe_a = xr.DataArray(
        URe,
        coords=[cc, yu, xu],
        name="URe",
        attrs={
            "field": "Re(U), vector W->E",
            "long_name": "Tidal transport complex ampl., Real part, at U-nodes",
            "units": "meter^2/s",
        },
    )
    VIm_a = xr.DataArray(
        VIm,
        coords=[cc, yv, xv],
        name="VIm",
        attrs={
            "field": "Im(V), vector W->E",
            "long_name": "Tidal transport complex ampl., Imag part, at V-nodes",
            "units": "meter^2/s",
        },
    )
    VRe_a = xr.DataArray(
        VRe,
        coords=[cc, yv, xv],
        name="VRe",
        attrs={
            "field": "Re(V), vector W->E",
            "long_name": "Tidal transport complex ampl., Real part, at V-nodes",
            "units": "meter^2/s",
        },
    )
    hIm_a = xr.DataArray(
        hIm,
        coords=[cc, yz, xz],
        name="hIm",
        attrs={
            "field": "Im(h), scalar",
            "long_name": "Tidal elevation complex amplitude, Imag part",
            "units": "meter",
        },
    )
    hRe_a = xr.DataArray(
        hRe,
        coords=[cc, yz, xz],
        name="hRe",
        attrs={
            "field": "Re(h), scalar",
            "long_name": "Tidal elevation complex amplitude, Real part",
            "units": "meter",
        },
    )
    hu_a = xr.DataArray(
        hz,
        coords=[yu, xu],
        name="hu",
        attrs={
            "field": "bathy, scalar",
            "long_name": "Bathymetry at U-nodes",
            "units": "meter",
        },
    )
    hv_a = xr.DataArray(
        hz,
        coords=[yv, xv],
        name="hv",
        attrs={
            "field": "bathy, scalar",
            "long_name": "Bathymetry at V-nodes",
            "units": "meter",
        },
    )
    hz_a = xr.DataArray(
        hz,
        coords=[yz, xz],
        name="hz",
        attrs={
            "field": "bathy, scalar",
            "long_name": "Bathymetry at Z-nodes",
            "units": "meter",
        },
    )
    mu_a = xr.DataArray(
        mz,
        coords=[yu, xu],
        name="mu",
        attrs={
            "long_name": "water land mask on U nodes",
            "option_0": "land",
            "option_1": "water",
        },
    )
    mv_a = xr.DataArray(
        mz,
        coords=[yv, xv],
        name="mv",
        attrs={
            "long_name": "water land mask on V nodes",
            "option_0": "land",
            "option_1": "water",
        },
    )
    mz_a = xr.DataArray(
        mz,
        coords=[yz, xz],
        name="mz",
        attrs={
            "long_name": "water land mask on Z nodes",
            "option_0": "land",
            "option_1": "water",
        },
    )
    uIm_a = xr.DataArray(
        uIm,
        coords=[cc, yu, xu],
        name="uIm",
        attrs={
            "long_name": "Tidal WE velocities complex ampl., Imag part, at U-nodes",
            "units": "meter/s",
        },
    )
    uRe_a = xr.DataArray(
        uRe,
        coords=[cc, yu, xu],
        name="uRe",
        attrs={
            "long_name": "Tidal WE velocities complex ampl., Real part, at U-nodes",
            "units": "meter/s",
        },
    )
    vIm_a = xr.DataArray(
        vIm,
        coords=[cc, yv, xv],
        name="vIm",
        attrs={
            "long_name": "Tidal NS velocities complex ampl., Imag part, at V-nodes",
            "units": "meter/s",
        },
    )
    vRe_a = xr.DataArray(
        vRe,
        coords=[cc, yv, xv],
        name="vRe",
        attrs={
            "long_name": "Tidal NS velocities complex ampl., Real part, at V-nodes",
            "units": "meter/s",
        },
    )

    if all([ha, hp, ua, up, va, vp, Ua, Up, Va, Vp]):
        # TODO: write those data variables here and add them to the dataset below
        raise NotImplementedError

    data_vars = {
        "UIm": UIm_a,
        "URe": URe_a,
        "VIm": VIm_a,
        "VRe": VRe_a,
        "hIm": hIm_a,
        "hRe": hRe_a,
        "hu": hu_a,
        "hv": hv_a,
        "hz": hz_a,
        "mu": mu_a,
        "mv": mv_a,
        "mz": mz_a,
        "uIm": uIm_a,
        "uRe": uRe_a,
        "vIm": vIm_a,
        "vRe": vRe_a,
    }

    if all([ha, hp, ua, up, va, vp, Ua, Up, Va, Vp]):
        # TODO: write those data variables here and add them to the dataset below
        #     for var in [ha_a, hp_a, ua_a, up_a, va_a, vp_a, Ua_a, Up_a, Va_a, Vp_a]:
        #         data_vars[var["name"]] = var
        raise NotImplementedError

    ds = xr.Dataset(data_vars=data_vars, attrs=attrs)

    return ds


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
