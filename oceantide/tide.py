"""Tide xarray accessor."""
import re
import datetime
import numpy as np
import dask.array as da
import xarray as xr

from oceantide.core.utils import nodal
from oceantide.constituents import OMEGA


@xr.register_dataset_accessor("tide")
class Tide:
    """Xarray accessor object to process tide constituents dataset."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._validate()

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self._obj))

    def _set_attributes_output(self, dset):
        """Set attributes in output timeseries dataset."""
        dset.attrs = {
            "description": "Tide elevation and currents prediction time series",
        }
        dset["et"].attrs = {
            "standard_name": "tidal_sea_surface_height_above_mean_sea_level",
            "units": "m",
        }
        dset["ut"].attrs = {
            "standard_name": "eastward_sea_water_velocity_due_to_tides",
            "units": "m s-1",
        }
        dset["vt"].attrs = {
            "standard_name": "northward_sea_water_velocity_due_to_tides",
            "units": "m s-1",
        }
        return dset

    def _nodal_time(self, time):
        """Nodal time.

        Args:
            time (datetime): Time to get nodal time from.

        """
        return (time - datetime.datetime(1992, 1, 1)).days + 48622.0

    def _validate(self):
        """Check if dataset object has already been constructed.

        The following are checked to ensure it has appropriate format:
            * Required complex variables are present and share same coordinates.
            * Constituents coordinate appropriately formatted.

        """
        required_vars = ["et", "ut", "vt"]
        if not set(required_vars).issubset(self._obj.data_vars):
            raise ValueError(
                f"Tide accessor requires variables {required_vars} in dataset but "
                f"only found {list(self._obj.data_vars.keys())}."
            )

        for v in required_vars:
            if not np.iscomplexobj(self._obj[v]):
                raise ValueError(f"Variable {v} must be complex type.")

        vars_dims = [self._obj[v].dims for v in required_vars]
        if len(set(vars_dims)) != 1:
            raise ValueError(f"Variables {required_vars} must share a common grid.")

        if self._obj.con.dtype != np.dtype("<U2"):
            raise ValueError(f"Coordinate `con` must be <U2 dtype, found {self._obj.con.dtype}")

    def predict(self, times, time_chunk=50):
        """Predict tide timeseries.

        Args:
            time (arr): Array of datetime objects to predict tide over.
            time_chunk (float): Time chunk size so that computation fit into memory.

        Returns:
            Dataset of tide currents and elevations timeseries.

        """
        conlist = list(self._obj.con.values)

        pu, pf, v0u = nodal(self._nodal_time(times[0]), conlist)

        # Variables for calculations
        pf = xr.DataArray(pf, coords={"con": conlist}, dims=("con",))
        pu = xr.DataArray(pu, coords={"con": conlist}, dims=("con",))
        v0u = xr.DataArray(v0u, coords={"con": conlist}, dims=("con",))
        omega = xr.DataArray(
            data=[OMEGA[c] for c in conlist], coords={"con": conlist}, dims=("con",)
        )
        tsec = xr.DataArray(
            data=[(t - datetime.datetime(1992, 1, 1)).total_seconds() for t in times],
            coords={"time": times},
            dims=("time",),
        ).chunk({"time": time_chunk})

        cos = da.cos(tsec * omega + v0u + pu)
        sin = da.sin(tsec * omega + v0u + pu)

        dsout = xr.Dataset()
        dsout["et"] = cos * pf * self._obj["et"].real - sin * pf * self._obj["et"].imag
        dsout["ut"] = cos * pf * self._obj["ut"].real - sin * pf * self._obj["ut"].imag
        dsout["vt"] = cos * pf * self._obj["vt"].real - sin * pf * self._obj["vt"].imag
        dsout = dsout.sum(dim="con", skipna=False)
        dsout = self._set_attributes_output(dsout)

        return dsout

    def ellipse(self):
        """Tidal ellipse parameters.

        Convert tidal amplitude and phase lag into tidal ellipse parameters.

        Returns:
            Dataset with ellipse variables:
                SEMA: Semi-major axes, the maximum speed.
                ECC: Eccentricity, the ratio of semi-minor axis over the semi-major
                    axis, negative value indicates ellipse traversed clockwise.
                INC: Inclination, the angles (in degrees) between the semi-major axes
                    and u-axis.
                PHA: Phase angles, the time (in angles and in degrees) when the tidal
                    currents reach their maximum speeds (i.e.  PHA=omega * tmax).
                w: A matrix whose rows allow for plotting ellipses and whose columns
                    are for different ellipses corresponding columnwise to SEMA. For
                    example, plot(np.real(w[0, :]), np.imag(w[0, :])) will let you
                    see the first ellipse. You may need to use squeeze function when
                    w is a more than two dimensional array. See example.py.

        Modified from:
        _______________________________________________________________________
        Zhigang Xu, Ph.D.
        (pronounced as Tsi Gahng Hsu)
        Research Scientist
        Coastal Circulation
        Bedford Institute of Oceanography
        1 Challenge Dr.
        P.O. Box 1006                    Phone  (902) 426-2307 (o)
        Dartmouth, Nova Scotia           Fax    (902) 426-7827
        CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca
        _______________________________________________________________________

        Reference:
            Xu, Zhigang (2000, 2002), Ellipse Parameters Conversion and Velocity Profiles for Tidal Currents in Matlab.
            https://svn.oss.deltares.nl/repos/openearthtools/trunk/matlab/applications/DelftDashBoard/utils/tidal_ellipse/tidal_ellipse.ps

        """
        # Complex amplitudes for u and v
        i = 1j
        u = self._obj.ut.real * xr.ufuncs.exp(-i * self._obj.ut.imag)
        v = self._obj.vt.real * xr.ufuncs.exp(-i * self._obj.vt.imag)

        # Calculate complex radius of clockwise circles
        wp = (u + i * v) / 2
        # Calculate complex radius of anticlockwise circles
        wm = np.conj(u - i * v) / 2
        # Amplitudes and angles
        Wp = np.abs(wp)
        Wm = np.abs(wm)
        THETAp = xr.ufuncs.angle(wp)
        THETAm = xr.ufuncs.angle(wm)

        # calculate ellipse parameters
        SEMA = Wp + Wm  # Semi Major Axis, or maximum speed
        SEMI = Wp - Wm  # Semi Minor Axis, or minimum speed
        ECC = SEMI / SEMA  # Eccentricity

        # Phase angle, the time (in angle) when the velocity reaches the maximum
        PHA = (THETAm - THETAp) / 2
        # Inclination, the angle between the semi major axis and x-axis (or u-axis)
        INC = (THETAm + THETAp) / 2

        # convert to degrees for output
        PHA = PHA / np.pi * 180
        INC = INC / np.pi * 180
        THETAp = THETAp / np.pi * 180
        THETAm = THETAm / np.pi * 180

        # map the resultant angles to the range of [0, 360].
        PHA = np.mod(PHA + 360, 360)
        INC = np.mod(INC + 360, 360)

        k = xr.ufuncs.fix(INC / 180)
        INC = INC - k * 180
        PHA = PHA + k * 180
        PHA = np.mod(PHA, 360)

        ndot = np.prod(np.shape(SEMA))
        dot = 2 * np.pi / ndot
        ot = np.arange(0, 2 * np.pi, dot)
        wp_stacked = wp.stack({"stacked": ("con", "lat", "lon")})
        wm_stacked = wm.stack({"stacked": ("con", "lat", "lon")})
        w = (wp_stacked * np.exp(i * ot) + wm_stacked * np.exp(-i * ot)).unstack()

        dsout = xr.Dataset()
        dsout["SEMA"] = SEMA
        dsout["ECC"] = ECC
        dsout["INC"] = INC
        dsout["PHA"] = PHA
        dsout["w"] = w

        return dsout
