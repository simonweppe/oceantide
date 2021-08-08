"""Tide xarray accessor."""
import os
import re
import glob
import types
from pathlib import Path
from importlib import import_module
from inspect import getmembers, isfunction
import datetime
import warnings
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr

from oceantide.core.utils import nodal, set_attributes
from oceantide.constituents import OMEGA


HERE = Path(__file__).parent


class Plugin(type):
    """Add output functions as bound methods at class creation."""

    def __new__(cls, name, bases, dct):
        module_names = [f.replace(".py", "") for f in glob.glob1(os.path.join(HERE, "output"), "*.py")]
        modules = [import_module(f"oceantide.output.{name}") for name in module_names]
        for module_name, module in zip(module_names, modules):
            for func_name, func in getmembers(module, isfunction):
                if func_name == f"to_{module_name}":
                    dct[func_name] = func
        return type.__new__(cls, name, bases, dct)


@xr.register_dataset_accessor("tide")
class Tide(metaclass=Plugin):
    """Xarray accessor object to process tide constituents dataset."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._validate()

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self._obj))

    def _set_attributes_output(self, dset):
        """Set attributes in output timeseries dataset."""
        set_attributes(dset, "timeseries")
        dset.attrs = {"description": "Tide elevation and currents time series"}
        return dset

    def _validate(self):
        """Check if dataset object has already been constructed.

        The following are checked to ensure it has appropriate format:
            * Required complex variables are present and share same coordinates.
            * Constituents coordinate appropriately formatted.

        """
        required_vars = ["h", "u", "v"]
        if not set(required_vars).issubset(self._obj.data_vars):
            warnings.warn(
                f"Tide accessor requires variables {required_vars} in dataset for full"
                f" functionality but only found {list(self._obj.data_vars.keys())}."
            )

        for v in required_vars:
            if v in self._obj.data_vars and not np.iscomplexobj(self._obj[v]):
                raise ValueError(f"Variable {v} must be complex type.")

        vars_dims = [self._obj[v].dims for v in required_vars]
        if len(set(vars_dims)) != 1:
            raise ValueError(f"Variables {required_vars} must share a common grid.")

        if self._obj.con.dtype.kind != "U":
            raise ValueError(
                f"`con` must be Unicode string dtype, found {self._obj.con.dtype}"
            )

    def amplitude(self, component="h"):
        """Tidal amplitude.

        :math:`\\lambda=\\sqrt{\\Re(z)+\\Im(z)}`

        Args:
            - component (str): Tidal component to calculate amplitude from,
              one of 'h', 'u', 'v'.

        """
        darr = np.absolute(self._obj[component])
        if component == "h":
            darr.attrs = {
                "standard_name": "sea_surface_height_amplitude_due_to_geocentric_ocean_tide",
                "long_name": "Tidal elevation amplitude",
                "units": "m",
            }
        elif component == "u":
            darr.attrs = {
                "standard_name": "eastward_sea_water_velocity_amplitude_due_to_tides",
                "long_name": "Tidal eastward velocity amplitude",
                "units": "m s-1",
            }
        elif component == "v":
            darr.attrs = {
                "standard_name": "northward_sea_water_velocity_amplitude_due_to_tides",
                "long_name": "Tidal northward velocity amplitude",
                "units": "m s-1",
            }
        return darr

    def phase(self, component="h"):
        """Tidal phase.

        Args:
            - component (str): Tidal component to calculate amplitude from,
              one of 'h', 'u', 'v'.

        """
        darr = 360 - (xr.ufuncs.angle(self._obj[component], deg=True)) % 360
        if component == "h":
            darr.attrs = {
                "standard_name": "sea_surface_height_phase_due_to_geocentric_ocean_tide",
                "long_name": "Tidal elevation phase",
                "units": "degree GMT",
            }
        elif component == "u":
            darr.attrs = {
                "standard_name": "eastward_sea_water_velocity_phase_due_to_tides",
                "long_name": "Tidal eastward velocity phase",
                "units": "degree GMT",
            }
        elif component == "v":
            darr.attrs = {
                "standard_name": "northward_sea_water_velocity_phase_due_to_tides",
                "long_name": "Tidal northward velocity phase",
                "units": "degree GMT",
            }
        return darr

    def predict(self, times, time_chunk=50, components=["h", "u", "v"]):
        """Predict tide timeseries.

        Args:
            times (arr): Array of datetime objects or DataArray of times to predict tide over. If an array, a new times dimension will be created.
            time_chunk (float): Time chunk size so that computation fit into memory.
            components (list): Tide variables to predict.

        Returns:
            Dataset predicted tide timeseries components.

        """
        if not components:
            raise ValueError("Choose at least one tide variable to predict")

        if isinstance(times, datetime.datetime):
            times = [times]
        if isinstance(times, (list, tuple, np.ndarray, pd.DatetimeIndex)):
            seconds_array = pd.array(times).astype(int) / 1e9 - 694224000
            tsec = xr.DataArray(
                data=seconds_array,
                coords={"time": times},
                dims=("time",),
            ).chunk({"time": time_chunk})
        elif isinstance(times, xr.DataArray):
            tsec = times.astype(int) / 1e9 - 694224000
        else:
            raise TypeError(
                "times argument must be a list of datetimes, pandas.DatetimeIndex, "
                "numpy array of datetime64 or xarray DataArray of datetime64"
            )

        cons = list(self._obj.con.values)
        pu, pf, v0u = nodal(tsec[0] / 86400 + 48622.0, cons)

        # Variables for calculations
        pf = xr.DataArray(pf, coords={"con": cons}, dims=("con",))
        pu = xr.DataArray(pu, coords={"con": cons}, dims=("con",))
        v0u = xr.DataArray(v0u, coords={"con": cons}, dims=("con",))
        omega = xr.DataArray(
            data=[OMEGA[c] for c in cons], coords={"con": cons}, dims=("con",)
        )

        cos = da.cos(tsec * omega + v0u + pu)
        sin = da.sin(tsec * omega + v0u + pu)

        dset = xr.Dataset()
        if "dep" in self._obj.data_vars:
            dset["dep"] = self._obj.dep
        if "h" in components:
            dset["h"] = cos * pf * self._obj["h"].real - sin * pf * self._obj["h"].imag
        if "u" in components:
            dset["u"] = cos * pf * self._obj["u"].real - sin * pf * self._obj["u"].imag
        if "v" in components:
            dset["v"] = cos * pf * self._obj["v"].real - sin * pf * self._obj["v"].imag
        dset = dset.sum(dim="con", skipna=False)
        dset = self._set_attributes_output(dset)

        return dset

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
        u = self._obj.u.real * xr.ufuncs.exp(-i * self._obj.u.imag)
        v = self._obj.v.real * xr.ufuncs.exp(-i * self._obj.v.imag)

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

        dset = xr.Dataset()
        dset["SEMA"] = SEMA
        dset["ECC"] = ECC
        dset["INC"] = INC
        dset["PHA"] = PHA
        dset["w"] = w

        return dset
