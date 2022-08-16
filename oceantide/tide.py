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
        module_names = [
            f.replace(".py", "")
            for f in glob.glob1(os.path.join(HERE, "output"), "*.py")
        ]
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

        Returns:
            - amp (DataArray): Amplitudes for component :math:`\\lambda(con,lat,lon)`.

        """
        darr = np.absolute(self._obj[component])
        darr.name = f"amp{component}"
        return darr

    def phase(self, component="h"):
        """Tidal phase relative to GMT.

        Args:
            - component (str): Tidal component to calculate amplitude from,
              one of 'h', 'u', 'v'.

        Returns:
            - phi (DataArray): Phases for component :math:`\\phi(con,lat,lon)`.

        """
        darr = self._obj[component]
        darr = np.rad2deg(np.arctan2(-darr.imag, darr.real)) % 360
        darr.name = f"phi{component}"
        set_attributes(darr, "dataset")
        return darr

    def predict(self, times, time_chunk=50, components=["h", "u", "v"]):
        """Predict tide timeseries.

        Args:
            times (arr): Array of datetime objects or DataArray of times to predict tide over. If an array, a new times dimension will be created.
            time_chunk (float): Time chunk size so that computation fit into memory.
            components (list): Tide variables to predict.

        Returns:
            ds (Dataset): Predicted tide timeseries components :math:`\\eta(time,lat,lon)`.

        """
        if not components or {"h", "u", "v"} - set(components) == {"h", "u", "v"}:
            raise ValueError("Choose at least one tide component h, u or v to predict")

        if isinstance(times, datetime.datetime):
            times = [times]
        if isinstance(times, (list, tuple, np.ndarray, pd.DatetimeIndex)):
            seconds_array = pd.array(times).view(int) / 1e9 - 694224000
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

        # Slice out unsupported cons
        non_supported = set(cons) - OMEGA.keys()
        if non_supported:
            warnings.warn(f"Cons {non_supported} not supported and will be ignored")
            cons = [con for con in cons if con not in non_supported]
            ds = self._obj.sel(con=cons)
        else:
            ds = self._obj

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
        if "dep" in ds.data_vars:
            dset["dep"] = ds.dep
        if "h" in components:
            dset["h"] = cos * pf * ds["h"].real - sin * pf * ds["h"].imag
        if "u" in components:
            dset["u"] = cos * pf * ds["u"].real - sin * pf * ds["u"].imag
        if "v" in components:
            dset["v"] = cos * pf * ds["v"].real - sin * pf * ds["v"].imag
        dset = dset.sum(dim="con", skipna=False)
        dset = self._set_attributes_output(dset)

        return dset
