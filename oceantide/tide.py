"""Tide xarray accessor."""
import re
import datetime
import numpy as np
import dask.array as da
import xarray as xr

from oceantide.core import nodal
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
        if not {"h", "u", "v"}.issubset(self._obj.data_vars):
            return False
        for v in ["h", "u", "v"]:
            if not np.iscomplexobj(self._obj[v]):
                return False
        if not (
            self._obj.h.coords.indexes
            == self._obj.u.coords.indexes
            == self._obj.v.coords.indexes
        ):
            return False
        if self._obj.con.dtype != np.dtype("<U2"):
            return False
        return True

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
