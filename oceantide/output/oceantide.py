"""Oceantide output."""
from pathlib import Path
import numpy as np
import xarray as xr
from zarr.codecs import FixedScaleOffset

from oceantide.core.utils import set_attributes, compute_scale_and_offset


AMPMIN = -20.0
AMPMAX = 20.0
DEPMIN = 0.0
DEPMAX = 12000.0
SCALE_FACTOR_D, ADD_OFFSET_D = compute_scale_and_offset(DEPMIN, DEPMAX)
SCALE_FACTOR_A, ADD_OFFSET_A = compute_scale_and_offset(AMPMIN, AMPMAX)

FILE_FORMATS = {
    ".nc": "netcdf",
    ".zarr": "zarr",
}


def to_oceantide(self, filename: str, file_format: str = None, **kwargs):
    """Write dataset as Oceantide format.

    The oceantide format has complex constituents variables split into real and imag
    variables allowing for better packing and making it easier to support netcdf.

    Parameters
    ----------
    self (oceantide.tide.Tide)
        Oceantide Tide instance.
    filename (str)
        Name for ouput file.
    file_format (str)
        Format for output file, `zarr` and `netcdf` are supported. If not specified it
        is guessed from the filename extension.
    kwargs
        Keyword argument to pass to to_netcdf or to_zarr method.

    """
    dset = self._obj[["dep"]]
    for v in ["h", "u", "v"]:
        if v in self._obj:
            dset[f"{v}_real"] = self._obj[v].real
            dset[f"{v}_imag"] = self._obj[v].imag
    set_attributes(dset, "oceantide")

    ext = Path(filename).suffix
    if not file_format:
        try:
            file_format = FILE_FORMATS[ext]
        except KeyError as err:
            raise ValueError(
                "The filename extension must be one of ['.nc', '.zarr'] if "
                f"file_format is not specified, got '{ext}'"
            ) from err
    try:
        writer = globals()[f"_write_{file_format}"]
    except KeyError as err:
        raise ValueError(
            f"Supported file formats are ['netcdf', 'zarr'], got '{file_format}'"
        ) from err
    writer(dset, filename, **kwargs)


def _write_zarr(dset: xr.Dataset, filename: str, **kwargs):
    """Write oceantide zarr file format.

    Parameters
    ----------
    dset (xr.Dataset)
        Dataset to write.
    filename (str)
        Name for ouput file.
    kwargs
        Keyword argument to pass to to_zarr method.

    """
    kw = {"dtype": "float32", "astype": "int16"}
    fd = FixedScaleOffset(offset=ADD_OFFSET_D, scale=1 / SCALE_FACTOR_D, **kw)
    fa = FixedScaleOffset(offset=ADD_OFFSET_A, scale=1 / SCALE_FACTOR_A, **kw)

    dset.dep.encoding = {"filters": [fd], "_FillValue": DEPMAX, "dtype": kw["dtype"]}
    for varname, dvar in dset.data_vars.items():
        if varname == "dep":
            continue
        dvar.encoding = {"filters": [fa], "_FillValue": AMPMAX, "dtype": kw["dtype"]}

    dset.to_zarr(filename, **kwargs)


def _write_netcdf(dset: xr.Dataset, filename: str, **kwargs):
    """Write oceantide netcdf file format.

    Parameters
    ----------
    dset (xr.Dataset)
        Dataset to write.
    filename (str)
        Name for ouput file.
    kwargs
        Keyword argument to pass to to_netcdf method.

    """
    encoding = {"zlib": True, "_FillValue": np.int16(2**16 / 2 - 1), "dtype": "int16"}
    encd = {**encoding, **{"scale_factor": SCALE_FACTOR_D, "add_offset": ADD_OFFSET_D}}
    enca = {**encoding, **{"scale_factor": SCALE_FACTOR_A, "add_offset": ADD_OFFSET_A}}

    for varname, dvar in dset.data_vars.items():
        if varname == "dep":
            dvar.encoding = encd
        else:
            dvar.encoding = enca

    dset.to_netcdf(filename, **kwargs)
