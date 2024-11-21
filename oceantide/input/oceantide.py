"""Read oceantide file format."""
import xarray as xr

from oceantide.tide import Tide
from oceantide.core.utils import set_attributes


def read_oceantide(
    filename: str = None, engine: str = None, chunks: dict = {}, suffix  = ['_real','_imag'],**kwargs
) -> xr.Dataset:
    """Read oceantide file format.

    The oceantide format has complex constituents variables split into real and imag
    variables allowing for better packing and making it easier to support netcdf.

    parameters
    ----------
    filename (str)
        Name of Oceantide dataset to read.
    engine (str)
        Engine to pass to xr.open_dataset, guessed by xarray based on filename
        extension if not specified.
    chunks (dict)
        Chunk sizes along each dimensions to pass to xr.open_dataset, by default use
        chunking specified on disk.
    suffix (list of string)
        Suffix used for the real and imaginary variables. Default : ['_real','_imag']
    kwargs
        Extra kwargs to pass to xr.open_dataset.

    Returns
    -------
    dset (xr.Dataset)
        Formatted dataset with complex variables and the tide accessor.

    """
    dset = xr.open_dataset(filename, engine=engine, chunks=chunks, **kwargs)
    dsout = dset[["dep"]]
    for v in ["h", "u", "v"]:
        dsout[v] = dset[f"{v}{suffix[0]}"] + 1j * dset[f"{v}{suffix[1]}"]
        # dsout[v] = dset[f"{v}_real"] + 1j * dset[f"{v}_imag"]
    set_attributes(dsout, "dataset")
    dsout["con"] = dsout.con.astype("U4")
    return dsout
