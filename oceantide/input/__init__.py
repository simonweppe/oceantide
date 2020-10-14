import xarray as xr
from fsspec import get_mapper


def read_dataset(
    filename_or_fileglob,
    file_format,
):
    """Read constituents dataset in either netcdf or zarr format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - file_format (str): format of file to open, one of `netcdf` or `zarr`.

    Returns:
        - dset (Dataset): spectra dataset object read from file.

    """
    if file_format == "netcdf":
        dset = xr.open_mfdataset(filename_or_fileglob, combine="by_coords")
    elif file_format == "zarr":
        fsmap = get_mapper(filename_or_fileglob)
        dset = xr.open_zarr(fsmap, consolidated=True)
    else:
        raise ValueError("file_format must be one of ('netcdf', 'zarr')")
    return dset
