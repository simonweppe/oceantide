"""Read Oceantide file format."""
import warnings
import xarray as xr

from oceantide.tide import Tide


def read_oceantide(filename=None, engine="zarr", chunks={}, **kwargs):
    """Read Oceantide file format.

    Args:
        - filename (str): Name of Oceantide dataset to read.
        - engine (str): Engine to pass to xr.open_dataset.
        - chunks (dict): Chunk sizes along each dimensions to pass to xr.open_dataset.
        - kwargs: Extra kwargs to pass to xr.open_dataset.

    Returns:
        - Formatted dataset with the tide accessor.

    """
    dset = xr.open_dataset(filename, engine=engine, chunks=chunks, **kwargs)

    # For backward compatibility
    if "et" in dset.data_vars and "h" not in dset.data_vars:
        warnings.warn(
            "Oceantide naming convention has been changed, only datasets with "
            "variables ['dep','h','u','v'] will be supported in the future.",
            category=FutureWarning,
        )
        dset = dset.rename({"et": "h", "ut": "u", "vt": "v", "depth": "dep"})

    return dset
