"""Read Oceantide file format."""
import warnings
import xarray as xr

from oceantide.tide import Tide


def read_oceantide(filename=None, engine="zarr", backend_kwargs={"consolidated": True}, **kwargs):
    """Read Oceantide file format.

    Args:
        - filename (str): Name of Oceantide dataset to read.
        - engine (str): Engine to pass to xr.open_dataset.
        - backend_kwargs (dict): Keywargs to pass to xr.open_dataset.
        - kwargs: Extra kwargs to pass to xr.open_dataset.

    Returns:
        - Formatted dataset with the tide accessor.

    Tide constituents datasets read with any reader function in oceantide returns
        datasets in the oceantide format. The dataset should look like:

    <xarray.Dataset>
    Dimensions:  (con: 9, lat: 251, lon: 481)
    Coordinates:
    * con      (con) <U2 'M2' 'S2' 'N2' 'K2' 'K1' 'O1' 'P1' 'Q1' 'M4'
    * lat      (lat) float64 48.47 48.53 48.58 48.63 ... 60.87 60.92 60.97 61.03
    * lon      (lon) float64 -11.02 -10.97 -10.92 -10.87 ... 12.92 12.97 13.02
    Data variables:
        dep     (lat, lon) float32 dask.array<chunksize=(126, 241), meta=np.ndarray>
        h       (con, lat, lon) complex128 dask.array<chunksize=(3, 126, 241), meta=np.ndarray>
        u       (con, lat, lon) complex128 dask.array<chunksize=(3, 126, 241), meta=np.ndarray>
        v       (con, lat, lon) complex128 dask.array<chunksize=(3, 126, 241), meta=np.ndarray>

    """
    dset = xr.open_dataset(
        filename, engine=engine, backend_kwargs=backend_kwargs, **kwargs
    )

    # For backward compatibility
    if "et" in dset.data_vars and "h" not in dset.data_vars:
        warnings.warn(
            "Oceantide naming convention has been changed, only datasets with "
            "variables ['dep','h','u','v'] will be supported in the future.",
            category=FutureWarning,
        )
        dset = dset.rename({"et": "h", "ut": "u", "vt": "v", "depth": "dep"})

    return dset
