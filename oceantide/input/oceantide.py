"""Read Oceantide file format."""
import xarray as xr
from fsspec import get_mapper

from oceantide.tide import Tide


def read_oceantide(filename=None, zarr_kwargs={"consolidated": True}):
    """Read Oceantide file format.

    Args:
        filename (str): Name of `Model_*` metadata file specifying other files to read.
            Only zarr is supported as netcdf does not deal well with complex array.
        zarr_kwargs (dict): Keywargs to pass to xr.open_zarr.

    Returns:
        Formatted dataset with the tide accessor.

    Tide constituents datasets read with any reader function in oceantide returns
        datasets in the oceantide format. The dataset should look like:

    <xarray.Dataset>
    Dimensions:  (con: 9, lat: 251, lon: 481)
    Coordinates:
    * con      (con) <U2 'M2' 'S2' 'N2' 'K2' 'K1' 'O1' 'P1' 'Q1' 'M4'
    * lat      (lat) float64 48.47 48.53 48.58 48.63 ... 60.87 60.92 60.97 61.03
    * lon      (lon) float64 -11.02 -10.97 -10.92 -10.87 ... 12.92 12.97 13.02
    Data variables:
        depth    (lat, lon) float32 dask.array<chunksize=(126, 241), meta=np.ndarray>
        et       (con, lat, lon) complex128 dask.array<chunksize=(3, 126, 241), meta=np.ndarray>
        ut       (con, lat, lon) complex128 dask.array<chunksize=(3, 126, 241), meta=np.ndarray>
        vt       (con, lat, lon) complex128 dask.array<chunksize=(3, 126, 241), meta=np.ndarray>

    """
    fsmap = get_mapper(filename)
    return xr.open_zarr(fsmap, consolidated=True)
