"""Read Otis custom merged netcdf file format."""
from oceantide.core.otis import from_otis
from oceantide.input import read_dataset
from oceantide.tide import Tide


def read_otis_merged(filename, file_format="netcdf"):
    """Read Otis custom merged netcdf file format.

    Args:
        filename (str): Name of Otis custom merged netcdf file to read.

    """
    dset = read_dataset(filename, file_format=file_format)
    return from_otis(dset)
