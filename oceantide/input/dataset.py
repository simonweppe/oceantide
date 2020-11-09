"""Infer dataset format to use appropriate reader."""
from oceantide.core.otis import from_otis
from oceantide.tide import Tide


def read_dataset(dset):
    """Return formatted dataset with Tide accessor from unidentified dataset.

    Convenience function to define the Tide accessor for a dataset rather than a
        file by guessing the original file format based on variable names.

    Args:
        dset (Dataset): Tide constituents dataset from any supported file format.

    Returns:
        Formatted dataset in Oceantide convention with the Tide accessor.

    """
    vars_oceantide = {"con", "lat", "lon", "et", "ut", "vt"}
    vars_otis = {"con", "lat_z", "lon_z", "lat_u", "lon_u", "lat_v", "lon_v", "hRe", "hIm", "URe", "UIm", "VRe", "VIm"}


    vars_dset = set(dset.variables.keys()).union(dset.dims)
    if not vars_oceantide - vars_dset:
        func = from_oceantide
    elif not vars_otis - vars_dset:
        func = from_otis
    else:
        raise ValueError(
            f"Cannot identify appropriate reader from dataset variables: {vars_dset}"
        )
    return func(dset)


def from_oceantide(dset):
    return dset