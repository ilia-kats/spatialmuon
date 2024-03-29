from typing import Literal, Optional, Union
from codecs import decode
import warnings
import colorama

import numpy as np
import pandas as pd
import h5py
from shapely.geometry import Polygon


def _read_hdf5_attribute(attrs: h5py.AttributeManager, name: str):
    """
    Read an HDF5 attribute and perform all necessary conversions.

    At the moment, this only implements conversions for string attributes, other types
    are passed through. String conversion is needed compatibility with other languages.
    For example Julia's HDF5.jl writes string attributes as fixed-size strings, which
    are read as bytes by h5py.
    """
    try:
        attr = attrs[name]
    except KeyError as e:
        print("debug")
        raise e
    attr_id = attrs.get_id(name)
    dtype = h5py.check_string_dtype(attr_id.dtype)
    if dtype is None:
        return attr
    else:
        if dtype.length is None:  # variable-length string, no problem
            return attr
        elif len(attr_id.shape) == 0:  # Python bytestring
            return attr.decode("utf-8")
        else:  # NumPy array
            return [decode(s, "utf-8") for s in attr]


def _get_hdf5_attribute(attrs: h5py.AttributeManager, name: str, default=None):
    if name in attrs:
        return _read_hdf5_attribute(attrs, name)
    else:
        return default


def preprocess_3d_polygon_mask(
    mask: Polygon, coords: np.ndarray, method: Literal["project", "discard"] = "discard"
):
    if method == "discard":
        bounds = list(mask.bounds)
        if mask.has_z:
            bounds[2] = float("-Inf")
            bounds[-1] = float("Inf")  # GeoPandas ignores the 3rd dimension
    elif method == "project":
        if mask.has_z:
            warnings.warn(
                "Method is `project` but masks has 3 dimensions. Assuming that masks is in-plane with the data and skipping projection."
                # noqa: E501
            )
        else:
            mean = coords.mean(axis=0)
            coords = coords - mean[:, np.newaxis]
            cov = coords.T @ coords
            projmat = np.linalg.eigh[cov][1][:, :2]
            mask = Polygon(np.asarray(mask.exterior.coords) @ projmat.T)
            bounds = mask.bounds
    return bounds


def read_dataframe_subset(grp: h5py.Group, yidx):
    enc = _read_hdf5_attribute(grp.attrs, "encoding-type")
    if enc != "dataframe":
        raise ValueError("not a data frame")
    cols = list(_read_hdf5_attribute(grp.attrs, "column-order"))
    idx_key = _read_hdf5_attribute(grp.attrs, "_index")
    columns = {}
    for c in cols:
        col = grp[c]
        categories = _get_hdf5_attribute(col.attrs, "categories")
        if categories is not None:
            cat_dset = grp[categories]
            ordered = _get_hdf5_attribute(cat_dset.attrs, "ordered", False)
            columns[c] = pd.Categorical.from_codes(col[yidx], categories, ordered=ordered)
        else:
            columns[c] = col[yidx] if not h5py.check_string_dtype(col.dtype) else col.asstr()[yidx]
    idx = (
        grp[idx_key][yidx]
        if not h5py.check_string_dtype(grp[idx_key].dtype)
        else grp[idx_key].asstr()[yidx]
    )
    df = pd.DataFrame(columns, index=idx, columns=cols)
    if idx_key != "_index":
        df.index.name = idx_key
    return df


class UnknownEncodingException(RuntimeError):
    def __init__(self, encoding: str):
        self.encoding = encoding


def is_h5smu(filename):
    with open(filename, "rb") as f:
        if f.read(13) != b"SpatialMuData":
            return False
    return h5py.is_hdf5(filename)


def angle_between(v1: np.array, v2: np.array, output: str = "degree"):
    """Returns the signed angle between 'v1' and 'v2' in degree or radians"""

    rotation = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])

    if output == "degree":
        return rotation * 180 / np.pi

    elif output == "radians":

        return rotation


def old_school_debugging(debug: bool):
    def print_red(s: str):
        if debug:
            print(f"{colorama.Fore.YELLOW}{s}{colorama.Fore.RESET}")

    return print_red


# either a color or a list of colors
ColorsType = Optional[
    Union[
        str,
        list[str],
        np.ndarray,
        list[np.ndarray],
        list[int],
        list[float],
        list[list[int]],
        list[list[float]],
    ]
]

ColorType = Union[str, np.ndarray, list[int], list[float]]
