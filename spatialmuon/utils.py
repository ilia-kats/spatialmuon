from codecs import decode
from os import PathLike

import pandas as pd
import h5py


def _read_hdf5_attribute(attrs: h5py.AttributeManager, name: str):
    """
    Read an HDF5 attribute and perform all necessary conversions.

    At the moment, this only implements conversions for string attributes, other types
    are passed through. String conversion is needed compatibility with other languages.
    For example Julia's HDF5.jl writes string attributes as fixed-size strings, which
    are read as bytes by h5py.
    """
    attr = attrs[name]
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
            cats = cat_dset.asstr()[()]
            ordered = _get_hdf5_attribute(cat_dset.attrs, "ordered", False)
            columns[c] = pd.Categorical.from_codes(col[yidx], categories, ordered=ordered)
        else:
            columns[c] = col[yidx]
    idx = grp[idx_key][yidx]
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
