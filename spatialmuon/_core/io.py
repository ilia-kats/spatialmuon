from os import PathLike
from typing import Union, Literal
from codecs import decode
import warnings

import h5py

from .spatialmudata import SpatialMuData
from .spatialmodality import SpatialModality
from .fieldofview import FieldOfView, UnknownEncodingException
from ..utils import _read_hdf5_attribute, _get_hdf5_attribute, is_h5smu


def _read_modality(grp: h5py.Group):
    raise ValueError('legacy code, not tested')
    attrs = grp.attrs
    scale = _get_hdf5_attribute(attrs, "scale", None)
    unit = _get_hdf5_attribute(attrs, "coordinate_unit", None)
    fovs = {}
    for f, fov in grp.items():
        try:
            fovs[f] = FieldOfView(backing=fov)
        except UnknownEncodingException as e:
            warnings.warn(f"Unknown field of view type {e.datatype}")
    fovs = {f: FieldOfView(backing=fov) for f, fov in grp.items()}
    return SpatialModality(fovs, scale, unit)


def read_h5smu(filename: PathLike, backed: Union[Literal["r", "r+"], bool, None] = True):
    assert backed in [
        None,
        True,
        False,
        "r",
        "r+",
    ], "Argument `backed` should be boolean, r, r+, or None"

    from anndata._io.utils import read_attribute

    if backed is True:
        mode = "r+"
    elif backed is not False and backed is not None:
        mode = backed
        backed = True
    else:
        mode = "r"

    if not is_h5smu(filename):
        raise RuntimeError(f"{filename} is not a SpatialMuData file")

    smudata = SpatialMuData(backing=filename, backingmode=mode)
    if not backed:
        smudata.set_backing()
    return smudata


def write_h5smu(filename: PathLike, smudata: SpatialMuData):
    from .. import __version__, __spatialmudataversion__

    with h5py.File(filename, "w", userblock_size=512, libver="latest") as f:
        smudata.write(f)

    with open(filename, "br+") as outfile:
        fname = "SpatialMuData (format-version={};".format(__spatialmudataversion__)
        fname += "creator=spatialmuon;"
        fname += "creator-version={})".format(__version__)
        outfile.write(fname.encode("utf-8"))
