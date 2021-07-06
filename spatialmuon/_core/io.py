from os import PathLike
from typing import Union
from codecs import decode
import warnings

import h5py

from .spatialmudata import SpatialMuData
from .spatialmodality import SpatialModality
from .fieldofview import FieldOfView, UnknownDatatypeException
from ..utils import _read_hdf5_attribute, _get_hdf5_attribute

def _read_modality(grp: h5py.Group):
    attrs = grp.attrs
    scale = _get_hdf5_attribute(attrs, "scale", None)
    unit = _get_hdf5_attribute(attrs, "coordinate_unit", None)
    fovs = {}
    for f, fov in grp.items():
        try:
            fovs[f] = FieldOfView(grp=fov)
        except UnknownDatatypeException as e:
            warnings.warn(f"Unknown field of view type {e.datatype}")
    fovs = {f: FieldOfView(grp=fov) for f, fov in grp.items()}
    return SpatialModality(fovs, scale, unit)

def read_h5smu(filename: PathLike, backed: Union[str, bool, None]=True):
    assert backed in [None, True, False, "r", "r+"], "Argument `backed` should be boolean, r, r+, or None"

    from anndata._io.utils import read_attribute

    with open(filename, "rb") as f:
        if f.read(13) != b"SpatialMuData":
            raise RuntimeError(f"{filename} is not a SpatialMuData file")

    with h5py.File(filename, "r") as f:
        mods = {}
        for m, mod in f["mod"].items():
            mods[m] = _read_modality(mod)
        return SpatialMuData(mods)

def write_h5smu(filename: PathLike, smudata: SpatialMuData):
    from anndata._io.utils import write_attribute
    from .. import __version__, __spatialmudataversion__

    with h5py.File(filename, "w", userblock_size=512, libver="latest") as f:
        f.attrs["encoder"] = "spatialmuon"
        f.attrs["encoder-version"] = __version__
        f.attrs["encoding"] = "SpatialMuData"
        f.attrs["encoding-version"] = __spatialmudataversion__

        mods = f.create_group("mod")
        for m, mod in smudata:
            write_spatialmodality(mods, m, mod)

def write_spatialmodality(parent: h5py.Group, key:str, mod: SpatialModality):
    pass
