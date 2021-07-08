from typing import Optional
import warnings

from .backing import BackableObject
from .fieldofview import FieldOfView, UnknownDatatypeException
from ..utils import _read_hdf5_attribute, _get_hdf5_attribute

import h5py

class SpatialModality(BackableObject, dict):
    def __init__(self, fovs:Optional[dict] = None, scale: Optional[float]=None, coordinate_unit: Optional[str]=None, backing: h5py.Group=None):
        super().__init__(backing)
        if self.isbacked:
            for f, fov in self.backing.items():
                try:
                    super().__setitem__(f, FieldOfView(backing=fov))
                except UnknownDatatypeException as e:
                    warnings.warn(f"Unknown field of view type {e.datatype}")
            self.scale = _get_hdf5_attribute(self.backing.attrs, "scale", None)
            self.coordinate_unit = _get_hdf5_attribute(self.backing.attrs, "coordinate_unit", None)
        else:
            if fovs is not None:
                self.update(fovs)
            self.scale = scale
            self.coordinate_unit = coordinate_unit

    @property
    def scale(self):
        return self._scale if self._scale is not None and self._scale > 0 else 1

    @scale.setter
    def scale(self, newscale:Optional[float]):
        if newscale is not None and newscale <= 0:
            newscale = None
        self._scale = newscale

    @staticmethod
    def _encoding():
        return "spatialmodality"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, grp: Optional[h5py.Group]=None):
        if grp is not None:
            self._write_attributes(grp)
            for f, fov in self.items():
                fov.backing = grp.require_group(f)
            else:
                for fov in self.values():
                    fov.backing = None

    def _write_attributes_impl(self, grp: h5py.Group):
        if self._scale is not None:
            grp.attrs["scale"] = self._scale
        if self.coordinate_unit is not None:
            grp.attrs["coordinate_unit"] = self.coordinate_unit

    def _write(self, grp):
        for f, fov in self.items():
            fov.write(grp, f)

    def __setitem__(self, key: str, fov: FieldOfView):
        super().__setitem__(key, fov)
        if self.isbacked:
            fov.backing = self.backing.require_group(key)
