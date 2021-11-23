from typing import Optional
import warnings

from .backing import BackableObject, BackedDictProxy
from .fieldofview import FieldOfView, UnknownEncodingException
from ..utils import _read_hdf5_attribute, _get_hdf5_attribute

import h5py


class SpatialModality(BackableObject, BackedDictProxy):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        fovs: Optional[dict] = None,
        coordinate_unit: Optional[str] = None,
    ):
        super().__init__(backing)
        if self.isbacked:
            for f, fov in self.backing.items():
                try:
                    self[f] = FieldOfView(backing=fov)
                except UnknownEncodingException as e:
                    warnings.warn(f"Unknown field of view type {e.encoding}")
            self.coordinate_unit = _get_hdf5_attribute(self.backing.attrs, "coordinate_unit", None)
        else:
            if fovs is not None:
                self.update(fovs)
            self.coordinate_unit = coordinate_unit

    @staticmethod
    def _encodingtype():
        return "spatialmodality"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, grp: Optional[h5py.Group] = None):
        super()._set_backing(grp)
        if grp is not None:
            self._write_attributes(grp)
            for f, fov in self.items():
                fov.set_backing(grp, f)
        else:
            for fov in self.values():
                fov.set_backing(None)

    def _write_attributes_impl(self, grp: h5py.Group):
        if self.coordinate_unit is not None:
            grp.attrs["coordinate_unit"] = self.coordinate_unit

    def _write(self, grp):
        for f, fov in self.items():
            fov.write(grp, f)
