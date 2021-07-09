from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Optional

import numpy as np
import h5py

from .backing import BackableObject
from ..utils import _read_hdf5_attribute

class UnknownDatatypeException(RuntimeError):
    def __init__(self, fovtype:str):
        self.datatype = fovtype

class FieldOfView(BackableObject):
    _datatypes = {}

    @classmethod
    def _load_datatypes(cls):
        if len(cls._datatypes) == 0:
            datatypes = entry_points()["spatialmuon.datatypes"]
            for ep in datatypes:
                klass = ep.load()
                cls._datatypes[klass._encoding()] = klass

    def __new__(cls, *, backing:Optional[h5py.Group]=None, **kwargs):
        if backing is not None:
            fovtype = _read_hdf5_attribute(backing.attrs, "encoding")
            cls._load_datatypes()
            if fovtype in cls._datatypes:
                klass = cls._datatypes[fovtype]
                return super(cls, klass).__new__(klass)
            else:
                raise UnknownDatatypeException(fovtype)
        else:
            return super().__new__(cls)

    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        images: Optional[dict] = None,
        feature_masks: Optional[dict] = None,
        image_masks: Optional[dict] = None,
        uns: Optional[dict] = None,
    ):
        super().__init__(backing)
        self._rotation = rotation
        self._translation = translation
        if rotation is None and self.isbacked and "rotation" in self.backing:
            self._rotation = self.backing["rotation"]
        if translation is None and self.isbacked and "translation" in self.backing:
            self._translation = self.backing["translation"]
        self.images = images if images is not None else {}
        self.feature_masks = feature_masks if feature_masks is not None else {}
        self.image_masks = image_masks if image_masks is not None else {}
        self.uns = uns if uns is not None else {}

    def _set_backing(self, value):
        super()._set_backing(value)
        pass # TODO

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    def rotation(self) -> np.ndarray:
        if self._rotation is not None:
            return self._rotation
        elif self.ndim is not None:
            return np.eye(self.ndim)
        else:
            return np.eye(3)

    @property
    def translation(self) -> np.ndarray:
        if self._translation is not None:
            return self._translation
        elif self.ndim is not None:
            return np.zeros(self.ndim)
        else:
            return np.zeros(3)

    def _write(self, grp):
        super()._write(grp)
        if self._rotation is not None:
            grp["rotation"] = self._rotation
        if self._translation is not None:
            grp["translation"] = self._translation
