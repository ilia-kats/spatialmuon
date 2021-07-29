from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Optional

import numpy as np
import h5py

from .backing import BackableObject, BackedDictProxy
from .image import Image
from ..utils import _read_hdf5_attribute


class UnknownDatatypeException(RuntimeError):
    def __init__(self, fovtype: str):
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

    def __new__(cls, *, backing: Optional[h5py.Group] = None, **kwargs):
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

    @staticmethod
    def __validate_image(fov, key, img):
        if img.rotation() is not None and img.rotation().shape != (fov.ndim, fov.ndim):
            return f"rotation matrix must have shape ({fov.ndim}, {fov.ndim})"
        if img.translation() is not None and img.translation().shape != (fov.ndim,):
            return f"translation vector must have shape ({fov.ndim},)"
        return None

    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        *,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        images: Optional[dict[Image]] = None,
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
        self.images = BackedDictProxy(self, key="images", items=images)
        if self.isbacked:
            for key, img in self.backing["images"].items():
                self.images[key] = Image(img)
        self.feature_masks = feature_masks if feature_masks is not None else {}
        self.image_masks = image_masks if image_masks is not None else {}
        self.uns = uns if uns is not None else {}

        # we don't want to validate stuff coming from HDF5, this may break I/O
        # but mostly we can't validate for a half-initalized object
        self.images.validatefun = self.__validate_image

    def _set_backing(self, obj):
        super()._set_backing(obj)
        for img in self.images:
            img.set_backing(obj)

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

    def _write_attributes_impl(self, obj: h5py.Group):
        super()._write_attributes_impl(obj)
        if self._rotation is not None:
            obj["rotation"] = self._rotation
        if self._translation is not None:
            obj["translation"] = self._translation

    def _write(self, obj: h5py.Group):
        for imname, img in self.images.items():
            img.write(obj, f"images/{imname}")
