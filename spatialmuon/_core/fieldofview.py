from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Optional

import numpy as np
import h5py

from .backing import BackableObject, BackedDictProxy
from .image import Image
from .mask import Mask
from ..utils import _read_hdf5_attribute, UnknownEncodingException


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
                raise UnknownEncodingException(fovtype)
        else:
            return super().__new__(cls)

    @staticmethod
    def __validate_image(fov, key, img):
        if img.rotation() is not None and img.rotation().shape != (fov.ndim, fov.ndim):
            return f"rotation matrix must have shape ({fov.ndim}, {fov.ndim})"
        if img.translation() is not None and img.translation().shape != (fov.ndim,):
            return f"translation vector must have shape ({fov.ndim},)"
        return None

    @staticmethod
    def __validate_mask(fov, key, mask):
        if mask.ndim != None and mask.ndim != fov.ndim:
            return f"mask with {mask.ndim} dimensions is being added to field of view with {fov.ndim} dimensions"
        mask.parentdataset = fov
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
        self.images = BackedDictProxy(self, key="images")
        if self.isbacked and "images" in self.backing:
            for key, img in self.backing["images"].items():
                self.images[key] = Image(img)

        self.feature_masks = BackedDictProxy(self, key="feature_masks")
        if self.isbacked and "feature_masks" in self.backing:
            for key, mask in self.backing["feature_masks"].items():
                self.feature_masks[key] = Mask(backing=mask)

        self.image_masks = BackedDictProxy(self, key="image_masks")
        if self.isbacked and "image_masks" in self.backing:
            for key, mask in self.backing["image_masks"].items():
                self.image_masks[key] = Mask(backing=mask)
        self.uns = uns if uns is not None else {}

        # we don't want to validate stuff coming from HDF5, this may break I/O
        # but mostly we can't validate for a half-initalized object
        self.images.validatefun = self.__validate_image
        self.feature_masks.validatefun = self.__validate_mask
        self.image_masks.validatefun = self.__validate_mask

        # init with validation
        # this requires that subclasses call this constructor at the end of their init method
        # because validation requires information from subclasses, e.g. ndim
        if not self.isbacked:
            if images is not None:
                self.images.update(images)
            if feature_masks is not None:
                self.feature_masks.update(feature_masks)
            if image_masks is not None:
                self.image_masks.update(image_masks)

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
