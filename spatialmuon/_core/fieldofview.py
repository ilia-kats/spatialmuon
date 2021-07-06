from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Optional

import numpy as np
import h5py

from ..utils import _read_hdf5_attribute

class UnknownDatatypeException(RuntimeError):
    def __init__(fovtype:str):
        self.datatype = fovtype

class FieldOfView(ABC):
    _datatypes = {}

    @classmethod
    def _load_datatypes(cls):
        if len(cls._datatypes) == 0:
            datatypes = entry_points()["spatialmuon.datatypes"]
            for ep in datatypes:
                klass = ep.load()
                cls._datatypes[klass.datatype()] = klass

    def __new__(cls, *, grp:Optional[h5py.Group]=None, **kwargs):
        if grp is not None:
            fovtype = _read_hdf5_attribute(grp.attrs, "encoding")
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
        *,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        images: Optional[dict] = None,
        feature_masks: Optional[dict] = None,
        image_masks: Optional[dict] = None,
        uns: Optional[dict] = None,
        grp: Optional[h5py.Group] = None,
    ):
        self._grp = grp
        self._index = None
        self.rotation = rotation
        self.translation = translation
        self.images = images if images is not None else {}
        self.feature_masks = feature_masks if feature_masks is not None else {}
        self.image_masks = image_masks if image_masks is not None else {}
        self.uns = uns if uns is not None else {}

    @staticmethod
    @abstractmethod
    def datatype() -> str:
        pass