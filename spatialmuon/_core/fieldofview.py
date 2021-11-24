from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Optional, Union, Literal
import warnings

import numpy as np
import h5py
from shapely.geometry import Polygon
from trimesh import Trimesh
from anndata._io.utils import read_attribute, write_attribute
from anndata.utils import make_index_unique
import pandas as pd

from .backing import BackableObject, BackedDictProxy
# from .image import Image
from .masks import Masks
from ..utils import _read_hdf5_attribute, _get_hdf5_attribute, UnknownEncodingException


class FieldOfView(BackableObject):
    _datatypes = {}

    @classmethod
    def _load_datatypes(cls):
        if len(cls._datatypes) == 0:
            datatypes = entry_points()["spatialmuon.datatypes"]
            for ep in datatypes:
                klass = ep.load()
                cls._datatypes[klass._encodingtype()] = klass

    def __new__(cls, *, backing: Optional[h5py.Group] = None, **kwargs):
        if backing is not None:
            fovtype = _read_hdf5_attribute(backing.attrs, "encoding-type")
            cls._load_datatypes()
            if fovtype in cls._datatypes:
                klass = cls._datatypes[fovtype]
                return super(cls, klass).__new__(klass)
            else:
                raise UnknownEncodingException(fovtype)
        else:
            return super().__new__(cls)

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
        scale: Optional[float] = None,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        var: Optional[pd.DataFrame] = None,
        masks: Optional[dict] = None,
        uns: Optional[dict] = None,
    ):
        super().__init__(backing)
        self._scale = scale
        self._rotation = rotation
        self._translation = translation
        if scale is None and self.isbacked:
            self._scale = _get_hdf5_attribute(self.backing.attrs, "scale")
        if rotation is None and self.isbacked:
            self._rotation = _get_hdf5_attribute(self.backing.attrs, "rotation")
        if translation is None and self.isbacked:
            self._translation = _get_hdf5_attribute(self.backing.attrs, "translation")

        self.masks = BackedDictProxy(self, key="masks")
        if self.isbacked and "masks" in self.backing:
            for key, mask in self.backing["masks"].items():
                self.masks[key] = Masks(backing=mask)

        if self.isbacked and "var" in self.backing:
            self._var = read_attribute(backing["var"])
        elif var is not None:
            # if var.shape[0] != self._X.shape[1]:
            #     raise ValueError("X shape is inconsistent with var")
            # else:
            self._var = var
            if not self._var.index.is_unique:
                warnings.warn(
                    "Gene names are not unique. This will negatively affect indexing/subsetting. Making unique names..."
                )
                self._var.index = make_index_unique(self._var.index)
        else:
            self._var = pd.DataFrame()
            # self._var = pd.DataFrame(index=range(self._X.shape[1]))
        if self.isbacked and "uns" in self.backing:
            self.uns = read_attribute(self.backing["uns"])
        elif not self.isbacked and uns is not None:
            self.uns = uns
        else:
            self.uns = {}

        # we don't want to validate stuff coming from HDF5, this may break I/O
        # but mostly we can't validate for a half-initalized object
        self.masks.validatefun = self.__validate_mask

        # init with validation
        # this requires that subclasses call this constructor at the end of their init method
        # because validation requires information from subclasses, e.g. ndim
        if not self.isbacked:
            if masks is not None:
                self.masks.update(masks)

    def _set_backing(self, obj):
        super()._set_backing(obj)
        for mask in self.masks:
            mask.set_backing(obj)

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    def scale(self) -> float:
        return self._scale if self._scale is not None and self._scale > 0 else 1

    @scale.setter
    def scale(self, newscale: Optional[float]):
        if newscale is not None and newscale <= 0:
            newscale = None
        self._scale = newscale

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

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def n_var(self) -> int:
        return self._var.shape[0]

    def __getitem__(self, index):
        polygon_method = "discard"
        if not isinstance(index, tuple):
            if isinstance(index, str) or isinstance(index, list):
                genes = index
                mask = None
            else:
                mask = index
                genes = None
        else:
            mask = index[0]
            genes = index[1]
            if len(index) > 2:
                polygon_method = index[2]
        if mask == slice(None):
            mask = None
        if genes == slice(None):
            genes = None

        return self._getitem(mask, genes, polygon_method)

    @abstractmethod
    def _getitem(
        self,
        mask: Optional[Union[Polygon, Trimesh]] = None,
        genes: Optional[Union[str, list[str]]] = None,
        polygon_method: Literal["discard", "project"] = "discard",
    ):
        pass

    def _write_attributes_impl(self, obj: h5py.Group):
        super()._write_attributes_impl(obj)
        if self._rotation is not None:
            obj.attrs["rotation"] = self._rotation
        if self._translation is not None:
            obj.attrs["translation"] = self._translation

    def _write(self, obj: h5py.Group):
        for maskname, mask in self.masks.items():
            mask.write(obj, f"masks/{maskname}")

        write_attribute(
            obj, "var", self._var, dataset_kwargs={"compression": "gzip", "compression_opts": 9}
        )
        write_attribute(
            obj, "uns", self.uns, dataset_kwargs={"compression": "gzip", "compression_opts": 9}
        )
