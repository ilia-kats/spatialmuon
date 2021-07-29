from collections.abc import MutableMapping
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import h5py
from shapely.geometry import Polygon

from ..utils import _read_hdf5_attribute, UnknownEncodingException
from .backing import BackableObject

class Mask(BackableObject):
    def __new__(cls, *, backing: Optional[h5py.Group] = None, **kwargs):
        if backing is not None:
            masktype = _read_hdf5_attribute(backing.attrs, "encoding")
            if masktype == "mask-polygon":
                return super(cls, PolygonMask).__new__(PolygonMask)
            elif masktype == "mask-raster":
                pass # TODO
            else:
                raise UnknownEncodingException(masktype)
        else:
            return super().__new__(cls)

    @property
    @abstractmethod
    def ndim(self):
        pass

class PolygonMask(Mask, MutableMapping):
    def __init__(self, backing:Optional[h5py.Group]=None, masks: Optional[dict[Union[np.ndarray, Polygon]]]=None):
        super().__init__(backing)

        self._data = {}
        self._ndim = None
        if masks is not None:
            if self.isbacked and len(self.backing) > 0:
                raise ValueError("trying to set masks on a non-empty backing store")
            for key, mask in masks:
                if isinstance(mask, Polygon):
                    ndim = 3 if mask.has_z else 2
                else:
                    ndim = mask.ndim
                if self.ndim is None:
                    self.ndim = ndim
                if ndim != self.ndim:
                    raise ValueError("all masks must have the same dimensionality")

                self[key] = mask

    @property
    def ndim(self):
        return self._ndim

    def __getitem__(self, key):
        if self.isbacked:
            return Polygon(self.backing[key][:])
        else:
            return self._data[key]

    def __setitem__(self, key: str, value: Union[np.ndarray, Polygon]):
        if self.isbacked:
            if isinstance(value, Polygon):
                value = np.asarray(value.exterior.coords)
            else:
                value = np.asarray(value)
            if self.ndim is not None and self.ndim != value.ndim:
                raise ValueError(f"value must have dimensionality {self._ndim}, but has {value.ndim}")
            self.backing.create_dataset(key, data=value, compression="gzip", compression_opts=9)
        else:
            if not isinstance(value, Polygon):
                ndim = value.ndim
                value = Polygon(value)
            else:
                ndim = 3 if value.has_z else 2
            if self.ndim is not None and self.ndim != ndim:
                raise ValueError(f"value must have dimensionality {self._ndim}, but has {ndim}")
            self._data[key] = value

    def __delitem__(self, key: str):
        if self.isbacked:
            del self.backing[key]
        else:
            del self._data[key]

    def __len__(self):
        if self.isbacked:
            return len(self.backing)
        else:
            return len(self._data)

    def __contains__(self, item):
        if self.isbacked:
            return item in self.backing
        else:
            return item in self._data

    def __iter__(self):
        if self.isbacked:
            return iter(self.backing)
        else:
            return iter(self._data)

    def items(self):
        raise NotImplementedError()

    def values(self):
        raise NotImplementedError()

    @staticmethod
    def _encoding():
        return "mask-polygon"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value):
        if value is None and self.backed:
            for k, v in self.backing.items():
                self._data[k] = Polygon(v[:])
        elif value is not None:
            self._write(value)

    def _write(self, obj):
        for k, v in self._data.items():
            obj.create_dataset(k, data=v.exterior.coords, compression="gzip", compression_opts=9)

    def _write_attributes_impl(self, obj):
        pass
