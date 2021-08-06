from collections.abc import MutableMapping
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import h5py
from shapely.geometry import Polygon
from trimesh import Trimesh
from skimage.measure import find_contours, marching_cubes

from ..utils import _read_hdf5_attribute, UnknownEncodingException
from .backing import BackableObject


class Mask(BackableObject):
    def __new__(cls, *, backing: Optional[h5py.Group] = None, **kwargs):
        if backing is not None:
            masktype = _read_hdf5_attribute(backing.attrs, "encoding")
            if masktype == "mask-polygon":
                return super(cls, PolygonMask).__new__(PolygonMask)
            elif masktype == "mask-raster":
                return super(cls, RasterMask).__new__(RasterMask)
            elif masktype == "mask-mesh":
                return super(cls, MeshMask).__new__(MeshMask)
            else:
                raise UnknownEncodingException(masktype)
        else:
            return super().__new__(cls)

    def __init__(self, backing: Optional[Union[h5py.Group, h5py.Dataset]] = None):
        super().__init__(backing)
        self._parentdataset = None

    @property
    def parentdataset(self):
        return self._parentdataset

    @parentdataset.setter
    def parentdataset(self, dset: "FieldOfView"):
        self._parentdataset = dset

    @property
    @abstractmethod
    def ndim(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass


class PolygonMask(Mask, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks: Optional[dict[str, Union[np.ndarray, Polygon]]] = None,
    ):
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

    def __getitem__(self, key) -> Polygon:
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
                raise ValueError(
                    f"value must have dimensionality {self.ndim}, but has {value.ndim}"
                )
            self.backing.create_dataset(key, data=value, compression="gzip", compression_opts=9)
        else:
            if not isinstance(value, Polygon):
                ndim = value.ndim
                value = Polygon(value)
            else:
                ndim = 3 if value.has_z else 2
            if self.ndim is not None and self.ndim != ndim:
                raise ValueError(f"value must have dimensionality {self.ndim}, but has {ndim}")
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

    def _set_backing(self, value: h5py.Group):
        if value is None and self.backed:
            for k, v in self.backing.items():
                self._data[k] = Polygon(v[:])
        elif value is not None:
            self._write(value)
            self._data.clear()

    def _write(self, obj: h5py.Group):
        for k, v in self._data.items():
            obj.create_dataset(k, data=v.exterior.coords, compression="gzip", compression_opts=9)


class MeshMask(Mask, MutableMapping):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        masks: Optional[dict[str, Union[Trimesh, tuple[np.ndarray, np.ndarray]]]] = None,
    ):
        super.__init__(backing)
        self._data = {}
        if masks is not None:
            if self.isbacked and len(self.backing) > 0:
                raise ValueError("trying to set masks on a non-empty backing store")
            self.update(masks)

    @property
    def ndim(self):
        return 3

    def __getitem__(self, key) -> Trimesh:
        if self.isbacked:
            return Trimesh(
                vertices=self.backing[key]["vertices"][()], faces=self.backing[key]["faces"][()]
            ).fix_normals()
        else:
            return self._data[key]

    def __setitem__(self, key: str, value: Union[Trimesh, tuple[np.ndarray, np.ndarray]]):
        if self.isbacked:
            if isinstance(value, Trimesh):
                vertices = value.vertices
                faces = value.faces
            else:
                vertices, faces = value
                if vertices.shape[1] != 3:
                    raise ValueError(
                        f"masks must be 3-dimensional, but mask {key} has dimensionality {vertices.shape[1]}"
                    )
                if faces.shape[1] != 3 and faces.shape[1] != 4:
                    raise ValueError(
                        f"faces must reference 3 or 4 vertices, but faces for mask {key} reference {faces.shape[1]} vertices"
                    )
                maxface = faces.max()
                if maxface > vertices.shape[0] - 1:
                    raise ValueError(
                        f"mask {key} has {vertices.shape[0]} vertices, but faces reference up to {maxface} vertices"
                    )

            self.backing.create_dataset(
                f"{key}/vertices", data=vertices, compression="gzip", compression_opts=9
            )
            self.backing.create_dataset(
                f"{key}/faces", data=faces, compression="gzip", compression_opts=9
            )

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
            return item in self._backing
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
        return "mask-mesh"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value: h5py.Group):
        if value is None and self.backed:
            for k in self.keys():
                self._data[k] = self[k]
        elif value is not None:
            self._write(value)
            self._data.clear()

    def _write(self, obj: h5py.Group):
        for k, v in self._data.items():
            self.backing.create_dataset(
                f"{k}/vertices", data=v.vertices, compression="gzip", compression_opts=9
            )
            self.backing.create_dataset(
                f"{k}/faces", data=v.faces, compression="gzip", compression_opts=9
            )


class RasterMask(Mask):
    def __init__(
        self,
        backing: Optional[h5py.Dataset] = None,
        mask: Optional[np.ndarray] = None,
        shape: Optional[Union[tuple[int, int], tuple[int, int, int]]] = None,
        dtype: Optional[type] = None,
    ):
        super().__init__(backing)
        self._mask = None

        if mask is not None:
            if self.isbacked and self.backing.size > 0:
                raise ValueError("attempting to set masks on a non-empty backing store")
            if mask.ndim < 2 or mask.ndim > 3:
                raise ValueError("mask must have 2 or 3 dimensions")
            if not np.issubdtype(mask.dtype, np.unsignedinteger):
                raise ValueError("mask must have an unsigned integer dtype")
            self._mask = mask
        elif not self.isbacked:
            if shape is None or dtype is None:
                raise ValueError("if mask is None shape and dtype must be given")
            if len(shape) < 2 or len(shape) > 3:
                raise ValueError("shape must have 2 or 3 dimensions")
            if not np.issubdtype(dtype, np.unsignedinteger):
                raise ValueError("dtype must be an unsigned integer type")
            self._mask = np.zeros(shape=shape, dtype=dtype)

    @property
    def ndim(self):
        if self.isbacked:
            return self.backing.ndim
        else:
            return self._mask.ndim

    @property
    def shape(self):
        if self.isbacked:
            return self.backing.shape
        else:
            return self._mask.shape

    @property
    def dtype(self):
        if self.isbacked:
            return self.backing.dtype
        else:
            return self._mask.dtype

    @property
    def data(self) -> Union[np.ndarray, h5py.Dataset]:
        if self.isbacked:
            return self.backing
        else:
            return self._mask

    def __getitem__(self, key):
        if not np.issubdtype(type(key), np.integer):
            raise TypeError("key must be an integer")
        if key >= 0:
            coords = np.where(self.data[()] == key + 1)
            if any(c.size == 0 for c in coords):
                raise KeyError(key)
            boundaries = []
            for i, c in enumerate(coords):
                min, max = np.min(c), np.max(c) + 1
                if min > 0:
                    min -= 1
                if max <= self.data.shape[i]:
                    max += 1
                boundaries.append(slice(min, max))
            cmask = self.data[tuple(boundaries)]
            cmask[cmask != key + 1] = 0
            if self.ndim == 2:
                contour = find_contours(cmask, fully_connected="high")
                for c in contour:
                    for d in range(self.ndim):
                        c[:, d] += boundaries[d].start
                return Polygon(contour[0][:, ::-1]) if len(contour) == 1 else [Polygon(c[:, ::-1]) for c in contour]
                # TODO: scale by px_size and px_distance from parentdataset
            else:
                vertices, faces, normals, _ = marching_cubes(cmask,  allow_degenerate=False)
                for d in range(self.ndim):
                    vertices[:, d] += boundaries[d].start
                    faces[:, d] += boundaries[d].start
                return Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
                # TODO: scale by px_size and px_distance from parentdataset

    @staticmethod
    def _encoding():
        return "mask-raster"

    @staticmethod
    def _encodingversion():
        return "0.1.0"

    def _set_backing(self, value: h5py.Dataset):
        if value is None and self.backed:
            self._mask = self.backing[:]
        elif value is not None:
            self._write(value)
            self._mask = None

    def _write(self, obj: h5py.Dataset):
        obj[()] = self._mask

    def _writeable_object(self, parent: h5py.Group, key: str) -> h5py.Dataset:
        if key in parent:
            dset = parent[key]
            if dset.dtype != self.dtype or dset.shape != self.shape:
                del parent[key]
            else:
                return dset
        return parent.create_dataset(
            key,
            shape=self.shape,
            dtype=self.dtype,
            maxshape=(None, None) if self.ndim == 2 else (None, None, None),
            compression="gzip",
            compression_opts=9,
        )
